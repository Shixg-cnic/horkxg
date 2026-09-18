#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Options {
    std::string graph_dir;
    std::string graph_name;
    std::string indptr_path;
    std::string indices_path;
    std::string output_dir;
    std::string output_name;
    std::string tmp_dir;
    uint64_t memory_mb = 4096;
    uint64_t read_chunk_entries = 1 << 22;
    uint64_t merge_buffer_entries = 1 << 20;
    size_t max_open_files = 192;
    bool keep_temp = false;
};

class Timer {
public:
    Timer() : start_(std::chrono::steady_clock::now()) {}
    double seconds() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    }
private:
    std::chrono::steady_clock::time_point start_;
};

template <typename T>
uint64_t element_count(const std::string& path) {
    const auto bytes = fs::file_size(path);
    if (bytes % sizeof(T) != 0) throw std::runtime_error("File size is not aligned: " + path);
    return static_cast<uint64_t>(bytes / sizeof(T));
}

template <typename T>
T read_one(std::ifstream& in, const std::string& path) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("Failed reading " + path);
    return value;
}

template <typename T>
void write_vector_binary(const std::string& path, const std::vector<T>& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open output file: " + path);
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
    if (!out) throw std::runtime_error("Failed writing " + path);
}

template <typename T>
class BufferedBinaryWriter {
public:
    BufferedBinaryWriter(const std::string& path, size_t buffer_entries)
        : out_(path, std::ios::binary), buffer_(buffer_entries) {
        if (!out_) throw std::runtime_error("Cannot open output file: " + path);
        if (buffer_.empty()) buffer_.resize(1);
    }

    ~BufferedBinaryWriter() {
        flush();
    }

    void write(T value) {
        if (pos_ == buffer_.size()) flush();
        buffer_[pos_++] = value;
    }

    void flush() {
        if (pos_ == 0) return;
        out_.write(reinterpret_cast<const char*>(buffer_.data()), static_cast<std::streamsize>(pos_ * sizeof(T)));
        if (!out_) throw std::runtime_error("Failed writing buffered binary output");
        pos_ = 0;
    }

private:
    std::ofstream out_;
    std::vector<T> buffer_;
    size_t pos_ = 0;
};

class Int64Stream {
public:
    Int64Stream(const std::string& path, uint64_t buffer_entries)
        : path_(path), in_(path, std::ios::binary), buffer_(static_cast<size_t>(buffer_entries)) {
        if (!in_) throw std::runtime_error("Cannot open " + path);
        if (buffer_.empty()) buffer_.resize(1);
    }

    int64_t next() {
        if (pos_ == end_) refill();
        return buffer_[pos_++];
    }

private:
    void refill() {
        in_.read(reinterpret_cast<char*>(buffer_.data()), static_cast<std::streamsize>(buffer_.size() * sizeof(int64_t)));
        const auto bytes = in_.gcount();
        if (bytes <= 0 || bytes % static_cast<std::streamsize>(sizeof(int64_t)) != 0) {
            throw std::runtime_error("Failed reading " + path_);
        }
        pos_ = 0;
        end_ = static_cast<size_t>(bytes / sizeof(int64_t));
    }

    std::string path_;
    std::ifstream in_;
    std::vector<int64_t> buffer_;
    size_t pos_ = 0;
    size_t end_ = 0;
};

class UInt64RunReader {
public:
    UInt64RunReader(const std::string& path, uint64_t buffer_entries)
        : path_(path), in_(path, std::ios::binary), buffer_(static_cast<size_t>(buffer_entries)) {
        if (!in_) throw std::runtime_error("Cannot open run file: " + path);
        if (buffer_.empty()) buffer_.resize(1);
    }

    bool next(uint64_t& value) {
        if (pos_ == end_ && !refill()) return false;
        value = buffer_[pos_++];
        return true;
    }

private:
    bool refill() {
        in_.read(reinterpret_cast<char*>(buffer_.data()), static_cast<std::streamsize>(buffer_.size() * sizeof(uint64_t)));
        const auto bytes = in_.gcount();
        if (bytes == 0) return false;
        if (bytes % static_cast<std::streamsize>(sizeof(uint64_t)) != 0) {
            throw std::runtime_error("Run file has partial uint64 entry: " + path_);
        }
        pos_ = 0;
        end_ = static_cast<size_t>(bytes / sizeof(uint64_t));
        return true;
    }

    std::string path_;
    std::ifstream in_;
    std::vector<uint64_t> buffer_;
    size_t pos_ = 0;
    size_t end_ = 0;
};

void print_usage(const char* prog) {
    std::cout
        << "Usage:\n"
        << "  " << prog << " GRAPH_DIR [--name NAME] [--output-dir DIR] [--output-name NAME]\n"
        << "  " << prog << " --indptr FILE --indices FILE --output-dir DIR --output-name NAME\n\n"
        << "Converts a directed or asymmetric int64 CSR graph into an unweighted symmetric\n"
        << "undirected CSR graph. Self-loops are removed and duplicate undirected edges are\n"
        << "deduplicated. Output files are NAME_indptr.bin and NAME_indices.bin.\n"
        << "For GRAPH_DIR=/path/dataset/NAME, the default output directory is\n"
        << "/path/dataset/process_data/NAME.\n\n"
        << "Large graphs are processed with external sort runs controlled by --memory-mb.\n\n"
        << "Options:\n"
        << "  --memory-mb N              Approximate sort chunk memory. Default: 4096.\n"
        << "  --tmp-dir DIR              Temporary run directory. Default: OUTPUT_DIR/.csr_symmetrize_tmp.\n"
        << "  --read-chunk-entries N     Sequential CSR index read buffer. Default: 4194304.\n"
        << "  --merge-buffer-entries N   Per-run merge read buffer. Default: 1048576.\n"
        << "  --max-open-files N         Merge fan-in limit. Default: 192.\n"
        << "  --keep-temp                Keep temporary files for debugging.\n";
}

std::string need_value(int& i, int argc, char** argv, const std::string& name) {
    if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
    return argv[++i];
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (a == "--graph-dir") {
            opt.graph_dir = need_value(i, argc, argv, a);
        } else if (a == "--name") {
            opt.graph_name = need_value(i, argc, argv, a);
        } else if (a == "--indptr") {
            opt.indptr_path = need_value(i, argc, argv, a);
        } else if (a == "--indices") {
            opt.indices_path = need_value(i, argc, argv, a);
        } else if (a == "--output-dir") {
            opt.output_dir = need_value(i, argc, argv, a);
        } else if (a == "--output-name") {
            opt.output_name = need_value(i, argc, argv, a);
        } else if (a == "--tmp-dir") {
            opt.tmp_dir = need_value(i, argc, argv, a);
        } else if (a == "--memory-mb") {
            opt.memory_mb = std::stoull(need_value(i, argc, argv, a));
        } else if (a == "--read-chunk-entries") {
            opt.read_chunk_entries = std::stoull(need_value(i, argc, argv, a));
        } else if (a == "--merge-buffer-entries") {
            opt.merge_buffer_entries = std::stoull(need_value(i, argc, argv, a));
        } else if (a == "--max-open-files") {
            opt.max_open_files = static_cast<size_t>(std::stoull(need_value(i, argc, argv, a)));
        } else if (a == "--keep-temp") {
            opt.keep_temp = true;
        } else if (!a.empty() && a[0] != '-') {
            if (!opt.graph_dir.empty()) throw std::runtime_error("Only one GRAPH_DIR positional argument is allowed");
            opt.graph_dir = a;
        } else {
            throw std::runtime_error("Unknown argument: " + a);
        }
    }

    if (!opt.graph_dir.empty()) {
        fs::path dir(opt.graph_dir);
        if (opt.graph_name.empty()) opt.graph_name = dir.filename().string();
        if (opt.indptr_path.empty()) opt.indptr_path = (dir / (opt.graph_name + "_indptr.bin")).string();
        if (opt.indices_path.empty()) opt.indices_path = (dir / (opt.graph_name + "_indices.bin")).string();
        if (opt.output_dir.empty()) {
            opt.output_dir =
                (dir.parent_path() / "process_data" / opt.graph_name).string();
        }
        if (opt.output_name.empty()) opt.output_name = opt.graph_name;
    }

    if (opt.indptr_path.empty() || opt.indices_path.empty()) {
        throw std::runtime_error("Provide GRAPH_DIR, or both --indptr and --indices");
    }
    if (opt.output_dir.empty()) throw std::runtime_error("Provide --output-dir");
    if (opt.output_name.empty()) throw std::runtime_error("Provide --output-name when using explicit CSR paths");
    if (opt.memory_mb == 0) throw std::runtime_error("--memory-mb must be positive");
    if (opt.read_chunk_entries == 0) throw std::runtime_error("--read-chunk-entries must be positive");
    if (opt.merge_buffer_entries == 0) throw std::runtime_error("--merge-buffer-entries must be positive");
    if (opt.max_open_files < 2) throw std::runtime_error("--max-open-files must be at least 2");
    if (opt.tmp_dir.empty()) opt.tmp_dir = (fs::path(opt.output_dir) / ".csr_symmetrize_tmp").string();
    return opt;
}

void check_encode_range(uint64_t vertices) {
    if (vertices == 0) return;
    const uint64_t max = std::numeric_limits<uint64_t>::max();
    if (vertices - 1 > max / vertices) {
        throw std::runtime_error("Too many vertices for uint64 edge-key encoding");
    }
}

uint64_t encode_pair(uint64_t src, uint64_t dst, uint64_t vertices) {
    return src * vertices + dst;
}

std::pair<uint64_t, uint64_t> decode_pair(uint64_t key, uint64_t vertices) {
    const uint64_t src = key / vertices;
    return {src, key - src * vertices};
}

std::string run_path(const fs::path& tmp_dir, const std::string& prefix, size_t id) {
    return (tmp_dir / (prefix + "_" + std::to_string(id) + ".bin")).string();
}

void write_sorted_unique_run(std::vector<uint64_t>& keys, const std::string& path) {
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    write_vector_binary(path, keys);
    keys.clear();
}

std::vector<std::string> make_undirected_runs(
    const Options& opt,
    uint64_t vertices,
    uint64_t directed_edges,
    uint64_t& self_loops
) {
    Timer timer;
    const uint64_t chunk_keys = std::max<uint64_t>(1, (opt.memory_mb << 20) / sizeof(uint64_t));
    std::vector<uint64_t> keys;
    keys.reserve(static_cast<size_t>(std::min<uint64_t>(chunk_keys, static_cast<uint64_t>(std::numeric_limits<size_t>::max()))));
    std::vector<std::string> runs;

    std::ifstream indptr(opt.indptr_path, std::ios::binary);
    if (!indptr) throw std::runtime_error("Cannot open " + opt.indptr_path);
    Int64Stream indices(opt.indices_path, opt.read_chunk_entries);

    int64_t prev = read_one<int64_t>(indptr, opt.indptr_path);
    if (prev != 0) throw std::runtime_error("Invalid CSR: indptr[0] must be 0");
    uint64_t seen_edges = 0;
    self_loops = 0;

    for (uint64_t row = 0; row < vertices; ++row) {
        const int64_t next = read_one<int64_t>(indptr, opt.indptr_path);
        if (next < prev) throw std::runtime_error("Invalid CSR: indptr is not monotonic");
        for (int64_t e = prev; e < next; ++e) {
            const int64_t raw_dst = indices.next();
            if (raw_dst < 0 || static_cast<uint64_t>(raw_dst) >= vertices) {
                throw std::runtime_error("Invalid CSR: column index out of range");
            }
            const uint64_t dst = static_cast<uint64_t>(raw_dst);
            if (dst == row) {
                ++self_loops;
            } else {
                const uint64_t u = std::min(row, dst);
                const uint64_t v = std::max(row, dst);
                keys.push_back(encode_pair(u, v, vertices));
                if (keys.size() == chunk_keys) {
                    const std::string path = run_path(opt.tmp_dir, "undir_run", runs.size());
                    write_sorted_unique_run(keys, path);
                    runs.push_back(path);
                    std::cout << "wrote_undirected_run: " << path << "\n";
                }
            }
            ++seen_edges;
        }
        prev = next;
    }
    if (seen_edges != directed_edges || static_cast<uint64_t>(prev) != directed_edges) {
        throw std::runtime_error("Invalid CSR: edge count does not match indices file size");
    }
    if (!keys.empty()) {
        const std::string path = run_path(opt.tmp_dir, "undir_run", runs.size());
        write_sorted_unique_run(keys, path);
        runs.push_back(path);
        std::cout << "wrote_undirected_run: " << path << "\n";
    }

    std::cout << "make_undirected_runs_seconds: " << timer.seconds() << "\n";
    return runs;
}

struct HeapItem {
    uint64_t key;
    size_t reader;
    bool operator>(const HeapItem& other) const {
        return key > other.key;
    }
};

uint64_t merge_runs_unique(
    const std::vector<std::string>& inputs,
    const std::string& output,
    uint64_t merge_buffer_entries
) {
    std::vector<UInt64RunReader> readers;
    readers.reserve(inputs.size());
    for (const auto& path : inputs) readers.emplace_back(path, merge_buffer_entries);

    std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>> heap;
    for (size_t i = 0; i < readers.size(); ++i) {
        uint64_t key = 0;
        if (readers[i].next(key)) heap.push({key, i});
    }

    BufferedBinaryWriter<uint64_t> writer(output, merge_buffer_entries);
    bool have_last = false;
    uint64_t last = 0;
    uint64_t unique_count = 0;
    while (!heap.empty()) {
        const auto item = heap.top();
        heap.pop();
        if (!have_last || item.key != last) {
            writer.write(item.key);
            last = item.key;
            have_last = true;
            ++unique_count;
        }
        uint64_t next = 0;
        if (readers[item.reader].next(next)) heap.push({next, item.reader});
    }
    return unique_count;
}

std::vector<std::string> reduce_run_count(
    const Options& opt,
    std::vector<std::string> runs,
    const std::string& prefix
) {
    size_t pass = 0;
    while (runs.size() > opt.max_open_files) {
        Timer timer;
        std::vector<std::string> next_runs;
        for (size_t begin = 0; begin < runs.size(); begin += opt.max_open_files) {
            const size_t end = std::min(begin + opt.max_open_files, runs.size());
            std::vector<std::string> group(runs.begin() + static_cast<std::ptrdiff_t>(begin),
                                           runs.begin() + static_cast<std::ptrdiff_t>(end));
            const std::string out = run_path(opt.tmp_dir, prefix + "_pass" + std::to_string(pass), next_runs.size());
            merge_runs_unique(group, out, opt.merge_buffer_entries);
            next_runs.push_back(out);
            for (const auto& path : group) fs::remove(path);
        }
        runs.swap(next_runs);
        std::cout << "merge_pass_" << pass << "_runs: " << runs.size()
                  << " seconds: " << timer.seconds() << "\n";
        ++pass;
    }
    return runs;
}

uint64_t final_merge_undirected(
    const Options& opt,
    const std::vector<std::string>& runs,
    const std::string& unique_edges_path,
    uint64_t vertices,
    std::vector<uint64_t>& degrees
) {
    Timer timer;
    std::vector<UInt64RunReader> readers;
    readers.reserve(runs.size());
    for (const auto& path : runs) readers.emplace_back(path, opt.merge_buffer_entries);

    std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>> heap;
    for (size_t i = 0; i < readers.size(); ++i) {
        uint64_t key = 0;
        if (readers[i].next(key)) heap.push({key, i});
    }

    BufferedBinaryWriter<uint64_t> unique_writer(unique_edges_path, opt.merge_buffer_entries);
    bool have_last = false;
    uint64_t last = 0;
    uint64_t unique_count = 0;
    while (!heap.empty()) {
        const auto item = heap.top();
        heap.pop();
        if (!have_last || item.key != last) {
            const auto [u, v] = decode_pair(item.key, vertices);
            ++degrees[static_cast<size_t>(u)];
            ++degrees[static_cast<size_t>(v)];
            unique_writer.write(item.key);
            last = item.key;
            have_last = true;
            ++unique_count;
        }
        uint64_t next = 0;
        if (readers[item.reader].next(next)) heap.push({next, item.reader});
    }
    unique_writer.flush();
    std::cout << "final_merge_undirected_seconds: " << timer.seconds() << "\n";
    return unique_count;
}

std::vector<int64_t> make_indptr(const std::vector<uint64_t>& degrees) {
    std::vector<int64_t> indptr(degrees.size() + 1, 0);
    uint64_t prefix = 0;
    for (size_t i = 0; i < degrees.size(); ++i) {
        if (degrees[i] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - prefix) {
            throw std::runtime_error("Symmetric CSR edge count exceeds int64 range");
        }
        prefix += degrees[i];
        indptr[i + 1] = static_cast<int64_t>(prefix);
    }
    return indptr;
}

std::vector<std::string> make_directed_runs(
    const Options& opt,
    const std::string& unique_edges_path,
    uint64_t vertices
) {
    Timer timer;
    const uint64_t chunk_keys = std::max<uint64_t>(2, (opt.memory_mb << 20) / sizeof(uint64_t));
    std::vector<uint64_t> keys;
    keys.reserve(static_cast<size_t>(std::min<uint64_t>(chunk_keys, static_cast<uint64_t>(std::numeric_limits<size_t>::max()))));
    std::vector<std::string> runs;
    UInt64RunReader reader(unique_edges_path, opt.merge_buffer_entries);

    uint64_t key = 0;
    while (reader.next(key)) {
        const auto [u, v] = decode_pair(key, vertices);
        keys.push_back(encode_pair(u, v, vertices));
        keys.push_back(encode_pair(v, u, vertices));
        if (keys.size() >= chunk_keys) {
            const std::string path = run_path(opt.tmp_dir, "directed_run", runs.size());
            std::sort(keys.begin(), keys.end());
            write_vector_binary(path, keys);
            keys.clear();
            runs.push_back(path);
            std::cout << "wrote_directed_run: " << path << "\n";
        }
    }
    if (!keys.empty()) {
        const std::string path = run_path(opt.tmp_dir, "directed_run", runs.size());
        std::sort(keys.begin(), keys.end());
        write_vector_binary(path, keys);
        runs.push_back(path);
        std::cout << "wrote_directed_run: " << path << "\n";
    }
    std::cout << "make_directed_runs_seconds: " << timer.seconds() << "\n";
    return runs;
}

uint64_t merge_directed_runs_to_indices(
    const Options& opt,
    std::vector<std::string> runs,
    const std::string& output_indices_path,
    uint64_t vertices
) {
    Timer timer;
    runs = reduce_run_count(opt, std::move(runs), "directed_merge");

    std::vector<UInt64RunReader> readers;
    readers.reserve(runs.size());
    for (const auto& path : runs) readers.emplace_back(path, opt.merge_buffer_entries);

    std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>> heap;
    for (size_t i = 0; i < readers.size(); ++i) {
        uint64_t key = 0;
        if (readers[i].next(key)) heap.push({key, i});
    }

    BufferedBinaryWriter<int64_t> writer(output_indices_path, opt.merge_buffer_entries);
    uint64_t count = 0;
    while (!heap.empty()) {
        const auto item = heap.top();
        heap.pop();
        const auto [src, dst] = decode_pair(item.key, vertices);
        (void)src;
        if (dst > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::runtime_error("Vertex id exceeds int64 range");
        }
        writer.write(static_cast<int64_t>(dst));
        ++count;
        uint64_t next = 0;
        if (readers[item.reader].next(next)) heap.push({next, item.reader});
    }
    writer.flush();

    for (const auto& path : runs) fs::remove(path);
    std::cout << "merge_directed_runs_to_indices_seconds: " << timer.seconds() << "\n";
    return count;
}

int main(int argc, char** argv) {
    try {
        const Options opt = parse_args(argc, argv);
        Timer total_timer;
        fs::create_directories(opt.output_dir);
        fs::remove_all(opt.tmp_dir);
        fs::create_directories(opt.tmp_dir);

        const uint64_t indptr_entries = element_count<int64_t>(opt.indptr_path);
        const uint64_t directed_edges = element_count<int64_t>(opt.indices_path);
        if (indptr_entries == 0) throw std::runtime_error("indptr is empty");
        const uint64_t vertices = indptr_entries - 1;
        check_encode_range(vertices);

        std::cout << "=========== CSR Symmetrize ===========\n"
                  << "vertices: " << vertices << "\n"
                  << "input_directed_edges: " << directed_edges << "\n"
                  << "memory_mb: " << opt.memory_mb << "\n"
                  << "tmp_dir: " << opt.tmp_dir << "\n";

        uint64_t self_loops = 0;
        auto undir_runs = make_undirected_runs(opt, vertices, directed_edges, self_loops);
        undir_runs = reduce_run_count(opt, std::move(undir_runs), "undir_merge");

        const std::string unique_edges_path = (fs::path(opt.tmp_dir) / "unique_undirected_edges.bin").string();
        std::vector<uint64_t> degrees(static_cast<size_t>(vertices), 0);
        const uint64_t unique_undirected_edges = final_merge_undirected(
            opt, undir_runs, unique_edges_path, vertices, degrees);
        for (const auto& path : undir_runs) fs::remove(path);

        const auto indptr = make_indptr(degrees);
        const uint64_t output_directed_edges = static_cast<uint64_t>(indptr.back());
        if (output_directed_edges != unique_undirected_edges * 2) {
            throw std::runtime_error("Internal error: degree sum does not equal 2 * unique edges");
        }

        const fs::path out_dir(opt.output_dir);
        const std::string out_indptr = (out_dir / (opt.output_name + "_indptr.bin")).string();
        const std::string out_indices = (out_dir / (opt.output_name + "_indices.bin")).string();
        write_vector_binary(out_indptr, indptr);

        auto directed_runs = make_directed_runs(opt, unique_edges_path, vertices);
        const uint64_t written_indices = merge_directed_runs_to_indices(
            opt, std::move(directed_runs), out_indices, vertices);
        if (written_indices != output_directed_edges) {
            throw std::runtime_error("Internal error: written indices count does not match indptr.back()");
        }

        if (!opt.keep_temp) fs::remove_all(opt.tmp_dir);

        std::cout << "self_loops_removed: " << self_loops << "\n"
                  << "unique_undirected_edges: " << unique_undirected_edges << "\n"
                  << "output_directed_edges: " << output_directed_edges << "\n"
                  << "output_indptr: " << out_indptr << "\n"
                  << "output_indices: " << out_indices << "\n"
                  << "total_seconds: " << total_timer.seconds() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
