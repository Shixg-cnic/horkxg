#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Options {
    std::string graph_dir;
    std::string graph_name;
    std::string indptr_path;
    std::string indices_path;
    std::string output_path;
    size_t read_chunk_entries = 1 << 22;
    size_t write_buffer_bytes = 1 << 24;
};

class BufferedWriter {
public:
    BufferedWriter(const std::string& path, size_t buffer_bytes)
        : file_(std::fopen(path.c_str(), "wb")), buffer_(buffer_bytes) {
        if (!file_) throw std::runtime_error("Cannot open output file: " + path);
    }

    ~BufferedWriter() {
        if (file_) {
            flush();
            std::fclose(file_);
        }
    }

    void write_char(char c) {
        if (pos_ == buffer_.size()) flush();
        buffer_[pos_++] = c;
    }

    void write_uint64(uint64_t value) {
        char tmp[32];
        auto [ptr, ec] = std::to_chars(tmp, tmp + sizeof(tmp), value);
        if (ec != std::errc()) throw std::runtime_error("Failed formatting integer");
        write_bytes(tmp, static_cast<size_t>(ptr - tmp));
    }

    void write_bytes(const char* data, size_t bytes) {
        while (bytes > 0) {
            if (pos_ == buffer_.size()) flush();
            const size_t room = buffer_.size() - pos_;
            const size_t take = bytes < room ? bytes : room;
            std::memcpy(buffer_.data() + pos_, data, take);
            pos_ += take;
            data += take;
            bytes -= take;
        }
    }

    void flush() {
        if (pos_ == 0) return;
        const size_t written = std::fwrite(buffer_.data(), 1, pos_, file_);
        if (written != pos_) throw std::runtime_error("Failed writing output file");
        pos_ = 0;
    }

private:
    std::FILE* file_ = nullptr;
    std::vector<char> buffer_;
    size_t pos_ = 0;
};

class Timer {
public:
    Timer() : start_(std::chrono::steady_clock::now()) {}
    double seconds() const {
        const auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start_).count();
    }
private:
    std::chrono::steady_clock::time_point start_;
};

void print_usage(const char* prog) {
    std::cout
        << "Usage:\n"
        << "  " << prog << " GRAPH_DIR [--name NAME] [--output FILE]\n"
        << "  " << prog << " --graph-dir DIR [--name NAME] [--output FILE]\n"
        << "  " << prog << " --indptr FILE --indices FILE --output FILE\n\n"
        << "Reads int64 CSR files and writes a 1-indexed undirected METIS adjacency file.\n"
        << "With --graph-dir, NAME defaults to the directory basename and files are\n"
        << "NAME_indptr.bin and NAME_indices.bin.\n\n"
        << "Every input edge u->v is treated as an undirected candidate edge {u,v}.\n"
        << "Self-loops are removed, duplicate undirected edges are deduplicated, and\n"
        << "the output adjacency is symmetric, as expected by METIS/PuLP/XtraPuLP.\n\n"
        << "Options:\n"
        << "  --read-chunk-entries N          Temporary index read buffer entries.\n"
        << "  --write-buffer-mb N             Text output buffer size in MiB.\n";
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
        } else if (a == "--symmetrize") {
            // Kept as a no-op for older scripts. Undirected symmetric METIS
            // output is now the only conversion mode.
        } else if (a == "--graph-dir") {
            opt.graph_dir = need_value(i, argc, argv, a);
        } else if (a == "--name") {
            opt.graph_name = need_value(i, argc, argv, a);
        } else if (a == "--indptr") {
            opt.indptr_path = need_value(i, argc, argv, a);
        } else if (a == "--indices") {
            opt.indices_path = need_value(i, argc, argv, a);
        } else if (a == "--output" || a == "-o") {
            opt.output_path = need_value(i, argc, argv, a);
        } else if (a == "--read-chunk-entries") {
            opt.read_chunk_entries = static_cast<size_t>(std::stoull(need_value(i, argc, argv, a)));
        } else if (a == "--write-buffer-mb") {
            opt.write_buffer_bytes = static_cast<size_t>(std::stoull(need_value(i, argc, argv, a))) << 20;
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
        if (opt.output_path.empty()) opt.output_path = (dir / (opt.graph_name + ".metis")).string();
    }

    if (opt.indptr_path.empty() || opt.indices_path.empty()) {
        throw std::runtime_error("Provide GRAPH_DIR, or both --indptr and --indices");
    }
    if (opt.output_path.empty()) throw std::runtime_error("Provide --output when using explicit CSR paths");
    if (opt.read_chunk_entries == 0) throw std::runtime_error("--read-chunk-entries must be positive");
    if (opt.write_buffer_bytes < 1024) throw std::runtime_error("--write-buffer-mb is too small");
    return opt;
}

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

void read_indices(std::ifstream& in, std::vector<int64_t>& buffer, uint64_t count, const std::string& path) {
    if (buffer.size() < count) buffer.resize(static_cast<size_t>(count));
    in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(count * sizeof(int64_t)));
    if (!in) throw std::runtime_error("Failed reading " + path);
}

void write_metis_header(BufferedWriter& out, uint64_t vertices, uint64_t undirected_edges) {
    out.write_uint64(vertices);
    out.write_char(' ');
    out.write_uint64(undirected_edges);
    out.write_char('\n');
}

uint64_t encode_edge(uint64_t u, uint64_t v, uint64_t vertices) {
    return u * vertices + v;
}

void check_encode_range(uint64_t vertices) {
    if (vertices == 0) return;
    const uint64_t max = std::numeric_limits<uint64_t>::max();
    if (vertices - 1 > max / vertices) {
        throw std::runtime_error("Too many vertices for uint64 edge-key encoding");
    }
}

void convert_to_undirected_metis(const Options& opt, uint64_t vertices, uint64_t directed_edges) {
    check_encode_range(vertices);
    Timer total_timer;

    std::ifstream indptr(opt.indptr_path, std::ios::binary);
    std::ifstream indices(opt.indices_path, std::ios::binary);
    if (!indptr) throw std::runtime_error("Cannot open " + opt.indptr_path);
    if (!indices) throw std::runtime_error("Cannot open " + opt.indices_path);

    Timer stage_timer;
    std::vector<uint64_t> edges;
    edges.reserve(static_cast<size_t>(directed_edges));
    std::vector<int64_t> edge_buffer(opt.read_chunk_entries);
    int64_t prev = read_one<int64_t>(indptr, opt.indptr_path);
    if (prev != 0) throw std::runtime_error("Invalid CSR: indptr[0] must be 0");

    uint64_t emitted_edges = 0;
    uint64_t self_loops = 0;
    for (uint64_t row = 0; row < vertices; ++row) {
        const int64_t next = read_one<int64_t>(indptr, opt.indptr_path);
        if (next < prev) throw std::runtime_error("Invalid CSR: indptr is not monotonic");
        uint64_t remaining = static_cast<uint64_t>(next - prev);
        while (remaining > 0) {
            const uint64_t take = remaining < opt.read_chunk_entries ? remaining : opt.read_chunk_entries;
            read_indices(indices, edge_buffer, take, opt.indices_path);
            for (uint64_t i = 0; i < take; ++i) {
                const int64_t raw_dst = edge_buffer[static_cast<size_t>(i)];
                if (raw_dst < 0 || static_cast<uint64_t>(raw_dst) >= vertices) {
                    throw std::runtime_error("Invalid CSR: column index out of range");
                }
                const uint64_t dst = static_cast<uint64_t>(raw_dst);
                if (dst == row) {
                    ++self_loops;
                    continue;
                }
                const uint64_t a = row < dst ? row : dst;
                const uint64_t b = row < dst ? dst : row;
                edges.push_back(encode_edge(a, b, vertices));
            }
            emitted_edges += take;
            remaining -= take;
        }
        prev = next;
    }
    if (emitted_edges != directed_edges || static_cast<uint64_t>(prev) != directed_edges) {
        throw std::runtime_error("Invalid CSR: edge count does not match indices file size");
    }
    std::cout << "collect_directed_edges_seconds: " << stage_timer.seconds() << "\n";

    stage_timer = Timer();
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    const uint64_t undirected_edges = static_cast<uint64_t>(edges.size());
    std::cout << "sort_unique_seconds: " << stage_timer.seconds() << "\n";

    stage_timer = Timer();
    std::vector<uint64_t> offsets(vertices + 1, 0);
    for (uint64_t key : edges) {
        const uint64_t u = key / vertices;
        const uint64_t v = key - u * vertices;
        ++offsets[u + 1];
        ++offsets[v + 1];
    }
    for (uint64_t i = 0; i < vertices; ++i) offsets[i + 1] += offsets[i];

    std::vector<uint64_t> cursor(offsets.begin(), offsets.end() - 1);
    std::vector<uint64_t> adjacency(offsets.back());
    for (uint64_t key : edges) {
        const uint64_t u = key / vertices;
        const uint64_t v = key - u * vertices;
        adjacency[cursor[u]++] = v;
        adjacency[cursor[v]++] = u;
    }
    cursor.clear();
    cursor.shrink_to_fit();
    std::cout << "build_symmetric_adjacency_seconds: " << stage_timer.seconds() << "\n";

    stage_timer = Timer();
    BufferedWriter out(opt.output_path, opt.write_buffer_bytes);
    write_metis_header(out, vertices, undirected_edges);
    for (uint64_t row = 0; row < vertices; ++row) {
        const uint64_t begin = offsets[row];
        const uint64_t end = offsets[row + 1];
        for (uint64_t p = begin; p < end; ++p) {
            if (p != begin) out.write_char(' ');
            out.write_uint64(adjacency[p] + 1);
        }
        out.write_char('\n');
    }
    out.flush();
    std::cout << "write_metis_seconds: " << stage_timer.seconds() << "\n"
              << "self_loops_removed: " << self_loops << "\n"
              << "undirected_edges_after_dedup: " << undirected_edges << "\n"
              << "total_seconds: " << total_timer.seconds() << "\n";
}

int main(int argc, char** argv) {
    try {
        const Options opt = parse_args(argc, argv);
        const uint64_t indptr_entries = element_count<int64_t>(opt.indptr_path);
        const uint64_t directed_edges = element_count<int64_t>(opt.indices_path);
        if (indptr_entries == 0) throw std::runtime_error("indptr is empty");
        const uint64_t vertices = indptr_entries - 1;

        std::cout << "=========== CSR to METIS ===========\n"
                  << "indptr: " << opt.indptr_path << "\n"
                  << "indices: " << opt.indices_path << "\n"
                  << "output: " << opt.output_path << "\n"
                  << "vertices: " << vertices << "\n"
                  << "directed_edges: " << directed_edges << "\n"
                  << "mode: undirected_deduplicated\n";

        convert_to_undirected_metis(opt, vertices, directed_edges);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }
    return 0;
}
