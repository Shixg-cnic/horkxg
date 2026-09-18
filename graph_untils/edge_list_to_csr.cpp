#include <zlib.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

struct Options {
    std::string input;
    std::string output_dir;
    std::string output_name;
    std::string format = "auto";
    int index_base = -1;
    uint64_t vertices = 0;
    bool keep_self_loops = false;
    bool remap_ids = false;
};

class Timer {
public:
    Timer() : start_(std::chrono::steady_clock::now()) {}
    double seconds() const {
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - start_)
            .count();
    }

private:
    std::chrono::steady_clock::time_point start_;
};

void usage(const char* prog) {
    std::cout
        << "Usage:\n"
        << "  " << prog << " INPUT --output-dir DIR --output-name NAME [options]\n\n"
        << "Converts a SNAP edge list or Matrix Market coordinate file to int64 CSR.\n"
        << "Plain text and gzip-compressed inputs are supported. The input edge direction\n"
        << "is preserved; symmetric Matrix Market entries are not expanded.\n\n"
        << "Options:\n"
        << "  --vertices N          Expected vertex count; recommended for SNAP files.\n"
        << "  --format F            auto, snap, or matrix-market. Default: auto.\n"
        << "  --index-base N        auto, 0, or 1. Default: Matrix Market=1, SNAP=0.\n"
        << "  --remap-ids           Densely remap non-contiguous vertex identifiers.\n"
        << "  --keep-self-loops     Preserve self-loop entries. Default: remove them.\n";
}

std::string value_after(int& i, int argc, char** argv, const std::string& arg) {
    if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
    return argv[++i];
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else if (arg == "--output-dir") {
            opt.output_dir = value_after(i, argc, argv, arg);
        } else if (arg == "--output-name") {
            opt.output_name = value_after(i, argc, argv, arg);
        } else if (arg == "--vertices") {
            opt.vertices = std::stoull(value_after(i, argc, argv, arg));
        } else if (arg == "--format") {
            opt.format = value_after(i, argc, argv, arg);
        } else if (arg == "--index-base") {
            const std::string value = value_after(i, argc, argv, arg);
            opt.index_base = value == "auto" ? -1 : std::stoi(value);
        } else if (arg == "--keep-self-loops") {
            opt.keep_self_loops = true;
        } else if (arg == "--remap-ids") {
            opt.remap_ids = true;
        } else if (!arg.empty() && arg[0] != '-') {
            if (!opt.input.empty()) throw std::runtime_error("Only one input file is allowed");
            opt.input = arg;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (opt.input.empty()) throw std::runtime_error("Missing INPUT");
    if (opt.output_dir.empty()) throw std::runtime_error("Missing --output-dir");
    if (opt.output_name.empty()) throw std::runtime_error("Missing --output-name");
    if (opt.format != "auto" && opt.format != "snap" &&
        opt.format != "matrix-market") {
        throw std::runtime_error("--format must be auto, snap, or matrix-market");
    }
    if (opt.index_base < -1 || opt.index_base > 1) {
        throw std::runtime_error("--index-base must be auto, 0, or 1");
    }
    return opt;
}

bool ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

class PlainReader {
public:
    explicit PlainReader(const std::string& path)
        : file_(std::fopen(path.c_str(), "rb")), buffer_(8U << 20) {
        if (!file_) throw std::runtime_error("Cannot open input: " + path);
    }
    ~PlainReader() { std::fclose(file_); }

    int get() {
        if (pos_ == size_) {
            size_ = std::fread(buffer_.data(), 1, buffer_.size(), file_);
            pos_ = 0;
            if (size_ == 0) return -1;
        }
        return static_cast<unsigned char>(buffer_[pos_++]);
    }

private:
    FILE* file_;
    std::vector<char> buffer_;
    size_t pos_ = 0;
    size_t size_ = 0;
};

class GzipReader {
public:
    explicit GzipReader(const std::string& path)
        : file_(gzopen(path.c_str(), "rb")), buffer_(8U << 20) {
        if (!file_) throw std::runtime_error("Cannot open gzip input: " + path);
    }
    ~GzipReader() { gzclose(file_); }

    int get() {
        if (pos_ == size_) {
            const int count = gzread(file_, buffer_.data(),
                                     static_cast<unsigned int>(buffer_.size()));
            if (count < 0) {
                int code = Z_OK;
                const char* message = gzerror(file_, &code);
                throw std::runtime_error(std::string("gzip read failed: ") + message);
            }
            pos_ = 0;
            size_ = static_cast<size_t>(count);
            if (size_ == 0) return -1;
        }
        return static_cast<unsigned char>(buffer_[pos_++]);
    }

private:
    gzFile file_;
    std::vector<char> buffer_;
    size_t pos_ = 0;
    size_t size_ = 0;
};

template <typename Reader>
class NumberLineScanner {
public:
    explicit NumberLineScanner(const std::string& path) : reader_(path) {}

    bool next(std::array<uint64_t, 3>& values, int& count) {
        count = 0;
        int c = next_nonspace();
        while (c >= 0) {
            if (c == '#' || c == '%') {
                skip_line(c);
                c = next_nonspace();
                continue;
            }
            while (c >= 0 && c != '\n' && c != '\r') {
                if (c >= '0' && c <= '9') {
                    uint64_t value = 0;
                    do {
                        const uint64_t digit = static_cast<uint64_t>(c - '0');
                        if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
                            throw std::runtime_error("Integer overflow while parsing input");
                        }
                        value = value * 10 + digit;
                        c = reader_.get();
                    } while (c >= '0' && c <= '9');
                    if (count < 3) values[static_cast<size_t>(count)] = value;
                    ++count;
                } else if (c == '-') {
                    throw std::runtime_error("Negative vertex identifiers are unsupported");
                } else {
                    c = reader_.get();
                }
            }
            if (count >= 2) return true;
            count = 0;
            c = next_nonspace();
        }
        return false;
    }

private:
    int next_nonspace() {
        int c = reader_.get();
        while (c == ' ' || c == '\t' || c == '\n' || c == '\r') c = reader_.get();
        return c;
    }

    void skip_line(int c) {
        while (c >= 0 && c != '\n' && c != '\r') c = reader_.get();
    }

    Reader reader_;
};

std::string read_prefix(const std::string& path) {
    std::string result(64, '\0');
    size_t count = 0;
    if (ends_with(path, ".gz")) {
        gzFile file = gzopen(path.c_str(), "rb");
        if (!file) throw std::runtime_error("Cannot open input: " + path);
        const int n = gzread(file, result.data(), static_cast<unsigned int>(result.size()));
        gzclose(file);
        if (n < 0) throw std::runtime_error("Cannot read gzip prefix: " + path);
        count = static_cast<size_t>(n);
    } else {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open input: " + path);
        in.read(result.data(), static_cast<std::streamsize>(result.size()));
        count = static_cast<size_t>(in.gcount());
    }
    result.resize(count);
    return result;
}

template <typename Callback>
void scan_records(const Options& opt, const std::string& format, Callback callback,
                  uint64_t& declared_vertices, uint64_t& declared_entries) {
    auto scan = [&](auto& scanner) {
        std::array<uint64_t, 3> values{};
        int count = 0;
        if (format == "matrix-market") {
            if (!scanner.next(values, count) || count < 3) {
                throw std::runtime_error("Missing Matrix Market size row");
            }
            if (values[0] != values[1]) {
                throw std::runtime_error("Only square Matrix Market graphs are supported");
            }
            declared_vertices = values[0];
            declared_entries = values[2];
        }
        while (scanner.next(values, count)) callback(values[0], values[1]);
    };

    if (ends_with(opt.input, ".gz")) {
        NumberLineScanner<GzipReader> scanner(opt.input);
        scan(scanner);
    } else {
        NumberLineScanner<PlainReader> scanner(opt.input);
        scan(scanner);
    }
}

struct MappedIndices {
    int fd = -1;
    int64_t* data = nullptr;
    size_t bytes = 0;

    MappedIndices(const std::string& path, uint64_t entries) {
        if (entries > std::numeric_limits<size_t>::max() / sizeof(int64_t)) {
            throw std::runtime_error("Output indices file is too large for this process");
        }
        bytes = static_cast<size_t>(entries * sizeof(int64_t));
        fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) throw std::runtime_error("Cannot create output: " + path);
        if (ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
            throw std::runtime_error("Failed sizing output: " + path);
        }
        if (bytes != 0) {
            void* ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (ptr == MAP_FAILED) throw std::runtime_error("mmap failed for output: " + path);
            data = static_cast<int64_t*>(ptr);
            madvise(data, bytes, MADV_SEQUENTIAL);
        }
    }

    ~MappedIndices() {
        if (data) {
            msync(data, bytes, MS_SYNC);
            munmap(data, bytes);
        }
        if (fd >= 0) {
            fsync(fd);
            close(fd);
        }
    }
};

void write_indptr(const std::string& path, const std::vector<int64_t>& indptr) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot create output: " + path);
    out.write(reinterpret_cast<const char*>(indptr.data()),
              static_cast<std::streamsize>(indptr.size() * sizeof(int64_t)));
    if (!out) throw std::runtime_error("Failed writing output: " + path);
}

void write_vertex_ids(const std::string& path, const std::vector<int64_t>& ids) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot create output: " + path);
    out.write(reinterpret_cast<const char*>(ids.data()),
              static_cast<std::streamsize>(ids.size() * sizeof(int64_t)));
    if (!out) throw std::runtime_error("Failed writing output: " + path);
}

int main(int argc, char** argv) {
    try {
        const Options opt = parse_args(argc, argv);
        std::string format = opt.format;
        if (format == "auto") {
            const std::string prefix = read_prefix(opt.input);
            format = prefix.rfind("%%MatrixMarket", 0) == 0 ? "matrix-market" : "snap";
        }
        const int index_base = opt.index_base >= 0
                                   ? opt.index_base
                                   : (format == "matrix-market" ? 1 : 0);

        fs::create_directories(opt.output_dir);
        const std::string indptr_path =
            (fs::path(opt.output_dir) / (opt.output_name + "_indptr.bin")).string();
        const std::string indices_path =
            (fs::path(opt.output_dir) / (opt.output_name + "_indices.bin")).string();
        const std::string vertex_ids_path =
            (fs::path(opt.output_dir) / (opt.output_name + "_vertex_ids.bin")).string();

        std::cout << "=========== Edge List To CSR ===========\n"
                  << "input: " << opt.input << '\n'
                  << "format: " << format << '\n'
                  << "index_base: " << index_base << '\n'
                  << "remap_ids: " << (opt.remap_ids ? "yes" : "no") << '\n';

        Timer total_timer;
        Timer pass_timer;
        uint64_t declared_vertices = 0;
        uint64_t declared_entries = 0;
        const uint64_t expected_vertices = opt.vertices;
        uint64_t vertices = opt.remap_ids ? 0 : opt.vertices;
        std::vector<uint64_t> degree;
        if (vertices != 0) degree.resize(static_cast<size_t>(vertices), 0);
        std::unordered_map<uint64_t, int64_t> id_map;
        std::vector<int64_t> original_ids;
        if (opt.remap_ids && expected_vertices != 0) {
            id_map.reserve(static_cast<size_t>(expected_vertices * 1.15));
            original_ids.reserve(static_cast<size_t>(expected_vertices));
        }
        uint64_t input_entries = 0;
        uint64_t output_entries = 0;
        uint64_t self_loops = 0;
        uint64_t max_id = 0;

        auto normalize = [&](uint64_t raw) -> uint64_t {
            if (raw < static_cast<uint64_t>(index_base)) {
                throw std::runtime_error("Vertex id is smaller than the selected index base");
            }
            return raw - static_cast<uint64_t>(index_base);
        };

        auto map_id = [&](uint64_t id) -> uint64_t {
            if (!opt.remap_ids) return id;
            auto [it, inserted] =
                id_map.emplace(id, static_cast<int64_t>(id_map.size()));
            if (inserted) {
                if (id > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                    throw std::runtime_error("Original vertex id exceeds int64 range");
                }
                original_ids.push_back(static_cast<int64_t>(id));
                degree.push_back(0);
            }
            return static_cast<uint64_t>(it->second);
        };

        scan_records(opt, format,
                     [&](uint64_t raw_src, uint64_t raw_dst) {
                         const uint64_t original_src = normalize(raw_src);
                         const uint64_t original_dst = normalize(raw_dst);
                         ++input_entries;
                         max_id = std::max(max_id, std::max(original_src, original_dst));
                         if (!opt.remap_ids && vertices != 0 &&
                             (original_src >= vertices || original_dst >= vertices)) {
                             throw std::runtime_error("Vertex id exceeds --vertices");
                         }
                         if (!opt.remap_ids && vertices == 0 && max_id >= degree.size()) {
                             degree.resize(static_cast<size_t>(max_id + 1), 0);
                         }
                         const uint64_t src = map_id(original_src);
                         const uint64_t dst = map_id(original_dst);
                         if (!opt.keep_self_loops && src == dst) {
                             ++self_loops;
                             return;
                         }
                         ++degree[static_cast<size_t>(src)];
                         ++output_entries;
                         if (input_entries % 100000000ULL == 0) {
                             std::cout << "pass1_entries: " << input_entries << '\n';
                         }
                     },
                     declared_vertices, declared_entries);

        if (opt.remap_ids) {
            vertices = id_map.size();
            if (expected_vertices != 0 && vertices != expected_vertices) {
                throw std::runtime_error(
                    "The number of distinct vertex ids differs from --vertices");
            }
            write_vertex_ids(vertex_ids_path, original_ids);
        } else if (declared_vertices != 0) {
            if (vertices != 0 && vertices != declared_vertices) {
                throw std::runtime_error("--vertices differs from Matrix Market dimensions");
            }
            vertices = declared_vertices;
            if (degree.size() != vertices) degree.resize(static_cast<size_t>(vertices), 0);
        } else if (vertices == 0) {
            vertices = degree.size();
        }
        if (declared_entries != 0 && input_entries != declared_entries) {
            throw std::runtime_error("Matrix Market entry count does not match its header");
        }
        if (vertices > static_cast<uint64_t>(std::numeric_limits<size_t>::max() - 1)) {
            throw std::runtime_error("Vertex count is too large for this process");
        }

        std::vector<int64_t> indptr(static_cast<size_t>(vertices + 1), 0);
        for (uint64_t u = 0; u < vertices; ++u) {
            const uint64_t next = static_cast<uint64_t>(indptr[static_cast<size_t>(u)]) +
                                  degree[static_cast<size_t>(u)];
            if (next > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                throw std::runtime_error("CSR edge count exceeds int64 range");
            }
            indptr[static_cast<size_t>(u + 1)] = static_cast<int64_t>(next);
        }
        degree.clear();
        degree.shrink_to_fit();
        write_indptr(indptr_path, indptr);
        std::cout << "pass1_seconds: " << pass_timer.seconds() << '\n';

        pass_timer = Timer();
        std::vector<int64_t> cursor(indptr.begin(), indptr.end() - 1);
        MappedIndices indices(indices_path, output_entries);
        uint64_t second_input_entries = 0;
        uint64_t written_entries = 0;
        uint64_t ignored_declared_vertices = 0;
        uint64_t ignored_declared_entries = 0;
        scan_records(opt, format,
                     [&](uint64_t raw_src, uint64_t raw_dst) {
                         const uint64_t original_src = normalize(raw_src);
                         const uint64_t original_dst = normalize(raw_dst);
                         uint64_t src = original_src;
                         uint64_t dst = original_dst;
                         if (opt.remap_ids) {
                             const auto src_it = id_map.find(original_src);
                             const auto dst_it = id_map.find(original_dst);
                             if (src_it == id_map.end() || dst_it == id_map.end()) {
                                 throw std::runtime_error("Input changed between conversion passes");
                             }
                             src = static_cast<uint64_t>(src_it->second);
                             dst = static_cast<uint64_t>(dst_it->second);
                         }
                         ++second_input_entries;
                         if (!opt.keep_self_loops && src == dst) return;
                         const int64_t pos = cursor[static_cast<size_t>(src)]++;
                         indices.data[static_cast<size_t>(pos)] = static_cast<int64_t>(dst);
                         ++written_entries;
                         if (second_input_entries % 100000000ULL == 0) {
                             std::cout << "pass2_entries: " << second_input_entries << '\n';
                         }
                     },
                     ignored_declared_vertices, ignored_declared_entries);

        if (second_input_entries != input_entries || written_entries != output_entries) {
            throw std::runtime_error("Input changed between conversion passes");
        }
        for (uint64_t u = 0; u < vertices; ++u) {
            if (cursor[static_cast<size_t>(u)] != indptr[static_cast<size_t>(u + 1)]) {
                throw std::runtime_error("CSR row fill count mismatch");
            }
        }

        std::cout << "pass2_seconds: " << pass_timer.seconds() << '\n'
                  << "vertices: " << vertices << '\n'
                  << "input_entries: " << input_entries << '\n'
                  << "self_loops_removed: " << self_loops << '\n'
                  << "output_directed_edges: " << output_entries << '\n'
                  << "max_vertex_id: " << max_id << '\n'
                  << "indptr: " << indptr_path << '\n'
                  << "indices: " << indices_path << '\n'
                  << (opt.remap_ids ? "vertex_ids: " + vertex_ids_path + "\n" : "")
                  << "total_seconds: " << total_timer.seconds() << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
