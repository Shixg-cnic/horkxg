#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string indptr;
    std::string indices;
    std::string output;
    std::size_t read_entries = 1U << 22;
    std::size_t write_bytes = 1U << 26;
};

class BufferedWriter {
public:
    BufferedWriter(const std::string& path, std::size_t bytes)
        : file_(std::fopen(path.c_str(), "wb")), buffer_(bytes) {
        if (!file_) throw std::runtime_error("cannot open output: " + path);
    }

    ~BufferedWriter() {
        if (file_) {
            flush();
            std::fclose(file_);
        }
    }

    void write_char(char value) {
        if (position_ == buffer_.size()) flush();
        buffer_[position_++] = value;
    }

    void write_uint64(std::uint64_t value) {
        char text[32];
        const auto result = std::to_chars(text, text + sizeof(text), value);
        if (result.ec != std::errc()) {
            throw std::runtime_error("integer formatting failed");
        }
        write(text, static_cast<std::size_t>(result.ptr - text));
    }

    void flush() {
        if (position_ == 0) return;
        if (std::fwrite(buffer_.data(), 1, position_, file_) != position_) {
            throw std::runtime_error("output write failed");
        }
        position_ = 0;
    }

private:
    void write(const char* data, std::size_t bytes) {
        while (bytes != 0) {
            if (position_ == buffer_.size()) flush();
            const auto room = buffer_.size() - position_;
            const auto take = bytes < room ? bytes : room;
            std::memcpy(buffer_.data() + position_, data, take);
            position_ += take;
            data += take;
            bytes -= take;
        }
    }

    std::FILE* file_ = nullptr;
    std::vector<char> buffer_;
    std::size_t position_ = 0;
};

std::string argument_value(int& index, int argc, char** argv) {
    if (++index >= argc) throw std::runtime_error("missing option value");
    return argv[index];
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--indptr") {
            options.indptr = argument_value(index, argc, argv);
        } else if (argument == "--indices") {
            options.indices = argument_value(index, argc, argv);
        } else if (argument == "--output" || argument == "-o") {
            options.output = argument_value(index, argc, argv);
        } else if (argument == "--read-chunk-entries") {
            options.read_entries = std::stoull(argument_value(index, argc, argv));
        } else if (argument == "--write-buffer-mb") {
            options.write_bytes =
                std::stoull(argument_value(index, argc, argv)) << 20;
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: " << argv[0]
                << " --indptr FILE --indices FILE --output FILE\n"
                << "The input must already be a loop-free symmetric int64 CSR.\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (options.indptr.empty() || options.indices.empty() ||
        options.output.empty()) {
        throw std::runtime_error("--indptr, --indices, and --output are required");
    }
    if (options.read_entries == 0 || options.write_bytes < 1024) {
        throw std::runtime_error("buffer sizes must be positive");
    }
    return options;
}

std::uint64_t int64_count(const std::string& path) {
    const auto bytes = fs::file_size(path);
    if (bytes % sizeof(std::int64_t) != 0) {
        throw std::runtime_error("file is not int64 aligned: " + path);
    }
    return bytes / sizeof(std::int64_t);
}

std::int64_t read_offset(std::ifstream& input, const std::string& path) {
    std::int64_t value = 0;
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) throw std::runtime_error("failed reading " + path);
    return value;
}

void convert(const Options& options) {
    const auto offset_count = int64_count(options.indptr);
    const auto edge_entries = int64_count(options.indices);
    if (offset_count == 0) throw std::runtime_error("empty indptr");
    if ((edge_entries & 1U) != 0) {
        throw std::runtime_error("symmetric CSR has an odd adjacency count");
    }
    const auto vertices = offset_count - 1;

    std::ifstream offsets(options.indptr, std::ios::binary);
    std::ifstream indices(options.indices, std::ios::binary);
    if (!offsets) throw std::runtime_error("cannot open " + options.indptr);
    if (!indices) throw std::runtime_error("cannot open " + options.indices);

    BufferedWriter output(options.output, options.write_bytes);
    output.write_uint64(vertices);
    output.write_char(' ');
    output.write_uint64(edge_entries / 2);
    output.write_char('\n');

    std::vector<std::int64_t> buffer(options.read_entries);
    std::int64_t previous = read_offset(offsets, options.indptr);
    if (previous != 0) throw std::runtime_error("indptr[0] is not zero");
    std::uint64_t consumed = 0;
    const auto start = std::chrono::steady_clock::now();

    for (std::uint64_t row = 0; row < vertices; ++row) {
        const auto next = read_offset(offsets, options.indptr);
        if (next < previous || static_cast<std::uint64_t>(next) > edge_entries) {
            throw std::runtime_error("invalid CSR offsets at row " +
                                     std::to_string(row));
        }
        auto remaining = static_cast<std::uint64_t>(next - previous);
        bool first = true;
        while (remaining != 0) {
            const auto take = static_cast<std::size_t>(
                remaining < buffer.size() ? remaining : buffer.size());
            indices.read(reinterpret_cast<char*>(buffer.data()),
                         static_cast<std::streamsize>(take * sizeof(std::int64_t)));
            if (!indices) throw std::runtime_error("failed reading " + options.indices);
            for (std::size_t index = 0; index < take; ++index) {
                const auto neighbor = buffer[index];
                if (neighbor < 0 || static_cast<std::uint64_t>(neighbor) >= vertices) {
                    throw std::runtime_error("neighbor out of range at row " +
                                             std::to_string(row));
                }
                if (static_cast<std::uint64_t>(neighbor) == row) {
                    throw std::runtime_error("self-loop found at row " +
                                             std::to_string(row));
                }
                if (!first) output.write_char(' ');
                output.write_uint64(static_cast<std::uint64_t>(neighbor) + 1);
                first = false;
            }
            consumed += take;
            remaining -= take;
        }
        output.write_char('\n');
        previous = next;
        if ((row + 1) % 10000000 == 0) {
            const auto elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            std::cout << "rows=" << (row + 1) << '/' << vertices
                      << " entries=" << consumed << '/' << edge_entries
                      << " seconds=" << elapsed << '\n';
        }
    }
    if (static_cast<std::uint64_t>(previous) != edge_entries ||
        consumed != edge_entries) {
        throw std::runtime_error("final CSR edge count mismatch");
    }
    output.flush();
    const auto seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "vertices=" << vertices << " edge_entries=" << edge_entries
              << " undirected_edges=" << edge_entries / 2
              << " conversion_seconds=" << seconds << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        convert(parse_options(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
