#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

struct Edge {
    int64_t src;
    int64_t dst;
    float value;
};

struct Options {
    int64_t vertices = 8;
    int64_t edges = 24;
    std::string name = "toy";
    std::string output_dir = "graph_generator/out";
    uint64_t seed = 1;
    bool allow_self_loops = false;
};

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + flag);
            return argv[++i];
        };

        if (arg == "--vertices") opt.vertices = std::stoll(value("--vertices"));
        else if (arg == "--edges") opt.edges = std::stoll(value("--edges"));
        else if (arg == "--name") opt.name = value("--name");
        else if (arg == "--output-dir") opt.output_dir = value("--output-dir");
        else if (arg == "--seed") opt.seed = static_cast<uint64_t>(std::stoull(value("--seed")));
        else if (arg == "--allow-self-loops") opt.allow_self_loops = true;
        else if (arg == "--help") {
            std::cout
                << "Usage: generate_graph --vertices N --edges M --name NAME --output-dir DIR [--seed S]\n"
                << "Writes NAME_coo.txt, NAME_indptr.bin, NAME_indices.bin, NAME_values.bin.\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return opt;
}

uint64_t edge_key(int64_t src, int64_t dst) {
    return (static_cast<uint64_t>(src) << 32) ^ static_cast<uint64_t>(dst);
}

void validate_options(const Options& opt) {
    if (opt.vertices <= 0) throw std::runtime_error("--vertices must be positive");
    if (opt.edges < 0) throw std::runtime_error("--edges must be non-negative");
    if (opt.vertices > (1LL << 32)) {
        throw std::runtime_error("This simple generator supports at most 2^32 vertices");
    }

    const int64_t max_edges = opt.allow_self_loops
        ? opt.vertices * opt.vertices
        : opt.vertices * (opt.vertices - 1);
    if (opt.edges > max_edges) {
        throw std::runtime_error("--edges is larger than the number of possible unique edges");
    }
}

std::vector<Edge> generate_edges(const Options& opt) {
    std::mt19937_64 rng(opt.seed);
    std::uniform_int_distribution<int64_t> vertex_dist(0, opt.vertices - 1);
    std::uniform_real_distribution<float> value_dist(0.0f, 1.0f);

    std::vector<Edge> edges;
    edges.reserve(static_cast<size_t>(opt.edges));

    std::unordered_set<uint64_t> seen;
    seen.reserve(static_cast<size_t>(opt.edges * 2 + 1));

    while (static_cast<int64_t>(edges.size()) < opt.edges) {
        int64_t src = vertex_dist(rng);
        int64_t dst = vertex_dist(rng);
        if (!opt.allow_self_loops && src == dst) continue;

        uint64_t key = edge_key(src, dst);
        if (!seen.insert(key).second) continue;

        edges.push_back({src, dst, value_dist(rng)});
    }

    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        if (a.src != b.src) return a.src < b.src;
        return a.dst < b.dst;
    });
    return edges;
}

template <typename T>
void write_binary_vector(const std::filesystem::path& path, const std::vector<T>& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write " + path.string());
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
    if (!out) throw std::runtime_error("Failed writing " + path.string());
}

void write_coo(const std::filesystem::path& path, const std::vector<Edge>& edges) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write " + path.string());
    for (const auto& e : edges) {
        out << e.src << ' ' << e.dst << ' ' << e.value << '\n';
    }
}

void write_csr_files(const Options& opt, const std::vector<Edge>& edges) {
    std::filesystem::create_directories(opt.output_dir);
    const std::filesystem::path dir(opt.output_dir);

    std::vector<int64_t> indptr(static_cast<size_t>(opt.vertices) + 1, 0);
    std::vector<int64_t> indices;
    std::vector<float> values;
    indices.reserve(edges.size());
    values.reserve(edges.size());

    int64_t current_src = 0;
    int64_t edge_pos = 0;
    for (const auto& e : edges) {
        while (current_src < e.src) {
            indptr[static_cast<size_t>(current_src) + 1] = edge_pos;
            ++current_src;
        }
        indices.push_back(e.dst);
        values.push_back(e.value);
        ++edge_pos;
    }
    while (current_src < opt.vertices) {
        indptr[static_cast<size_t>(current_src) + 1] = edge_pos;
        ++current_src;
    }

    write_coo(dir / (opt.name + "_coo.txt"), edges);
    write_binary_vector(dir / (opt.name + "_indptr.bin"), indptr);
    write_binary_vector(dir / (opt.name + "_indices.bin"), indices);
    write_binary_vector(dir / (opt.name + "_values.bin"), values);

    std::cout << "wrote " << (dir / (opt.name + "_coo.txt")) << "\n";
    std::cout << "wrote " << (dir / (opt.name + "_indptr.bin")) << " int64 elements=" << indptr.size() << "\n";
    std::cout << "wrote " << (dir / (opt.name + "_indices.bin")) << " int64 elements=" << indices.size() << "\n";
    std::cout << "wrote " << (dir / (opt.name + "_values.bin")) << " float elements=" << values.size() << "\n";
}

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);
        validate_options(opt);
        auto edges = generate_edges(opt);
        write_csr_files(opt, edges);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

