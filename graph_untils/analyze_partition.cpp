#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

struct Options {
    std::vector<std::string> partition_paths;
    std::string indptr_path;
    std::string indices_path;
    std::string values_path;
    std::string graph_dir;
    std::string graph_name;
    int parts = 0;
    double balance_ratio = 1.05;
    double edge_balance_ratio = 1.05;
    bool detail = false;
};

struct PartStats {
    int64_t vertices = 0;
    int64_t internal_edges = 0;
    int64_t outgoing_cut_edges = 0;
    int64_t incoming_cut_edges = 0;
    double weighted_internal_edges = 0.0;
    double weighted_outgoing_cut = 0.0;
    double weighted_incoming_cut = 0.0;
};

struct AnalysisResult {
    std::string partition_path;
    int num_parts = 0;
    int64_t directed_edge_cut = 0;
    double weighted_directed_edge_cut = 0.0;
    int64_t max_local_cut = 0;
    double avg_local_cut = 0.0;
    double weighted_max_local_cut = 0.0;
    double weighted_avg_local_cut = 0.0;
    double vertex_avg = 0.0;
    int64_t min_vertices = 0;
    int64_t max_vertices = 0;
    double vertex_imbalance = 0.0;
    double vertex_max_deviation = 0.0;
    double edge_set_avg = 0.0;
    int64_t min_edge_set = 0;
    int64_t max_edge_set = 0;
    double edge_set_imbalance = 0.0;
    double edge_set_max_deviation = 0.0;
    std::vector<PartStats> stats;
};

void print_usage(const char* prog) {
    std::cout
        << "Usage:\n"
        << "  " << prog << " PARTITION1.txt [PARTITION2.txt ...] --graph-dir DIR --name NAME [--parts K]\n"
        << "  " << prog << " --partition PARTITION.txt [--partition PARTITION2.txt ...] --indptr FILE --indices FILE [--values FILE] [--parts K]\n\n"
        << "Metrics are computed for directed CSR adjacency entries. Each cross-part edge counts\n"
        << "once in the global directed edge cut and once as outgoing/incoming boundary for\n"
        << "the source/destination partitions. Multiple partition files are analyzed against\n"
        << "the same original CSR graph.\n\n"
        << "Default output is a compact partition-quality table. Add --detail for per-part\n"
        << "debug metrics.\n";
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + name);
            return argv[++i];
        };
        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (a == "--partition" || a == "--parts-file") {
            opt.partition_paths.push_back(need_value(a.c_str()));
        } else if (a == "--indptr") {
            opt.indptr_path = need_value("--indptr");
        } else if (a == "--indices") {
            opt.indices_path = need_value("--indices");
        } else if (a == "--values") {
            opt.values_path = need_value("--values");
        } else if (a == "--graph-dir") {
            opt.graph_dir = need_value("--graph-dir");
        } else if (a == "--name") {
            opt.graph_name = need_value("--name");
        } else if (a == "--parts") {
            opt.parts = std::stoi(need_value("--parts"));
        } else if (a == "--balance-ratio") {
            opt.balance_ratio = std::stod(need_value("--balance-ratio"));
        } else if (a == "--edge-balance-ratio") {
            opt.edge_balance_ratio = std::stod(need_value("--edge-balance-ratio"));
        } else if (a == "--detail") {
            opt.detail = true;
        } else if (!a.empty() && a[0] != '-') {
            opt.partition_paths.push_back(a);
        } else {
            throw std::runtime_error("Unknown argument: " + a);
        }
    }

    if (!opt.graph_dir.empty()) {
        if (opt.graph_name.empty()) throw std::runtime_error("--graph-dir requires --name");
        std::filesystem::path dir(opt.graph_dir);
        if (opt.indptr_path.empty()) opt.indptr_path = (dir / (opt.graph_name + "_indptr.bin")).string();
        if (opt.indices_path.empty()) opt.indices_path = (dir / (opt.graph_name + "_indices.bin")).string();
        if (opt.values_path.empty()) {
            const auto values = dir / (opt.graph_name + "_values.bin");
            if (std::filesystem::exists(values)) opt.values_path = values.string();
        }
    }

    if (opt.partition_paths.empty()) throw std::runtime_error("Provide at least one partition txt file");
    if (opt.indptr_path.empty() || opt.indices_path.empty()) {
        throw std::runtime_error("Provide original graph CSR with --indptr and --indices, or --graph-dir and --name");
    }
    if (opt.parts < 0) throw std::runtime_error("--parts must be non-negative");
    if (opt.balance_ratio <= 0.0) throw std::runtime_error("--balance-ratio must be positive");
    if (opt.edge_balance_ratio <= 0.0) throw std::runtime_error("--edge-balance-ratio must be positive");
    return opt;
}

template <typename T>
std::vector<T> read_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open " + path);
    const auto bytes = file.tellg();
    if (bytes % static_cast<std::streamoff>(sizeof(T)) != 0) {
        throw std::runtime_error("File size is not a multiple of element size: " + path);
    }
    std::vector<T> data(static_cast<size_t>(bytes) / sizeof(T));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(data.data()), bytes);
    if (!file) throw std::runtime_error("Failed reading " + path);
    return data;
}

std::vector<int> read_partitions(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open " + path);
    std::vector<int> parts;
    int p = 0;
    while (in >> p) {
        if (p < 0) throw std::runtime_error("Partition id must be non-negative in " + path);
        parts.push_back(p);
    }
    if (!in.eof()) throw std::runtime_error("Failed parsing partition file " + path);
    return parts;
}

std::vector<float> default_edge_values(size_t num_edges) {
    return std::vector<float>(num_edges, 1.0f);
}

void validate_csr(const std::vector<int64_t>& indptr, const std::vector<int64_t>& indices) {
    if (indptr.empty() || indptr.front() != 0) throw std::runtime_error("Invalid CSR indptr");
    if (indptr.back() != static_cast<int64_t>(indices.size())) {
        throw std::runtime_error("CSR indptr.back() must equal indices.size()");
    }
    for (size_t i = 1; i < indptr.size(); ++i) {
        if (indptr[i] < indptr[i - 1]) throw std::runtime_error("CSR indptr must be nondecreasing");
    }
    const int64_t n = static_cast<int64_t>(indptr.size()) - 1;
    for (int64_t dst : indices) {
        if (dst < 0 || dst >= n) throw std::runtime_error("CSR index out of vertex range");
    }
}

int infer_num_parts(const std::vector<int>& part_ids, int requested_parts) {
    int max_part = -1;
    for (int p : part_ids) max_part = std::max(max_part, p);
    const int inferred = max_part + 1;
    if (requested_parts == 0) return inferred;
    if (inferred > requested_parts) throw std::runtime_error("Partition file contains part id >= --parts");
    return requested_parts;
}

double deviation_ratio(double value, double average) {
    if (average <= 0.0) return value == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
    return std::abs(value - average) / average;
}

double imbalance_ratio(double max_value, double average) {
    return average > 0.0 ? max_value / average : 0.0;
}

void print_separator(const std::string& title) {
    std::cout << "\n=========== " << title << " ===========\n";
}

void print_balance_block(
    const std::string& label,
    double average,
    double min_value,
    double max_value,
    double ratio_limit
) {
    const double imbalance = imbalance_ratio(max_value, average);
    const double max_deviation = deviation_ratio(max_value, average);
    const double min_deviation = deviation_ratio(min_value, average);
    std::cout
        << "metric: " << label << "\n"
        << "  avg: " << average << "\n"
        << "  min: " << min_value << "\n"
        << "  max: " << max_value << "\n"
        << "  imbalance(max/avg): " << imbalance << "\n"
        << "  max_deviation(abs(part-avg)/avg): " << std::max(max_deviation, min_deviation) << "\n"
        << "  allowed_max(avg*ratio): " << average * ratio_limit << "\n"
        << "  within_ratio(" << ratio_limit << "): " << (imbalance <= ratio_limit ? "yes" : "no") << "\n";
}

AnalysisResult analyze_partition(
    const std::string& partition_path,
    const std::vector<int64_t>& indptr,
    const std::vector<int64_t>& indices,
    const std::vector<float>& edge_values,
    int requested_parts
) {
    const auto part_ids = read_partitions(partition_path);
    const int64_t n = static_cast<int64_t>(indptr.size()) - 1;
    if (static_cast<int64_t>(part_ids.size()) != n) {
        throw std::runtime_error("Partition line count must equal CSR vertex count for " + partition_path);
    }

    AnalysisResult result;
    result.partition_path = partition_path;
    result.num_parts = infer_num_parts(part_ids, requested_parts);
    if (result.num_parts <= 0) throw std::runtime_error("Partition file is empty or has no parts: " + partition_path);

    result.stats.assign(static_cast<size_t>(result.num_parts), PartStats{});
    for (int p : part_ids) ++result.stats[static_cast<size_t>(p)].vertices;

    for (int64_t src = 0; src < n; ++src) {
        const int src_part = part_ids[static_cast<size_t>(src)];
        for (int64_t e = indptr[static_cast<size_t>(src)]; e < indptr[static_cast<size_t>(src + 1)]; ++e) {
            const int64_t dst = indices[static_cast<size_t>(e)];
            const int dst_part = part_ids[static_cast<size_t>(dst)];
            const double w = edge_values[static_cast<size_t>(e)];
            if (src_part == dst_part) {
                ++result.stats[static_cast<size_t>(src_part)].internal_edges;
                result.stats[static_cast<size_t>(src_part)].weighted_internal_edges += w;
            } else {
                ++result.directed_edge_cut;
                result.weighted_directed_edge_cut += w;
                ++result.stats[static_cast<size_t>(src_part)].outgoing_cut_edges;
                ++result.stats[static_cast<size_t>(dst_part)].incoming_cut_edges;
                result.stats[static_cast<size_t>(src_part)].weighted_outgoing_cut += w;
                result.stats[static_cast<size_t>(dst_part)].weighted_incoming_cut += w;
            }
        }
    }

    result.vertex_avg = static_cast<double>(n) / static_cast<double>(result.num_parts);
    const double edge_set_total = static_cast<double>(indices.size() + result.directed_edge_cut);
    result.edge_set_avg = edge_set_total / static_cast<double>(result.num_parts);
    result.min_vertices = std::numeric_limits<int64_t>::max();
    result.min_edge_set = std::numeric_limits<int64_t>::max();

    for (const auto& part_stat : result.stats) {
        const int64_t edge_set = part_stat.internal_edges + part_stat.outgoing_cut_edges + part_stat.incoming_cut_edges;
        const int64_t local_cut = part_stat.outgoing_cut_edges + part_stat.incoming_cut_edges;
        const double weighted_local_cut = part_stat.weighted_outgoing_cut + part_stat.weighted_incoming_cut;
        result.avg_local_cut += static_cast<double>(local_cut);
        result.weighted_avg_local_cut += weighted_local_cut;
        result.min_vertices = std::min(result.min_vertices, part_stat.vertices);
        result.max_vertices = std::max(result.max_vertices, part_stat.vertices);
        result.min_edge_set = std::min(result.min_edge_set, edge_set);
        result.max_edge_set = std::max(result.max_edge_set, edge_set);
        result.max_local_cut = std::max(result.max_local_cut, local_cut);
        result.weighted_max_local_cut = std::max(result.weighted_max_local_cut, weighted_local_cut);
    }
    result.avg_local_cut /= static_cast<double>(result.num_parts);
    result.weighted_avg_local_cut /= static_cast<double>(result.num_parts);

    result.vertex_imbalance = imbalance_ratio(result.max_vertices, result.vertex_avg);
    result.vertex_max_deviation = std::max(
        deviation_ratio(result.max_vertices, result.vertex_avg),
        deviation_ratio(result.min_vertices, result.vertex_avg)
    );
    result.edge_set_imbalance = imbalance_ratio(result.max_edge_set, result.edge_set_avg);
    result.edge_set_max_deviation = std::max(
        deviation_ratio(result.max_edge_set, result.edge_set_avg),
        deviation_ratio(result.min_edge_set, result.edge_set_avg)
    );
    return result;
}

void print_result_detail(
    const AnalysisResult& result,
    const Options& opt,
    const std::vector<int64_t>& indices,
    int64_t vertex_count
) {
    print_separator("Partition Result");
    std::cout << "partition_file: " << result.partition_path << "\n";

    print_separator("Graph Summary");
    std::cout
        << "vertices |V|: " << vertex_count << "\n"
        << "directed_edges |E|: " << indices.size() << "\n"
        << "parts: " << result.num_parts << "\n";

    print_separator("Cut Objectives");
    std::cout
        << "global_cut |C(G,P)|: " << result.directed_edge_cut << "\n"
        << "weighted_global_cut: " << result.weighted_directed_edge_cut << "\n"
        << "avg_local_cut avg_i |C(G,pi_i)|: " << result.avg_local_cut << "\n"
        << "max_local_cut max_i |C(G,pi_i)|: " << result.max_local_cut << "\n"
        << "weighted_avg_local_cut: " << result.weighted_avg_local_cut << "\n"
        << "weighted_max_local_cut: " << result.weighted_max_local_cut << "\n";

    print_separator("Balance Constraints");
    print_balance_block("vertex_set |V(pi_i)|", result.vertex_avg, result.min_vertices, result.max_vertices, opt.balance_ratio);
    std::cout << "\n";
    print_balance_block("edge_set |E(pi_i)|", result.edge_set_avg, result.min_edge_set, result.max_edge_set, opt.edge_balance_ratio);

    print_separator("Per-Part Metrics: Counts");
    std::cout
        << std::right
        << std::setw(6) << "part"
        << std::setw(14) << "|V(pi)|"
        << std::setw(14) << "|E(pi)|"
        << std::setw(16) << "internal_edges"
        << std::setw(15) << "outgoing_cut"
        << std::setw(15) << "incoming_cut"
        << std::setw(14) << "local_cut"
        << "\n";
    for (int p = 0; p < result.num_parts; ++p) {
        const auto& part_stat = result.stats[static_cast<size_t>(p)];
        const int64_t edge_set = part_stat.internal_edges + part_stat.outgoing_cut_edges + part_stat.incoming_cut_edges;
        const int64_t local_cut = part_stat.outgoing_cut_edges + part_stat.incoming_cut_edges;
        std::cout
            << std::right
            << std::setw(6) << p
            << std::setw(14) << part_stat.vertices
            << std::setw(14) << edge_set
            << std::setw(16) << part_stat.internal_edges
            << std::setw(15) << part_stat.outgoing_cut_edges
            << std::setw(15) << part_stat.incoming_cut_edges
            << std::setw(14) << local_cut
            << "\n";
    }

    print_separator("Per-Part Metrics: Weights And Deviation");
    std::cout
        << std::right
        << std::setw(6) << "part"
        << std::setw(22) << "weighted_internal"
        << std::setw(22) << "weighted_outgoing"
        << std::setw(22) << "weighted_incoming"
        << std::setw(22) << "weighted_local_cut"
        << std::setw(20) << "vertex_deviation"
        << std::setw(22) << "edge_set_deviation"
        << "\n";
    for (int p = 0; p < result.num_parts; ++p) {
        const auto& part_stat = result.stats[static_cast<size_t>(p)];
        const int64_t edge_set = part_stat.internal_edges + part_stat.outgoing_cut_edges + part_stat.incoming_cut_edges;
        const double weighted_local_cut = part_stat.weighted_outgoing_cut + part_stat.weighted_incoming_cut;
        std::cout
            << std::right
            << std::setw(6) << p
            << std::setw(22) << part_stat.weighted_internal_edges
            << std::setw(22) << part_stat.weighted_outgoing_cut
            << std::setw(22) << part_stat.weighted_incoming_cut
            << std::setw(22) << weighted_local_cut
            << std::setw(20) << deviation_ratio(static_cast<double>(part_stat.vertices), result.vertex_avg)
            << std::setw(22) << deviation_ratio(static_cast<double>(edge_set), result.edge_set_avg)
            << "\n";
    }
}

double edge_cut_ratio(const AnalysisResult& result, int64_t edge_count) {
    return edge_count > 0
        ? static_cast<double>(result.directed_edge_cut) / static_cast<double>(edge_count)
        : 0.0;
}

void print_quality_summary(
    const std::vector<AnalysisResult>& results,
    int64_t vertex_count,
    int64_t edge_count,
    const Options& opt
) {
    size_t path_width = std::string("partition_file").size();
    for (const auto& result : results) {
        path_width = std::max(path_width, result.partition_path.size());
    }
    path_width += 2;

    print_separator("Graph Summary");
    std::cout
        << "vertices: " << vertex_count << "\n"
        << "directed_edges: " << edge_count << "\n";

    print_separator("Partition Quality");
    std::cout
        << std::left
        << std::setw(static_cast<int>(path_width)) << "partition_file";
    std::cout
        << std::right
        << std::setw(7) << "parts"
        << std::setw(16) << "edge_cut"
        << std::setw(18) << "edge_cut_ratio"
        << std::setw(18) << "vertex_imb"
        << std::setw(18) << "edge_imb"
        << std::setw(18) << "avg_local_cut"
        << std::setw(18) << "max_local_cut"
        << "\n";
    for (const auto& result : results) {
        std::cout
            << std::left
            << std::setw(static_cast<int>(path_width)) << result.partition_path;
        std::cout
            << std::right
            << std::setw(7) << result.num_parts
            << std::setw(16) << result.directed_edge_cut
            << std::setw(18) << edge_cut_ratio(result, edge_count)
            << std::setw(18) << result.vertex_imbalance
            << std::setw(18) << result.edge_set_imbalance
            << std::setw(18) << result.avg_local_cut
            << std::setw(18) << result.max_local_cut
            << "\n";
    }

    print_separator("Balance Limits");
    std::cout
        << "vertex_imb_limit: " << opt.balance_ratio << "\n"
        << "edge_imb_limit: " << opt.edge_balance_ratio << "\n"
        << "imbalance values closer to 1.0 are better\n";
}

void print_detail_summary(const std::vector<AnalysisResult>& results) {
    if (results.size() <= 1) return;

    size_t path_width = std::string("partition_file").size();
    for (const auto& result : results) {
        path_width = std::max(path_width, result.partition_path.size());
    }
    path_width += 2;

    print_separator("Detailed Comparison");
    std::cout
        << std::left
        << std::setw(static_cast<int>(path_width)) << "partition_file";
    std::cout
        << std::right
        << std::setw(22) << "weighted_edge_cut"
        << std::setw(24) << "weighted_avg_local_cut"
        << std::setw(24) << "weighted_max_local_cut"
        << std::setw(24) << "vertex_max_deviation"
        << std::setw(24) << "edge_max_deviation"
        << "\n";
    for (const auto& result : results) {
        std::cout
            << std::left
            << std::setw(static_cast<int>(path_width)) << result.partition_path;
        std::cout
            << std::right
            << std::setw(22) << result.weighted_directed_edge_cut
            << std::setw(24) << result.weighted_avg_local_cut
            << std::setw(24) << result.weighted_max_local_cut
            << std::setw(24) << result.vertex_max_deviation
            << std::setw(24) << result.edge_set_max_deviation
            << "\n";
    }
}

int main(int argc, char** argv) {
    try {
        const Options opt = parse_args(argc, argv);
        const auto indptr = read_binary<int64_t>(opt.indptr_path);
        const auto indices = read_binary<int64_t>(opt.indices_path);
        const auto edge_values = opt.values_path.empty()
            ? default_edge_values(indices.size())
            : read_binary<float>(opt.values_path);

        validate_csr(indptr, indices);
        if (edge_values.size() != indices.size()) throw std::runtime_error("values.size() must equal indices.size()");

        std::cout << std::fixed << std::setprecision(6);
        if (opt.detail) {
            print_separator("Input");
            std::cout
                << "partition_files: " << opt.partition_paths.size() << "\n";
            for (const auto& path : opt.partition_paths) std::cout << "  " << path << "\n";
            std::cout
                << "indptr: " << opt.indptr_path << "\n"
                << "indices: " << opt.indices_path << "\n"
                << "values: " << (opt.values_path.empty() ? "<unit weights>" : opt.values_path) << "\n";
        }

        const int64_t vertex_count = static_cast<int64_t>(indptr.size()) - 1;
        const int64_t edge_count = static_cast<int64_t>(indices.size());
        std::vector<AnalysisResult> results;
        results.reserve(opt.partition_paths.size());
        for (const auto& partition_path : opt.partition_paths) {
            results.push_back(analyze_partition(partition_path, indptr, indices, edge_values, opt.parts));
        }

        print_quality_summary(results, vertex_count, edge_count, opt);
        if (opt.detail) {
            print_detail_summary(results);
            for (const auto& result : results) {
                print_result_detail(result, opt, indices, vertex_count);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
