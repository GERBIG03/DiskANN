// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <cstring>
#include <boost/program_options.hpp>

#include "index.h"
#include "utils.h"
#include "program_options_utils.hpp"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif

#include "memory_mapper.h"
#include "ann_exception.h"
#include "index_factory.h"

namespace po = boost::program_options;

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, data_path, index_path_prefix, label_file, universal_label, label_type,
        short_edge_mode, short_edge_exact_path;
    uint32_t num_threads, R, L, Lf, build_PQ_bytes, short_edge_n_samples, short_edge_max_candidates,
        short_edge_approx_L, adaptive_reverse_prune_min_neighbors_to_check;
    float alpha, short_edge_alpha, adaptive_reverse_prune_alpha, adaptive_reverse_prune_k_mad,
        adaptive_reverse_prune_r_min_ratio;
    bool use_pq_build, use_opq, force_reordered_start, enable_adaptive_reverse_prune,
        adaptive_reverse_prune_strict_require_reorder;

    po::options_description desc{
        program_options_utils::make_program_description("build_memory_index", "Build a memory-based DiskANN index.")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("data_path", po::value<std::string>(&data_path)->required(),
                                       program_options_utils::INPUT_DATA_PATH);

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("max_degree,R", po::value<uint32_t>(&R)->default_value(64),
                                       program_options_utils::MAX_BUILD_DEGREE);
        optional_configs.add_options()("Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
                                       program_options_utils::GRAPH_BUILD_COMPLEXITY);
        optional_configs.add_options()("alpha", po::value<float>(&alpha)->default_value(1.2f),
                                       program_options_utils::GRAPH_BUILD_ALPHA);
        optional_configs.add_options()("build_PQ_bytes", po::value<uint32_t>(&build_PQ_bytes)->default_value(0),
                                       program_options_utils::BUIlD_GRAPH_PQ_BYTES);
        optional_configs.add_options()("use_opq", po::bool_switch()->default_value(false),
                                       program_options_utils::USE_OPQ);
        optional_configs.add_options()("force_reordered_start", po::bool_switch()->default_value(false),
                                       "Force the entry point to use the first vector in the input dataset");
        optional_configs.add_options()("short_edge_mode",
                                       po::value<std::string>(&short_edge_mode)->default_value("none"),
                                       "Short-edge augmentation mode: none, exact, approx");
        optional_configs.add_options()("short_edge_exact_path",
                                       po::value<std::string>(&short_edge_exact_path)->default_value(""),
                                       "Path to exact KNN text file for short-edge augmentation");
        optional_configs.add_options()("short_edge_n_samples",
                                       po::value<uint32_t>(&short_edge_n_samples)->default_value(1024),
                                       "Number of sampled nodes for global radius estimation");
        optional_configs.add_options()("short_edge_max_candidates",
                                       po::value<uint32_t>(&short_edge_max_candidates)->default_value(0),
                                       "Upper bound on extra short edges added per node");
        optional_configs.add_options()("short_edge_approx_L",
                                       po::value<uint32_t>(&short_edge_approx_L)->default_value(50),
                                       "Search list size for approximate short-edge augmentation");
        optional_configs.add_options()("short_edge_alpha",
                                       po::value<float>(&short_edge_alpha)->default_value(1.0f),
                                       "Sigmoid sharpness for density-adaptive short-edge augmentation");
        optional_configs.add_options()("enable_adaptive_reverse_prune",
                                       po::bool_switch()->default_value(false),
                                       "Enable adaptive pruning for reverse-edge insertion");
        optional_configs.add_options()("adaptive_reverse_prune_strict_require_reorder",
                                       po::bool_switch()->default_value(false),
                                       "Require reordered input and --force_reordered_start when adaptive reverse pruning is enabled");
        optional_configs.add_options()("adaptive_reverse_prune_alpha",
                                       po::value<float>(&adaptive_reverse_prune_alpha)->default_value(1.0f),
                                       "Alpha multiplier in adaptive reverse prune trigger threshold");
        optional_configs.add_options()("adaptive_reverse_prune_k_mad",
                                       po::value<float>(&adaptive_reverse_prune_k_mad)->default_value(1.0f),
                                       "MAD multiplier in adaptive reverse prune cutoff");
        optional_configs.add_options()("adaptive_reverse_prune_r_min_ratio",
                                       po::value<float>(&adaptive_reverse_prune_r_min_ratio)->default_value(0.5f),
                                       "Minimum degree ratio against R before checking reverse-edge pruning");
        optional_configs.add_options()("adaptive_reverse_prune_min_neighbors_to_check",
                                       po::value<uint32_t>(&adaptive_reverse_prune_min_neighbors_to_check)
                                           ->default_value(4),
                                       "Minimum existing neighbors before checking adaptive reverse pruning");
        optional_configs.add_options()("label_file", po::value<std::string>(&label_file)->default_value(""),
                                       program_options_utils::LABEL_FILE);
        optional_configs.add_options()("universal_label", po::value<std::string>(&universal_label)->default_value(""),
                                       program_options_utils::UNIVERSAL_LABEL);

        optional_configs.add_options()("FilteredLbuild", po::value<uint32_t>(&Lf)->default_value(0),
                                       program_options_utils::FILTERED_LBUILD);
        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                                       program_options_utils::LABEL_TYPE_DESCRIPTION);

        // Merge required and optional parameters
        desc.add(required_configs).add(optional_configs);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
        use_pq_build = (build_PQ_bytes > 0);
        use_opq = vm["use_opq"].as<bool>();
        force_reordered_start = vm["force_reordered_start"].as<bool>();
        enable_adaptive_reverse_prune = vm["enable_adaptive_reverse_prune"].as<bool>();
        adaptive_reverse_prune_strict_require_reorder =
            vm["adaptive_reverse_prune_strict_require_reorder"].as<bool>();
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    diskann::Metric metric;
    if (dist_fn == std::string("mips"))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently only L2/ Inner "
                     "Product/Cosine are supported."
                  << std::endl;
        return -1;
    }

    try
    {
        diskann::ShortEdgeAugmentationMode short_edge_augmentation_mode =
            diskann::ShortEdgeAugmentationMode::NONE;
        if (short_edge_mode == "exact")
        {
            short_edge_augmentation_mode = diskann::ShortEdgeAugmentationMode::EXACT;
        }
        else if (short_edge_mode == "approx")
        {
            short_edge_augmentation_mode = diskann::ShortEdgeAugmentationMode::APPROX;
        }
        else if (short_edge_mode != "none")
        {
            std::cerr << "Unsupported short_edge_mode: " << short_edge_mode << std::endl;
            return -1;
        }

        diskann::cout << "Starting index build with R: " << R << "  Lbuild: " << L << "  alpha: " << alpha
                      << "  #threads: " << num_threads << std::endl;
        diskann::cout << "Effective config: force_reordered_start=" << (force_reordered_start ? "true" : "false")
                      << " short_edge_mode=" << short_edge_mode
                      << " adaptive_reverse_prune="
                      << (enable_adaptive_reverse_prune ? "true" : "false") << std::endl;

        size_t data_num, data_dim;
        diskann::get_bin_metadata(data_path, data_num, data_dim);

        auto index_build_params = diskann::IndexWriteParametersBuilder(L, R)
                                      .with_filter_list_size(Lf)
                                      .with_alpha(alpha)
                                      .with_saturate_graph(false)
                                      .with_num_threads(num_threads)
                                      .build();

        auto filter_params = diskann::IndexFilterParamsBuilder()
                                 .with_universal_label(universal_label)
                                 .with_label_file(label_file)
                                 .with_save_path_prefix(index_path_prefix)
                                 .build();
        diskann::AdaptiveReversePruneParams adaptive_reverse_prune_params;
        adaptive_reverse_prune_params.enabled = enable_adaptive_reverse_prune;
        adaptive_reverse_prune_params.strict_require_reorder = adaptive_reverse_prune_strict_require_reorder;
        adaptive_reverse_prune_params.alpha = adaptive_reverse_prune_alpha;
        adaptive_reverse_prune_params.k_mad = adaptive_reverse_prune_k_mad;
        adaptive_reverse_prune_params.r_min_ratio = adaptive_reverse_prune_r_min_ratio;
        adaptive_reverse_prune_params.min_neighbors_to_check =
            adaptive_reverse_prune_min_neighbors_to_check;
        auto config = diskann::IndexConfigBuilder()
                          .with_metric(metric)
                          .with_dimension(data_dim)
                          .with_max_points(data_num)
                          .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                          .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                          .with_data_type(data_type)
                          .with_label_type(label_type)
                          .is_dynamic_index(false)
                          .with_index_write_params(index_build_params)
                          .is_enable_tags(false)
                          .is_use_opq(use_opq)
                          .force_reordered_start(force_reordered_start)
                          .with_adaptive_reverse_prune_params(adaptive_reverse_prune_params)
                          .with_short_edge_augmentation_mode(short_edge_augmentation_mode)
                          .with_short_edge_n_samples(short_edge_n_samples)
                          .with_short_edge_max_candidates(short_edge_max_candidates == 0 ? R : short_edge_max_candidates)
                          .with_short_edge_approx_L(short_edge_approx_L)
                          .with_short_edge_alpha(short_edge_alpha)
                          .with_short_edge_exact_path(short_edge_exact_path)
                          .is_pq_dist_build(use_pq_build)
                          .with_num_pq_chunks(build_PQ_bytes)
                          .build();

        auto index_factory = diskann::IndexFactory(config);
        auto index = index_factory.create_instance();
        index->build(data_path, data_num, filter_params);
        index->save(index_path_prefix.c_str());
        index.reset();
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index build failed." << std::endl;
        return -1;
    }
}
