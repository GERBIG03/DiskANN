#pragma once

#include "common_includes.h"
#include "ann_exception.h"
#include "distance.h"
#include "logger.h"
#include "parameters.h"

namespace diskann
{
enum class DataStoreStrategy
{
    MEMORY
};

enum class GraphStoreStrategy
{
    MEMORY
};

enum class ShortEdgeAugmentationMode
{
    NONE,
    EXACT,
    APPROX
};

struct AdaptiveReversePruneParams
{
    bool enabled = false;
    bool strict_require_reorder = false;
    float alpha = 1.0f;
    float k_mad = 1.0f;
    float r_min_ratio = 0.5f;
    uint32_t min_neighbors_to_check = 4;
};

struct IndexConfig
{
    DataStoreStrategy data_strategy;
    GraphStoreStrategy graph_strategy;

    Metric metric;
    size_t dimension;
    size_t max_points;

    bool dynamic_index;
    bool enable_tags;
    bool pq_dist_build;
    bool concurrent_consolidate;
    bool use_opq;
    bool filtered_index;
    bool force_reordered_start;
    ShortEdgeAugmentationMode short_edge_augmentation_mode;
    AdaptiveReversePruneParams adaptive_reverse_prune_params;

    size_t num_pq_chunks;
    size_t num_frozen_pts;
    size_t short_edge_n_samples;
    size_t short_edge_max_candidates;
    size_t short_edge_approx_L;
    float short_edge_alpha;
    std::string short_edge_exact_path;
    std::string label_type;
    std::string tag_type;
    std::string data_type;

    // Params for building index
    std::shared_ptr<IndexWriteParameters> index_write_params;
    // Params for searching index
    std::shared_ptr<IndexSearchParams> index_search_params;

  private:
    IndexConfig(DataStoreStrategy data_strategy, GraphStoreStrategy graph_strategy, Metric metric, size_t dimension,
                size_t max_points, size_t num_pq_chunks, size_t num_frozen_points, bool dynamic_index, bool enable_tags,
                bool pq_dist_build, bool concurrent_consolidate, bool use_opq, bool filtered_index,
                bool force_reordered_start, ShortEdgeAugmentationMode short_edge_augmentation_mode,
                AdaptiveReversePruneParams adaptive_reverse_prune_params, size_t short_edge_n_samples,
                size_t short_edge_max_candidates, size_t short_edge_approx_L,
                float short_edge_alpha, std::string &short_edge_exact_path, std::string &data_type,
                const std::string &tag_type, const std::string &label_type,
                std::shared_ptr<IndexWriteParameters> index_write_params,
                std::shared_ptr<IndexSearchParams> index_search_params)
        : data_strategy(data_strategy), graph_strategy(graph_strategy), metric(metric), dimension(dimension),
          max_points(max_points), dynamic_index(dynamic_index), enable_tags(enable_tags), pq_dist_build(pq_dist_build),
          concurrent_consolidate(concurrent_consolidate), use_opq(use_opq), filtered_index(filtered_index),
          force_reordered_start(force_reordered_start), short_edge_augmentation_mode(short_edge_augmentation_mode),
          adaptive_reverse_prune_params(adaptive_reverse_prune_params),
          num_pq_chunks(num_pq_chunks), num_frozen_pts(num_frozen_points), short_edge_n_samples(short_edge_n_samples),
          short_edge_max_candidates(short_edge_max_candidates), short_edge_approx_L(short_edge_approx_L),
          short_edge_alpha(short_edge_alpha), short_edge_exact_path(short_edge_exact_path), label_type(label_type),
          tag_type(tag_type), data_type(data_type), index_write_params(index_write_params),
          index_search_params(index_search_params)
    {
    }

    friend class IndexConfigBuilder;
};

class IndexConfigBuilder
{
  public:
    IndexConfigBuilder() = default;

    IndexConfigBuilder &with_metric(Metric m)
    {
        this->_metric = m;
        return *this;
    }

    IndexConfigBuilder &with_graph_load_store_strategy(GraphStoreStrategy graph_strategy)
    {
        this->_graph_strategy = graph_strategy;
        return *this;
    }

    IndexConfigBuilder &with_data_load_store_strategy(DataStoreStrategy data_strategy)
    {
        this->_data_strategy = data_strategy;
        return *this;
    }

    IndexConfigBuilder &with_dimension(size_t dimension)
    {
        this->_dimension = dimension;
        return *this;
    }

    IndexConfigBuilder &with_max_points(size_t max_points)
    {
        this->_max_points = max_points;
        return *this;
    }

    IndexConfigBuilder &is_dynamic_index(bool dynamic_index)
    {
        this->_dynamic_index = dynamic_index;
        return *this;
    }

    IndexConfigBuilder &is_enable_tags(bool enable_tags)
    {
        this->_enable_tags = enable_tags;
        return *this;
    }

    IndexConfigBuilder &is_pq_dist_build(bool pq_dist_build)
    {
        this->_pq_dist_build = pq_dist_build;
        return *this;
    }

    IndexConfigBuilder &is_concurrent_consolidate(bool concurrent_consolidate)
    {
        this->_concurrent_consolidate = concurrent_consolidate;
        return *this;
    }

    IndexConfigBuilder &is_use_opq(bool use_opq)
    {
        this->_use_opq = use_opq;
        return *this;
    }

    IndexConfigBuilder &is_filtered(bool is_filtered)
    {
        this->_filtered_index = is_filtered;
        return *this;
    }

    IndexConfigBuilder &force_reordered_start(bool force_reordered_start)
    {
        this->_force_reordered_start = force_reordered_start;
        return *this;
    }

    IndexConfigBuilder &with_short_edge_augmentation_mode(ShortEdgeAugmentationMode mode)
    {
        this->_short_edge_augmentation_mode = mode;
        return *this;
    }

    IndexConfigBuilder &with_adaptive_reverse_prune_params(const AdaptiveReversePruneParams &params)
    {
        this->_adaptive_reverse_prune_params = params;
        return *this;
    }

    IndexConfigBuilder &enable_adaptive_reverse_prune(bool enabled)
    {
        this->_adaptive_reverse_prune_params.enabled = enabled;
        return *this;
    }

    IndexConfigBuilder &with_adaptive_reverse_prune_strict_require_reorder(bool strict_require_reorder)
    {
        this->_adaptive_reverse_prune_params.strict_require_reorder = strict_require_reorder;
        return *this;
    }

    IndexConfigBuilder &with_adaptive_reverse_prune_alpha(float alpha)
    {
        this->_adaptive_reverse_prune_params.alpha = alpha;
        return *this;
    }

    IndexConfigBuilder &with_adaptive_reverse_prune_k_mad(float k_mad)
    {
        this->_adaptive_reverse_prune_params.k_mad = k_mad;
        return *this;
    }

    IndexConfigBuilder &with_adaptive_reverse_prune_r_min_ratio(float r_min_ratio)
    {
        this->_adaptive_reverse_prune_params.r_min_ratio = r_min_ratio;
        return *this;
    }

    IndexConfigBuilder &with_adaptive_reverse_prune_min_neighbors_to_check(uint32_t min_neighbors_to_check)
    {
        this->_adaptive_reverse_prune_params.min_neighbors_to_check = min_neighbors_to_check;
        return *this;
    }

    IndexConfigBuilder &with_short_edge_n_samples(size_t n_samples)
    {
        this->_short_edge_n_samples = n_samples;
        return *this;
    }

    IndexConfigBuilder &with_short_edge_max_candidates(size_t max_candidates)
    {
        this->_short_edge_max_candidates = max_candidates;
        return *this;
    }

    IndexConfigBuilder &with_short_edge_approx_L(size_t approx_L)
    {
        this->_short_edge_approx_L = approx_L;
        return *this;
    }

    IndexConfigBuilder &with_short_edge_alpha(float alpha)
    {
        this->_short_edge_alpha = alpha;
        return *this;
    }

    IndexConfigBuilder &with_short_edge_exact_path(const std::string &exact_path)
    {
        this->_short_edge_exact_path = exact_path;
        return *this;
    }

    IndexConfigBuilder &with_num_pq_chunks(size_t num_pq_chunks)
    {
        this->_num_pq_chunks = num_pq_chunks;
        return *this;
    }

    IndexConfigBuilder &with_num_frozen_pts(size_t num_frozen_pts)
    {
        this->_num_frozen_pts = num_frozen_pts;
        return *this;
    }

    IndexConfigBuilder &with_label_type(const std::string &label_type)
    {
        this->_label_type = label_type;
        return *this;
    }

    IndexConfigBuilder &with_tag_type(const std::string &tag_type)
    {
        this->_tag_type = tag_type;
        return *this;
    }

    IndexConfigBuilder &with_data_type(const std::string &data_type)
    {
        this->_data_type = data_type;
        return *this;
    }

    IndexConfigBuilder &with_index_write_params(IndexWriteParameters &index_write_params)
    {
        this->_index_write_params = std::make_shared<IndexWriteParameters>(index_write_params);
        return *this;
    }

    IndexConfigBuilder &with_index_write_params(std::shared_ptr<IndexWriteParameters> index_write_params_ptr)
    {
        if (index_write_params_ptr == nullptr)
        {
            diskann::cout << "Passed, empty build_params while creating index config" << std::endl;
            return *this;
        }
        this->_index_write_params = index_write_params_ptr;
        return *this;
    }

    IndexConfigBuilder &with_index_search_params(IndexSearchParams &search_params)
    {
        this->_index_search_params = std::make_shared<IndexSearchParams>(search_params);
        return *this;
    }

    IndexConfigBuilder &with_index_search_params(std::shared_ptr<IndexSearchParams> search_params_ptr)
    {
        if (search_params_ptr == nullptr)
        {
            diskann::cout << "Passed, empty search_params while creating index config" << std::endl;
            return *this;
        }
        this->_index_search_params = search_params_ptr;
        return *this;
    }

    IndexConfig build()
    {
        if (_data_type == "" || _data_type.empty())
            throw ANNException("Error: data_type can not be empty", -1);

        if (_dynamic_index && _num_frozen_pts == 0)
        {
            _num_frozen_pts = 1;
        }

        if (_dynamic_index)
        {
            if (_index_search_params != nullptr && _index_search_params->initial_search_list_size == 0)
                throw ANNException("Error: please pass initial_search_list_size for building dynamic index.", -1);
        }

        // sanity check
        if (_dynamic_index && _num_frozen_pts == 0)
        {
            diskann::cout << "_num_frozen_pts passed as 0 for dynamic_index. Setting it to 1 for safety." << std::endl;
            _num_frozen_pts = 1;
        }

        return IndexConfig(_data_strategy, _graph_strategy, _metric, _dimension, _max_points, _num_pq_chunks,
                           _num_frozen_pts, _dynamic_index, _enable_tags, _pq_dist_build, _concurrent_consolidate,
                           _use_opq, _filtered_index, _force_reordered_start, _short_edge_augmentation_mode,
                           _adaptive_reverse_prune_params, _short_edge_n_samples, _short_edge_max_candidates,
                           _short_edge_approx_L, _short_edge_alpha, _short_edge_exact_path, _data_type, _tag_type,
                           _label_type, _index_write_params, _index_search_params);
    }

    IndexConfigBuilder(const IndexConfigBuilder &) = delete;
    IndexConfigBuilder &operator=(const IndexConfigBuilder &) = delete;

  private:
    DataStoreStrategy _data_strategy;
    GraphStoreStrategy _graph_strategy;

    Metric _metric;
    size_t _dimension;
    size_t _max_points;

    bool _dynamic_index = false;
    bool _enable_tags = false;
    bool _pq_dist_build = false;
    bool _concurrent_consolidate = false;
    bool _use_opq = false;
    bool _filtered_index{defaults::HAS_LABELS};
    bool _force_reordered_start = false;
    ShortEdgeAugmentationMode _short_edge_augmentation_mode = ShortEdgeAugmentationMode::NONE;
    AdaptiveReversePruneParams _adaptive_reverse_prune_params;

    size_t _num_pq_chunks = 0;
    size_t _num_frozen_pts{defaults::NUM_FROZEN_POINTS_STATIC};
    size_t _short_edge_n_samples = 1024;
    size_t _short_edge_max_candidates = 0;
    size_t _short_edge_approx_L = 50;
    float _short_edge_alpha = 1.0f;

    std::string _label_type{"uint32"};
    std::string _tag_type{"uint32"};
    std::string _data_type;
    std::string _short_edge_exact_path;

    std::shared_ptr<IndexWriteParameters> _index_write_params;
    std::shared_ptr<IndexSearchParams> _index_search_params;
};
} // namespace diskann
