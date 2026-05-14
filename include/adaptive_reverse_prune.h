#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

#include "index_config.h"

#ifndef DISKANN_ENABLE_ADAPTIVE_REVERSE_PRUNE
#define DISKANN_ENABLE_ADAPTIVE_REVERSE_PRUNE 1
#endif

namespace diskann
{

template <typename T> class AdaptiveReversePruner
{
  public:
    AdaptiveReversePruner(const AdaptiveReversePruneParams &params, const std::vector<float> &global_centroid,
                          size_t dim, uint32_t max_degree)
        : _params(params), _global_centroid(global_centroid), _dim(dim), _max_degree(max_degree)
    {
    }

    bool should_accept(const T *candidate_vec, const T *anchor_vec, const std::vector<uint32_t> &anchor_neighbors,
                       const std::function<void(uint32_t, T *)> &load_vector, float avg_out_degree,
                       float median_neighbor_distance) const
    {
#if DISKANN_ENABLE_ADAPTIVE_REVERSE_PRUNE
        if (!_params.enabled)
        {
            return true;
        }

        if (anchor_neighbors.size() < _params.min_neighbors_to_check)
        {
            return true;
        }

        const float trigger = std::max(static_cast<float>(_max_degree) * _params.r_min_ratio,
                                       avg_out_degree * _params.alpha);
        if (static_cast<float>(anchor_neighbors.size()) < trigger)
        {
            return true;
        }
        if (median_neighbor_distance == 0.0f)
        {
            return true;
        }

        std::vector<float> scores;
        scores.reserve(anchor_neighbors.size());
        std::vector<T> neighbor_vec(_dim);
        for (uint32_t neighbor : anchor_neighbors)
        {
            load_vector(neighbor, neighbor_vec.data());
            scores.push_back(compute_score(anchor_vec, neighbor_vec.data(), median_neighbor_distance));
        }

        if (scores.empty())
        {
            return true;
        }

        const float candidate_distance = compute_distance(anchor_vec, candidate_vec);
        const float candidate_len_grade = candidate_distance / median_neighbor_distance;
        const float candidate_cosine = compute_direction_cosine(anchor_vec, candidate_vec);
        const float candidate_score = compute_score(anchor_vec, candidate_vec, median_neighbor_distance);
        const float max_existing_score = *std::max_element(scores.begin(), scores.end());
        const float median_score = compute_median(scores);
        float mad = compute_mad(scores, median_score);
        if (mad == 0.0f)
        {
            mad = compute_stddev(scores, median_score);
        }
        if (mad == 0.0f)
        {
            return true;
        }

        const float cutoff = median_score + _params.k_mad * mad;
        const bool strong_reverse = candidate_cosine < 0.0f;
        const bool clearly_long = candidate_len_grade > 1.5f;
        return !(strong_reverse && clearly_long && candidate_score > cutoff && candidate_score > max_existing_score);
#else
        (void)candidate_vec;
        (void)anchor_vec;
        (void)anchor_neighbors;
        (void)load_vector;
        (void)avg_out_degree;
        (void)median_neighbor_distance;
        return true;
#endif
    }

  private:
    float compute_score(const T *anchor_vec, const T *target_vec, float median_neighbor_distance) const
    {
        const float direction = compute_reverse_grade(anchor_vec, target_vec);
        const float distance = compute_distance(anchor_vec, target_vec);
        return (distance / median_neighbor_distance) * direction;
    }

    float compute_reverse_grade(const T *anchor_vec, const T *target_vec) const
    {
        return (1.0f - compute_direction_cosine(anchor_vec, target_vec)) * 0.5f;
    }

    float compute_direction_cosine(const T *anchor_vec, const T *target_vec) const
    {
        float dot = 0.0f;
        float norm_to_centroid = 0.0f;
        float norm_to_target = 0.0f;
        for (size_t d = 0; d < _dim; ++d)
        {
            const float to_centroid = _global_centroid[d] - static_cast<float>(anchor_vec[d]);
            const float to_target = static_cast<float>(target_vec[d]) - static_cast<float>(anchor_vec[d]);
            dot += to_centroid * to_target;
            norm_to_centroid += to_centroid * to_centroid;
            norm_to_target += to_target * to_target;
        }
        if (norm_to_centroid == 0.0f || norm_to_target == 0.0f)
        {
            return 0.0f;
        }
        return dot / (std::sqrt(norm_to_centroid) * std::sqrt(norm_to_target));
    }

    float compute_distance(const T *lhs, const T *rhs) const
    {
        float sum = 0.0f;
        for (size_t d = 0; d < _dim; ++d)
        {
            const float diff = static_cast<float>(lhs[d]) - static_cast<float>(rhs[d]);
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    static float compute_median(std::vector<float> values)
    {
        const size_t mid = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + mid, values.end());
        float median = values[mid];
        if (values.size() % 2 == 0)
        {
            std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
            median = (median + values[mid - 1]) * 0.5f;
        }
        return median;
    }

    static float compute_mad(const std::vector<float> &values, float median)
    {
        std::vector<float> deviations;
        deviations.reserve(values.size());
        for (float value : values)
        {
            deviations.push_back(std::fabs(value - median));
        }
        return compute_median(std::move(deviations));
    }

    static float compute_stddev(const std::vector<float> &values, float center)
    {
        if (values.empty())
        {
            return 0.0f;
        }
        float sum = 0.0f;
        for (float value : values)
        {
            const float diff = value - center;
            sum += diff * diff;
        }
        return std::sqrt(sum / static_cast<float>(values.size()));
    }

    AdaptiveReversePruneParams _params;
    const std::vector<float> &_global_centroid;
    size_t _dim;
    uint32_t _max_degree;
};

} // namespace diskann
