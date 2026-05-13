// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <omp.h>

#include "utils.h"

namespace po = boost::program_options;

namespace
{
template <typename T>
void reorder_dataset(const std::string &input_path, const std::string &output_path, uint32_t num_threads)
{
    std::unique_ptr<T[]> data;
    size_t num_points, dim;
    diskann::load_bin<T>(input_path, data, num_points, dim);

    if (num_points == 0)
    {
        throw diskann::ANNException("Input dataset is empty", -1);
    }

    omp_set_num_threads(num_threads == 0 ? omp_get_num_procs() : num_threads);

    std::vector<double> centroid(dim, 0.0);
#pragma omp parallel
    {
        std::vector<double> local_centroid(dim, 0.0);
#pragma omp for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(num_points); ++i)
        {
            const T *row = data.get() + static_cast<size_t>(i) * dim;
            for (size_t j = 0; j < dim; ++j)
            {
                local_centroid[j] += static_cast<double>(row[j]);
            }
        }
#pragma omp critical
        {
            for (size_t j = 0; j < dim; ++j)
            {
                centroid[j] += local_centroid[j];
            }
        }
    }

    for (size_t j = 0; j < dim; ++j)
    {
        centroid[j] /= static_cast<double>(num_points);
    }

    uint32_t anchor_id = 0;
    double anchor_dist = std::numeric_limits<double>::max();
#pragma omp parallel
    {
        uint32_t local_anchor_id = 0;
        double local_anchor_dist = std::numeric_limits<double>::max();
#pragma omp for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(num_points); ++i)
        {
            const T *row = data.get() + static_cast<size_t>(i) * dim;
            double dist = 0.0;
            for (size_t j = 0; j < dim; ++j)
            {
                const double diff = centroid[j] - static_cast<double>(row[j]);
                dist += diff * diff;
            }
            if (dist < local_anchor_dist || (dist == local_anchor_dist && static_cast<uint32_t>(i) < local_anchor_id))
            {
                local_anchor_dist = dist;
                local_anchor_id = static_cast<uint32_t>(i);
            }
        }
#pragma omp critical
        {
            if (local_anchor_dist < anchor_dist || (local_anchor_dist == anchor_dist && local_anchor_id < anchor_id))
            {
                anchor_dist = local_anchor_dist;
                anchor_id = local_anchor_id;
            }
        }
    }

    std::vector<float> distances(num_points, 0.0f);
    const T *anchor = data.get() + static_cast<size_t>(anchor_id) * dim;
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(num_points); ++i)
    {
        if (static_cast<uint32_t>(i) == anchor_id)
        {
            distances[static_cast<size_t>(i)] = -1.0f;
            continue;
        }

        const T *row = data.get() + static_cast<size_t>(i) * dim;
        float dist = 0.0f;
        for (size_t j = 0; j < dim; ++j)
        {
            const float diff = static_cast<float>(row[j]) - static_cast<float>(anchor[j]);
            dist += diff * diff;
        }
        distances[static_cast<size_t>(i)] = dist;
    }

    std::vector<uint32_t> permutation(num_points);
    permutation[0] = anchor_id;
    for (uint32_t i = 0, pos = 1; i < static_cast<uint32_t>(num_points); ++i)
    {
        if (i != anchor_id)
        {
            permutation[pos++] = i;
        }
    }

    std::sort(permutation.begin() + 1, permutation.end(), [&distances](uint32_t left, uint32_t right) {
        if (distances[left] == distances[right])
        {
            return left < right;
        }
        return distances[left] < distances[right];
    });

    std::ofstream writer(output_path, std::ios::binary | std::ios::trunc);
    writer.exceptions(std::ofstream::failbit | std::ofstream::badbit);

    const int32_t npts_i32 = static_cast<int32_t>(num_points);
    const int32_t dim_i32 = static_cast<int32_t>(dim);
    writer.write(reinterpret_cast<const char *>(&npts_i32), sizeof(int32_t));
    writer.write(reinterpret_cast<const char *>(&dim_i32), sizeof(int32_t));

    for (size_t new_id = 0; new_id < num_points; ++new_id)
    {
        const uint32_t old_id = permutation[new_id];
        const T *row = data.get() + static_cast<size_t>(old_id) * dim;
        writer.write(reinterpret_cast<const char *>(row), static_cast<std::streamsize>(dim * sizeof(T)));
    }

    writer.close();

    std::cout << "Reordered dataset written to " << output_path << std::endl;
    std::cout << "Anchor original id: " << anchor_id << std::endl;
    std::cout << "Anchor distance to centroid: " << anchor_dist << std::endl;
}
} // namespace

int main(int argc, char **argv)
{
    std::string data_type, input_path, output_path;
    uint32_t num_threads;

    po::options_description desc("Reorder dataset by centroid-nearest anchor and anchor distance.");
    desc.add_options()("help,h", "Print information on arguments")(
        "data_type", po::value<std::string>(&data_type)->required(), "data type, one of {int8, uint8, float}")(
        "input_file", po::value<std::string>(&input_path)->required(), "Input dataset file in .bin format")(
        "output_file", po::value<std::string>(&output_path)->required(), "Output reordered dataset file in .bin format")(
        "num_threads,T", po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()), "Number of threads");

    try
    {
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);

        if (data_type == "float")
        {
            reorder_dataset<float>(input_path, output_path, num_threads);
        }
        else if (data_type == "int8")
        {
            reorder_dataset<int8_t>(input_path, output_path, num_threads);
        }
        else if (data_type == "uint8")
        {
            reorder_dataset<uint8_t>(input_path, output_path, num_threads);
        }
        else
        {
            std::cerr << "Unsupported data_type: " << data_type << std::endl;
            return -1;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
