# DiskANN Extensions: Reorder, Short-Edge Augmentation, and Adaptive Reverse Prune

This document explains the three graph-construction extensions added in this fork, how to enable them, what each parameter means, and how to run the main experiment configurations.

## Overview

This fork adds three build-time methods on top of the in-memory DiskANN graph builder:

1. **Reorder-based start-point control**
2. **Short-edge augmentation**
   - `exact`
   - `approx`
   - density-adaptive per-node edge count
3. **Adaptive reverse-edge pruning**

These methods are designed to be independently switchable so that ablation experiments can be run cleanly.

---

## 1. Reorder-based start-point control

### What it does

The reorder pipeline reorders the dataset offline, then forces graph construction to start from the first vector of the reordered dataset.

This does **not** happen automatically inside `build_memory_index`.
You must:

1. generate a reordered dataset first
2. pass that reordered dataset to `build_memory_index`
3. enable `--force_reordered_start`

### Reorder utility

```bash
/root/DiskANN/build/apps/reorder_by_centroid \
  --data_type float \
  --input_file /root/DiskANN/dataset/gist/gist_base.fbin \
  --output_file /root/DiskANN/results/gist_reordered_data/gist_base_reordered.fbin \
  --num_threads 16
```

### Build switch

```bash
--force_reordered_start
```

### Effect

When enabled, the graph entry point is forced to node `0` of the current input file.
This is intended to be used with a reordered dataset.

---

## 2. Short-edge augmentation

### What it does

After the normal Vamana graph is built, the index can add extra short edges to improve local connectivity.

This happens **after** the main `link()` stage.

### Modes

```bash
--short_edge_mode none|exact|approx
```

- `none`: no short-edge augmentation
- `exact`: use an external exact-KNN file as the candidate source
- `approx`: search the current graph itself to get approximate local candidates

### Common parameters

#### `--short_edge_n_samples`
Number of sampled nodes used to estimate the global radius.

- Larger value: more stable estimate, slower build
- Smaller value: faster build, noisier estimate

#### `--short_edge_max_candidates`
Maximum number of extra short edges that can be added to a single node.

This is the **upper bound** for density-adaptive augmentation.
If this is `0`, short-edge augmentation is effectively disabled.

#### `--short_edge_alpha`
Controls how aggressively local density is mapped to added-edge count through a sigmoid.

- Larger value: denser regions receive more extra edges, more aggressively
- Smaller value: smoother and flatter allocation

### Approx mode parameter

#### `--short_edge_approx_L`
Search width used when `approx` mode queries the current graph for candidate short neighbors.

- Larger value: candidates closer to exact, slower build
- Smaller value: faster build, lower-quality candidates

### Exact mode parameter

#### `--short_edge_exact_path`
Path to the exact-KNN text file used when `--short_edge_mode exact`.

**Important:** the point IDs in this file must match the current dataset order.
That means:

- original-order build -> original-order exact-KNN file
- reordered build -> reordered exact-KNN file

### Density-adaptive edge count

Short-edge augmentation does **not** add the same number of edges to every node.
The per-node edge count is computed adaptively from local density.

In practice, the final number of added edges per node is:

- at least `0`
- at most `--short_edge_max_candidates`

So yes: **there is a maximum number limit**, and it is controlled by `--short_edge_max_candidates`.

### Example: reorder + approx short-edge

```bash
/root/DiskANN/build/apps/build_memory_index \
  --data_type float \
  --dist_fn l2 \
  --data_path /root/DiskANN/results/gist_reordered_data/gist_base_reordered.fbin \
  --index_path_prefix /root/DiskANN/results/gist_aug_approx/index_gist_R32_L50_A1.2 \
  -T 16 -R 32 -L 50 --alpha 1.2 \
  --force_reordered_start \
  --short_edge_mode approx \
  --short_edge_n_samples 512 \
  --short_edge_max_candidates 8 \
  --short_edge_approx_L 50 \
  --short_edge_alpha 0.1
```

### Example: reorder + exact short-edge

```bash
/root/DiskANN/build/apps/build_memory_index \
  --data_type float \
  --dist_fn l2 \
  --data_path /root/DiskANN/results/gist_reordered_data/gist_base_reordered.fbin \
  --index_path_prefix /root/DiskANN/results/gist_aug_exact/index_gist_R32_L50_A1.2 \
  -T 16 -R 32 -L 50 --alpha 1.2 \
  --force_reordered_start \
  --short_edge_mode exact \
  --short_edge_exact_path /root/DiskANN/dataset/gist/gist_centroid_entry_reorder_gt_K50_all.txt \
  --short_edge_n_samples 512 \
  --short_edge_max_candidates 8 \
  --short_edge_alpha 0.1
```

---

## 3. Adaptive reverse-edge pruning

### What it does

During reverse-edge insertion, DiskANN normally adds reverse links from a selected neighbor back to the current node.

This extension rejects reverse edges that are both:

- sufficiently **reverse-oriented** relative to the global centroid direction
- sufficiently **long** relative to the local neighbor-distance scale
- sufficiently **abnormal** under a robust score threshold

This is intended to remove reverse edges that are unlikely to help greedy graph traversal.

### Enable switch

```bash
--enable_adaptive_reverse_prune
```

### Reorder requirement

Adaptive reverse prune is intended for reordered input.
In this implementation, if adaptive reverse prune is enabled, you must also use:

```bash
--force_reordered_start
```

and provide a reordered base file.

### Parameters

#### `--adaptive_reverse_prune_alpha`
Controls the trigger threshold for when a node has enough existing neighbors to start checking reverse-edge rejection.

Larger value -> more conservative checking  
Smaller value -> more aggressive checking

#### `--adaptive_reverse_prune_k_mad`
Controls the robust cutoff:

- cutoff = median(score) + k * MAD(score)

Larger value -> harder to reject edges  
Smaller value -> easier to reject edges

#### `--adaptive_reverse_prune_r_min_ratio`
Sets the minimum degree ratio relative to `R` before reverse-prune checks begin.

Larger value -> more conservative  
Smaller value -> more aggressive

#### `--adaptive_reverse_prune_min_neighbors_to_check`
Minimum existing out-neighbor count required before checking whether a reverse edge should be rejected.

Larger value -> more nodes bypass checking  
Smaller value -> more nodes enter checking

#### `--adaptive_reverse_prune_strict_require_reorder`
This flag is exposed for completeness and future control.
In practice, adaptive reverse prune should already be treated as requiring reordered input and `--force_reordered_start`.

### Current conservative rejection rule

The current implementation only rejects a reverse edge when all of the following are true:

1. direction cosine is negative
2. `len_grade > 1.5`
3. candidate score is above the robust cutoff
4. candidate score is worse than every existing neighbor score

This is the conservative version that gave the most stable recall/QPS tradeoff in our current experiments.

### Example: reorder + adaptive reverse prune

```bash
/root/DiskANN/build/apps/build_memory_index \
  --data_type float \
  --dist_fn l2 \
  --data_path /root/DiskANN/results/gist_reordered_data/gist_base_reordered.fbin \
  --index_path_prefix /root/DiskANN/results/gist_adaptive_reverse/index_gist_R32_L50_A1.2 \
  -T 16 -R 32 -L 50 --alpha 1.2 \
  --force_reordered_start \
  --enable_adaptive_reverse_prune \
  --adaptive_reverse_prune_alpha 1.0 \
  --adaptive_reverse_prune_k_mad 1.0 \
  --adaptive_reverse_prune_r_min_ratio 0.5 \
  --adaptive_reverse_prune_min_neighbors_to_check 4
```

---

## 4. Combining methods

The current build order is:

1. load input data
2. initialize adaptive reverse prune state if enabled
3. build the main graph with `link()`
4. apply short-edge augmentation if enabled
5. save the index

This means the supported combinations are:

- base
- base + approx
- base + exact
- reorder
- reorder + approx
- reorder + exact
- reorder + adaptive_reverse
- reorder + adaptive_reverse + approx
- reorder + adaptive_reverse + exact

---

## 5. Convenience script

Use:

```bash
/root/DiskANN/scripts/run_gist_in_memory.sh
```

This script supports:

- single builds/searches
- short-edge parameter forwarding
- adaptive reverse prune parameter forwarding
- a 9-cell ablation matrix mode

### Single-run example

```bash
/root/DiskANN/scripts/run_gist_in_memory.sh \
  --base_file /root/DiskANN/results/gist_reordered_data/gist_base_reordered.fbin \
  --gt_file /root/DiskANN/results/gist_reordered_data/gist_reordered_gt10 \
  --result_dir /root/DiskANN/results/gist_adaptive_reverse_conservative \
  --force_reordered_start \
  --enable_adaptive_reverse_prune \
  --adaptive_reverse_prune_alpha 1.0 \
  --adaptive_reverse_prune_k_mad 1.0 \
  --adaptive_reverse_prune_r_min_ratio 0.5 \
  --adaptive_reverse_prune_min_neighbors_to_check 4 \
  --search_ls "100 150 200 250 300 350 400 500 600 700 800 900 1000 1500 2000"
```

### Full 9-cell ablation example

```bash
/root/DiskANN/scripts/run_gist_in_memory.sh \
  --ablation_matrix \
  --original_base_file /root/DiskANN/dataset/gist/gist_base.fbin \
  --reordered_base_file /root/DiskANN/results/gist_reordered_data/gist_base_reordered.fbin \
  --original_gt_file /root/DiskANN/dataset/gist/gist_gt10 \
  --reordered_gt_file /root/DiskANN/results/gist_reordered_data/gist_reordered_gt10 \
  --original_exact_knn_path /root/DiskANN/dataset/gist/gist100w_base_gt_K50_all.txt \
  --reordered_exact_knn_path /root/DiskANN/dataset/gist/gist_centroid_entry_reorder_gt_K50_all.txt \
  --ablation_root_dir /root/DiskANN/results/gist_ablation \
  --search_ls "100 150 200 250 300 350 400 500 600 700 800 900 1000 1500 2000"
```

The script creates one directory per configuration and stores:

- build log
- search log
- build command
- module flag summary

---

## 6. Parameter summary table

| Module | Parameter | Meaning | Larger value tends to... |
|---|---|---|---|
| reorder | `--force_reordered_start` | use node 0 as the entry point | enforce reordered-start semantics |
| short-edge | `--short_edge_mode` | candidate source / enable mode | switch algorithm behavior |
| short-edge | `--short_edge_exact_path` | exact-KNN source file | N/A |
| short-edge | `--short_edge_n_samples` | samples for global-radius estimate | stabilize estimate, slow build |
| short-edge | `--short_edge_max_candidates` | max added edges per node | densify graph |
| short-edge | `--short_edge_approx_L` | search width in approx mode | improve candidate quality, slow build |
| short-edge | `--short_edge_alpha` | density-to-count sigmoid sharpness | increase density sensitivity |
| adaptive reverse | `--enable_adaptive_reverse_prune` | enable reverse-edge filtering | turn method on |
| adaptive reverse | `--adaptive_reverse_prune_alpha` | trigger threshold multiplier | make checking more conservative |
| adaptive reverse | `--adaptive_reverse_prune_k_mad` | robust cutoff multiplier | reject fewer edges |
| adaptive reverse | `--adaptive_reverse_prune_r_min_ratio` | minimum degree ratio before checks | delay checking |
| adaptive reverse | `--adaptive_reverse_prune_min_neighbors_to_check` | minimum neighbors before checks | bypass more low-degree nodes |
| adaptive reverse | `--adaptive_reverse_prune_strict_require_reorder` | reorder requirement control | reserved / semantic guard |

---

## 7. Practical guidance

### If you want the highest recall
Use:
- reorder + exact short-edge
- or reorder + adaptive_reverse + exact

### If you want to avoid external exact KNN files
Use:
- reorder + approx short-edge
- or reorder + adaptive_reverse + approx

### If you want a lightweight graph change with minimal degree inflation
Use:
- reorder + adaptive reverse prune

### If you want the cleanest ablation baseline
Use:
- base
- reorder
- base+exact
- reorder+exact
- reorder+adaptive_reverse

These are usually the most interpretable comparison points.
