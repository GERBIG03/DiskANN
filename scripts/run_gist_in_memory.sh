#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/DiskANN"
BUILD_DIR="$ROOT/build/apps"
DATA_DIR="$ROOT/dataset/gist"
RESULT_DIR="$ROOT/results/gist_in_memory"
THREADS="$(nproc)"
R="32"
LBUILD="50"
ALPHA="1.2"
K="10"
SEARCH_LS=(10 20 50 100)
DIST_FN="l2"
DATA_TYPE="float"
BASE_FILE="$DATA_DIR/gist_base.fbin"
QUERY_FILE="$DATA_DIR/gist_query.fbin"
GT_FILE="$DATA_DIR/gist_gt10"
SKIP_BUILD="0"
SKIP_SEARCH="0"
FORCE_REORDERED_START="0"
SHORT_EDGE_MODE="none"
SHORT_EDGE_EXACT_PATH=""
SHORT_EDGE_N_SAMPLES="1024"
SHORT_EDGE_MAX_CANDIDATES="0"
SHORT_EDGE_APPROX_L="50"
SHORT_EDGE_ALPHA="1.0"
ENABLE_ADAPTIVE_REVERSE_PRUNE="0"
ADAPTIVE_REVERSE_PRUNE_STRICT_REQUIRE_REORDER="0"
ADAPTIVE_REVERSE_PRUNE_ALPHA="1.0"
ADAPTIVE_REVERSE_PRUNE_K_MAD="1.0"
ADAPTIVE_REVERSE_PRUNE_R_MIN_RATIO="0.5"
ADAPTIVE_REVERSE_PRUNE_MIN_NEIGHBORS_TO_CHECK="4"
ABLATION_MATRIX="0"
ORIGINAL_BASE_FILE=""
REORDERED_BASE_FILE=""
EXACT_KNN_PATH=""
ORIGINAL_GT_FILE=""
REORDERED_GT_FILE=""
ORIGINAL_EXACT_KNN_PATH=""
REORDERED_EXACT_KNN_PATH=""
ABLATION_ROOT_DIR=""

usage() {
  cat <<'EOF'
Usage:
  run_gist_in_memory.sh [options]

Options:
  --result_dir PATH        Output directory for index, logs, and search results
  --base_file PATH         Base dataset file (.fbin)
  --query_file PATH        Query dataset file (.fbin)
  --gt_file PATH           Ground truth file
  --threads N              Number of threads for build/search
  -R, --max_degree N       Graph max degree
  -Lbuild, --Lbuild N      Build search list size
  --alpha FLOAT            Build alpha
  -K, --topk N             Number of neighbors to retrieve
  --search_ls "A B C"      Search L list as a quoted space-separated string
  --dist_fn NAME           Distance function (default: l2)
  --data_type NAME         Data type (default: float)
  --force_reordered_start  Force the entry point to use the first vector in the input dataset
  --short_edge_mode MODE   Short-edge augmentation mode: none, exact, approx
  --short_edge_exact_path  Path to exact KNN text file for short-edge augmentation
  --short_edge_n_samples N Number of sampled nodes for global radius estimation
  --short_edge_max_candidates N  Upper bound on extra short edges added per node
  --short_edge_approx_L N  Search list size for approximate short-edge augmentation
  --short_edge_alpha FLOAT Sigmoid sharpness for density-adaptive short-edge augmentation
  --enable_adaptive_reverse_prune  Enable adaptive reverse prune during reverse-edge insertion
  --adaptive_reverse_prune_strict_require_reorder  Require reordered input when adaptive reverse prune is enabled
  --adaptive_reverse_prune_alpha FLOAT  Trigger multiplier for adaptive reverse prune
  --adaptive_reverse_prune_k_mad FLOAT  MAD multiplier for adaptive reverse prune cutoff
  --adaptive_reverse_prune_r_min_ratio FLOAT  Minimum degree ratio before adaptive reverse prune checks
  --adaptive_reverse_prune_min_neighbors_to_check N  Minimum neighbors before adaptive reverse prune checks
  --ablation_matrix       Run the 9-cell ablation matrix
  --original_base_file PATH  Original-order base dataset for ablation runs
  --reordered_base_file PATH Reordered base dataset for ablation runs
  --exact_knn_path PATH   Exact KNN file for exact short-edge augmentation
  --original_gt_file PATH Ground truth for original-order ablation runs
  --reordered_gt_file PATH Ground truth for reordered ablation runs
  --original_exact_knn_path PATH Exact KNN file for original-order exact augmentation
  --reordered_exact_knn_path PATH Exact KNN file for reordered exact augmentation
  --ablation_root_dir PATH Root output directory for ablation runs
  --skip_build             Skip index build and only run search
  --skip_search            Skip search and only build index
  -h, --help               Show this help message

Examples:
  /root/DiskANN/scripts/run_gist_in_memory.sh
  /root/DiskANN/scripts/run_gist_in_memory.sh --threads 8 -R 64 -Lbuild 100 --search_ls "50 100 150"
  /root/DiskANN/scripts/run_gist_in_memory.sh --result_dir /root/DiskANN/results/gist_r64 --skip_build
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --result_dir)
      RESULT_DIR="$2"
      shift 2
      ;;
    --base_file)
      BASE_FILE="$2"
      shift 2
      ;;
    --query_file)
      QUERY_FILE="$2"
      shift 2
      ;;
    --gt_file)
      GT_FILE="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    -R|--max_degree)
      R="$2"
      shift 2
      ;;
    -Lbuild|--Lbuild)
      LBUILD="$2"
      shift 2
      ;;
    --alpha)
      ALPHA="$2"
      shift 2
      ;;
    -K|--topk)
      K="$2"
      shift 2
      ;;
    --search_ls)
      read -r -a SEARCH_LS <<< "$2"
      shift 2
      ;;
    --dist_fn)
      DIST_FN="$2"
      shift 2
      ;;
    --data_type)
      DATA_TYPE="$2"
      shift 2
      ;;
    --force_reordered_start)
      FORCE_REORDERED_START="1"
      shift
      ;;
    --short_edge_mode)
      SHORT_EDGE_MODE="$2"
      shift 2
      ;;
    --short_edge_exact_path)
      SHORT_EDGE_EXACT_PATH="$2"
      shift 2
      ;;
    --short_edge_n_samples)
      SHORT_EDGE_N_SAMPLES="$2"
      shift 2
      ;;
    --short_edge_max_candidates)
      SHORT_EDGE_MAX_CANDIDATES="$2"
      shift 2
      ;;
    --short_edge_approx_L)
      SHORT_EDGE_APPROX_L="$2"
      shift 2
      ;;
    --short_edge_alpha)
      SHORT_EDGE_ALPHA="$2"
      shift 2
      ;;
    --enable_adaptive_reverse_prune)
      ENABLE_ADAPTIVE_REVERSE_PRUNE="1"
      shift
      ;;
    --adaptive_reverse_prune_strict_require_reorder)
      ADAPTIVE_REVERSE_PRUNE_STRICT_REQUIRE_REORDER="1"
      shift
      ;;
    --adaptive_reverse_prune_alpha)
      ADAPTIVE_REVERSE_PRUNE_ALPHA="$2"
      shift 2
      ;;
    --adaptive_reverse_prune_k_mad)
      ADAPTIVE_REVERSE_PRUNE_K_MAD="$2"
      shift 2
      ;;
    --adaptive_reverse_prune_r_min_ratio)
      ADAPTIVE_REVERSE_PRUNE_R_MIN_RATIO="$2"
      shift 2
      ;;
    --adaptive_reverse_prune_min_neighbors_to_check)
      ADAPTIVE_REVERSE_PRUNE_MIN_NEIGHBORS_TO_CHECK="$2"
      shift 2
      ;;
    --ablation_matrix)
      ABLATION_MATRIX="1"
      shift
      ;;
    --original_base_file)
      ORIGINAL_BASE_FILE="$2"
      shift 2
      ;;
    --reordered_base_file)
      REORDERED_BASE_FILE="$2"
      shift 2
      ;;
    --exact_knn_path)
      EXACT_KNN_PATH="$2"
      shift 2
      ;;
    --original_gt_file)
      ORIGINAL_GT_FILE="$2"
      shift 2
      ;;
    --reordered_gt_file)
      REORDERED_GT_FILE="$2"
      shift 2
      ;;
    --original_exact_knn_path)
      ORIGINAL_EXACT_KNN_PATH="$2"
      shift 2
      ;;
    --reordered_exact_knn_path)
      REORDERED_EXACT_KNN_PATH="$2"
      shift 2
      ;;
    --ablation_root_dir)
      ABLATION_ROOT_DIR="$2"
      shift 2
      ;;
    --skip_build)
      SKIP_BUILD="1"
      shift
      ;;
    --skip_search)
      SKIP_SEARCH="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

INDEX_PREFIX="$RESULT_DIR/index_gist_R${R}_L${LBUILD}_A${ALPHA}"
SEARCH_RESULT_PREFIX="$RESULT_DIR/search_gist"
LOG_DIR="$RESULT_DIR/logs"
BUILD_BIN="$BUILD_DIR/build_memory_index"
SEARCH_BIN="$BUILD_DIR/search_memory_index"

run_single_configuration() {
  local config_name="$1"
  local config_base_file="$2"
  local config_force_reordered_start="$3"
  local config_short_edge_mode="$4"
  local config_enable_adaptive_reverse_prune="$5"
  local config_gt_file="$6"
  local config_exact_knn_path="$7"
  local config_result_dir="$8"

  local config_log_dir="$config_result_dir/logs"
  local config_index_prefix="$config_result_dir/index_gist_R${R}_L${LBUILD}_A${ALPHA}"
  local config_search_result_prefix="$config_result_dir/search_gist"
  local config_build_log="$config_log_dir/build.log"
  local config_search_log="$config_log_dir/search.log"
  local config_build_cmd="$config_log_dir/build_cmd.txt"
  local config_flags_file="$config_log_dir/module_flags.txt"

  mkdir -p "$config_result_dir" "$config_log_dir"

  printf 'config=%s\nbase_file=%s\nreordered=%s\nforce_reordered_start=%s\nshort_edge_mode=%s\nadaptive_reverse=%s\ngt_file=%s\nexact_knn_path=%s\n' \
    "$config_name" \
    "$config_base_file" \
    "$(if [[ "$config_force_reordered_start" == "1" ]]; then printf 'yes'; else printf 'no'; fi)" \
    "$(if [[ "$config_force_reordered_start" == "1" ]]; then printf 'yes'; else printf 'no'; fi)" \
    "$config_short_edge_mode" \
    "$(if [[ "$config_enable_adaptive_reverse_prune" == "1" ]]; then printf 'yes'; else printf 'no'; fi)" \
    "$config_gt_file" \
    "$config_exact_knn_path" \
    > "$config_flags_file"

  local -a config_build_args=(
    --data_type "$DATA_TYPE"
    --dist_fn "$DIST_FN"
    --data_path "$config_base_file"
    --index_path_prefix "$config_index_prefix"
    -T "$THREADS"
    -R "$R"
    -L "$LBUILD"
    --alpha "$ALPHA"
    --short_edge_mode "$config_short_edge_mode"
    --short_edge_n_samples "$SHORT_EDGE_N_SAMPLES"
    --short_edge_max_candidates "$SHORT_EDGE_MAX_CANDIDATES"
    --short_edge_approx_L "$SHORT_EDGE_APPROX_L"
    --short_edge_alpha "$SHORT_EDGE_ALPHA"
    --adaptive_reverse_prune_alpha "$ADAPTIVE_REVERSE_PRUNE_ALPHA"
    --adaptive_reverse_prune_k_mad "$ADAPTIVE_REVERSE_PRUNE_K_MAD"
    --adaptive_reverse_prune_r_min_ratio "$ADAPTIVE_REVERSE_PRUNE_R_MIN_RATIO"
    --adaptive_reverse_prune_min_neighbors_to_check "$ADAPTIVE_REVERSE_PRUNE_MIN_NEIGHBORS_TO_CHECK"
  )

  if [[ "$config_force_reordered_start" == "1" ]]; then
    config_build_args+=(--force_reordered_start)
  fi
  if [[ "$config_enable_adaptive_reverse_prune" == "1" ]]; then
    config_build_args+=(--enable_adaptive_reverse_prune)
    if [[ "$ADAPTIVE_REVERSE_PRUNE_STRICT_REQUIRE_REORDER" == "1" ]]; then
      config_build_args+=(--adaptive_reverse_prune_strict_require_reorder)
    fi
  fi
  if [[ "$config_short_edge_mode" == "exact" ]]; then
    config_build_args+=(--short_edge_exact_path "$config_exact_knn_path")
  elif [[ -n "$SHORT_EDGE_EXACT_PATH" ]]; then
    config_build_args+=(--short_edge_exact_path "$SHORT_EDGE_EXACT_PATH")
  fi

  printf '%q ' "$BUILD_BIN" "${config_build_args[@]}" > "$config_build_cmd"
  printf '\n' >> "$config_build_cmd"

  if [[ "$SKIP_BUILD" != "1" ]]; then
    printf 'Running ablation config=%s\n' "$config_name"
    "$BUILD_BIN" "${config_build_args[@]}" | tee "$config_build_log"
  fi

  if [[ "$SKIP_SEARCH" != "1" ]]; then
    "$SEARCH_BIN" \
      --data_type "$DATA_TYPE" \
      --dist_fn "$DIST_FN" \
      --index_path_prefix "$config_index_prefix" \
      --query_file "$QUERY_FILE" \
      --gt_file "$config_gt_file" \
      --result_path "$config_search_result_prefix" \
      -T "$THREADS" \
      -K "$K" \
      -L "${SEARCH_LS[@]}" | tee "$config_search_log"
  fi
}

run_ablation_matrix() {
  local root_dir="$ABLATION_ROOT_DIR"
  if [[ -z "$root_dir" ]]; then
    root_dir="$ROOT/results/gist_ablation"
  fi
  if [[ -z "$ORIGINAL_BASE_FILE" || -z "$REORDERED_BASE_FILE" || -z "$ORIGINAL_GT_FILE" || -z "$REORDERED_GT_FILE" || -z "$ORIGINAL_EXACT_KNN_PATH" || -z "$REORDERED_EXACT_KNN_PATH" ]]; then
    echo "Missing required ablation inputs: --original_base_file, --reordered_base_file, --original_gt_file, --reordered_gt_file, --original_exact_knn_path, --reordered_exact_knn_path" >&2
    exit 1
  fi

  run_single_configuration "base" "$ORIGINAL_BASE_FILE" "0" "none" "0" "$ORIGINAL_GT_FILE" "" "$root_dir/base"
  run_single_configuration "base_approx" "$ORIGINAL_BASE_FILE" "0" "approx" "0" "$ORIGINAL_GT_FILE" "" "$root_dir/base_approx"
  run_single_configuration "base_exact" "$ORIGINAL_BASE_FILE" "0" "exact" "0" "$ORIGINAL_GT_FILE" "$ORIGINAL_EXACT_KNN_PATH" "$root_dir/base_exact"
  run_single_configuration "reorder" "$REORDERED_BASE_FILE" "1" "none" "0" "$REORDERED_GT_FILE" "" "$root_dir/reorder"
  run_single_configuration "reorder_approx" "$REORDERED_BASE_FILE" "1" "approx" "0" "$REORDERED_GT_FILE" "" "$root_dir/reorder_approx"
  run_single_configuration "reorder_exact" "$REORDERED_BASE_FILE" "1" "exact" "0" "$REORDERED_GT_FILE" "$REORDERED_EXACT_KNN_PATH" "$root_dir/reorder_exact"
  run_single_configuration "reorder_adaptive_reverse" "$REORDERED_BASE_FILE" "1" "none" "1" "$REORDERED_GT_FILE" "" "$root_dir/reorder_adaptive_reverse"
  run_single_configuration "reorder_adaptive_reverse_approx" "$REORDERED_BASE_FILE" "1" "approx" "1" "$REORDERED_GT_FILE" "" "$root_dir/reorder_adaptive_reverse_approx"
  run_single_configuration "reorder_adaptive_reverse_exact" "$REORDERED_BASE_FILE" "1" "exact" "1" "$REORDERED_GT_FILE" "$REORDERED_EXACT_KNN_PATH" "$root_dir/reorder_adaptive_reverse_exact"
}

mkdir -p "$RESULT_DIR" "$LOG_DIR"

for path in "$BUILD_BIN" "$SEARCH_BIN" "$BASE_FILE" "$QUERY_FILE"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

if [[ "$SKIP_SEARCH" != "1" && ! -e "$GT_FILE" ]]; then
  echo "Missing required path: $GT_FILE" >&2
  exit 1
fi

BUILD_LOG="$LOG_DIR/build_$(date +%Y%m%d_%H%M%S).log"
SEARCH_LOG="$LOG_DIR/search_$(date +%Y%m%d_%H%M%S).log"

if [[ "$ABLATION_MATRIX" == "1" ]]; then
  for path in "$ORIGINAL_BASE_FILE" "$REORDERED_BASE_FILE" "$ORIGINAL_GT_FILE" "$REORDERED_GT_FILE" "$ORIGINAL_EXACT_KNN_PATH" "$REORDERED_EXACT_KNN_PATH"; do
    if [[ ! -e "$path" ]]; then
      echo "Missing required ablation path: $path" >&2
      exit 1
    fi
  done
  run_ablation_matrix
  printf '\nDone. Ablation results are in %s\n' "${ABLATION_ROOT_DIR:-$ROOT/results/gist_ablation}"
  exit 0
fi

if [[ "$SKIP_BUILD" != "1" ]]; then
  printf 'Running build with threads=%s R=%s Lbuild=%s alpha=%s short_edge_mode=%s adaptive_reverse=%s\n' "$THREADS" "$R" "$LBUILD" "$ALPHA" "$SHORT_EDGE_MODE" "$ENABLE_ADAPTIVE_REVERSE_PRUNE"
  BUILD_ARGS=(
    --data_type "$DATA_TYPE"
    --dist_fn "$DIST_FN"
    --data_path "$BASE_FILE"
    --index_path_prefix "$INDEX_PREFIX"
    -T "$THREADS"
    -R "$R"
    -L "$LBUILD"
    --alpha "$ALPHA"
    --short_edge_mode "$SHORT_EDGE_MODE"
    --short_edge_n_samples "$SHORT_EDGE_N_SAMPLES"
    --short_edge_max_candidates "$SHORT_EDGE_MAX_CANDIDATES"
    --short_edge_approx_L "$SHORT_EDGE_APPROX_L"
    --short_edge_alpha "$SHORT_EDGE_ALPHA"
    --adaptive_reverse_prune_alpha "$ADAPTIVE_REVERSE_PRUNE_ALPHA"
    --adaptive_reverse_prune_k_mad "$ADAPTIVE_REVERSE_PRUNE_K_MAD"
    --adaptive_reverse_prune_r_min_ratio "$ADAPTIVE_REVERSE_PRUNE_R_MIN_RATIO"
    --adaptive_reverse_prune_min_neighbors_to_check "$ADAPTIVE_REVERSE_PRUNE_MIN_NEIGHBORS_TO_CHECK"
  )
  if [[ "$FORCE_REORDERED_START" == "1" ]]; then
    BUILD_ARGS+=(--force_reordered_start)
  fi
  if [[ "$ENABLE_ADAPTIVE_REVERSE_PRUNE" == "1" ]]; then
    BUILD_ARGS+=(--enable_adaptive_reverse_prune)
    if [[ "$ADAPTIVE_REVERSE_PRUNE_STRICT_REQUIRE_REORDER" == "1" ]]; then
      BUILD_ARGS+=(--adaptive_reverse_prune_strict_require_reorder)
    fi
  fi
  if [[ -n "$SHORT_EDGE_EXACT_PATH" ]]; then
    BUILD_ARGS+=(--short_edge_exact_path "$SHORT_EDGE_EXACT_PATH")
  fi
  "$BUILD_BIN" "${BUILD_ARGS[@]}" | tee "$BUILD_LOG"
fi

if [[ "$SKIP_SEARCH" != "1" ]]; then
  printf '\nRunning search with threads=%s K=%s Ls=%s\n' "$THREADS" "$K" "${SEARCH_LS[*]}"
  "$SEARCH_BIN" \
    --data_type "$DATA_TYPE" \
    --dist_fn "$DIST_FN" \
    --index_path_prefix "$INDEX_PREFIX" \
    --query_file "$QUERY_FILE" \
    --gt_file "$GT_FILE" \
    --result_path "$SEARCH_RESULT_PREFIX" \
    -T "$THREADS" \
    -K "$K" \
    -L "${SEARCH_LS[@]}" | tee "$SEARCH_LOG"
fi

printf '\nDone. Results are in %s\n' "$RESULT_DIR"
