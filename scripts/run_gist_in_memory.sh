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

if [[ "$SKIP_BUILD" != "1" ]]; then
  printf 'Running build with threads=%s R=%s Lbuild=%s alpha=%s short_edge_mode=%s\n' "$THREADS" "$R" "$LBUILD" "$ALPHA" "$SHORT_EDGE_MODE"
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
  )
  if [[ "$FORCE_REORDERED_START" == "1" ]]; then
    BUILD_ARGS+=(--force_reordered_start)
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
