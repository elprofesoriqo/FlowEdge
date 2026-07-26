#!/usr/bin/env bash
set -e

FILTER="${1:-}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BACKEND="${FLOWEDGE_BACKEND:-cpu}"
PY="$(command -v py || command -v python3 || command -v python)"

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT_BRANCH" == "main" ]]; then
    echo "You are already on main branch. Please run this from your PR branch." >&2
    exit 1
fi

# stash uncommitted changes
STASHED=0
if ! git diff-index --quiet HEAD --; then
    git stash -q
    STASHED=1
fi

# MinGW runtime DLLs live in Strawberry
case "$(uname -s)" in MINGW* | MSYS* | CYGWIN*) export PATH="/c/Strawberry/c/bin:$PATH" ;; esac

run_benchmark() {
    local branch=$1
    local out_csv=$2
    
    rm -rf build/CMakeCache.txt build/CMakeFiles/ build/_deps/googlebenchmark-subbuild/CMakeCache.txt build/_deps/googlebenchmark-subbuild/CMakeFiles/
    if ! cmake -B build -S . -DFLOWEDGE_BENCH=ON -DFLOWEDGE_BACKEND="${BACKEND}" -DCMAKE_BUILD_TYPE=Release > build/cmake_log.txt 2>&1; then
        exit 1
    fi
    
    if ! cmake --build build --config Release -j 4 > build/build_log.txt 2>&1; then
        exit 1
    fi
    local exe_k="./build/bench/flowedge_kernels_bench"
    local exe_e="./build/bench/flowedge_engine_bench"
    if [[ -f "./build/bench/flowedge_kernels_bench.exe" ]]; then
        exe_k="./build/bench/flowedge_kernels_bench.exe"
        exe_e="./build/bench/flowedge_engine_bench.exe"
    elif [[ -f "./build/flowedge_kernels_bench.exe" ]]; then
        exe_k="./build/flowedge_kernels_bench.exe"
        exe_e="./build/flowedge_engine_bench.exe"
    elif [[ -f "./build/flowedge_kernels_bench" ]]; then
        exe_k="./build/flowedge_kernels_bench"
        exe_e="./build/flowedge_engine_bench"
    fi
    local cmd_k="$exe_k --benchmark_out=${out_csv}_k --benchmark_out_format=csv"
    export FLOWEDGE_MODEL="$ROOT/models/mamba_flow.safetensors"
    local cmd_e="$exe_e --benchmark_out=${out_csv}_e --benchmark_out_format=csv"
    if [[ -n "$FILTER" ]]; then
        cmd_k="$cmd_k --benchmark_filter=$FILTER"
        cmd_e="$cmd_e --benchmark_filter=$FILTER"
    fi
    
    $cmd_k >/dev/null 2>&1 || true
    $cmd_e >/dev/null 2>&1 || true
    
    cat "${out_csv}_k" "${out_csv}_e" > "$out_csv" 2>/dev/null || true
}

cleanup() {
    git switch -q "$CURRENT_BRANCH"
    if [[ "$STASHED" -eq 1 ]]; then
        git stash pop -q
    fi
}
trap cleanup EXIT

MODEL="models/mamba_flow.safetensors"

if [[ ! -f "$MODEL" ]]; then
    mkdir -p models
    wget -qO "$MODEL" "https://huggingface.co/ReForceMind/mamba_flow/resolve/main/mamba_flow.safetensors"
fi

run_benchmark "$CURRENT_BRANCH" "$ROOT/build/current.csv"

git switch -q main
run_benchmark "main" "$ROOT/build/main.csv"
git switch -q "$CURRENT_BRANCH"

PT_FORWARD="N/A"
if [[ -f "$MODEL" ]]; then
    # Only try to fetch PyTorch baseline if we are benchmarking the engine macro
    if [[ "$FILTER" == *"engine"* ]] || [[ -z "$FILTER" ]]; then
        PT_RES=$("$PY" scripts/torch_ref.py bench "$MODEL" 1 | awk '/PyTorch forward:/ {print $3}')
        if [[ -n "$PT_RES" ]]; then
            PT_FORWARD="$PT_RES"
        fi
    fi
fi

awk -F, -v pt_fwd="$PT_FORWARD" '
BEGIN {
    print "### Microbenchmarks"
    print "| Benchmark | Main | Current | Speedup (vs Main) |"
    print "|-----------|------|---------|-------------------|"
}
NR==FNR {
    if ($1 ~ /^"BM_/) {
        name = $1; gsub(/"/, "", name)
        main_val[name] = $3
        unit[name] = $5
    }
    next
}
{
    if ($1 ~ /^"BM_/) {
        name = $1; gsub(/"/, "", name)
        cur_val = $3
        u = $5
        
        speedup = 0
        if (cur_val > 0) speedup = main_val[name] / cur_val
        
        if (name == "BM_engine_forward") {
            engine_main = main_val[name]
            engine_cur = cur_val
            engine_u = u
            engine_speedup = speedup
        } else {
            if (name in main_val) {
                printf "| %s | %.4f %s | %.4f %s | %.2fx |\n", name, main_val[name], u, cur_val, u, speedup
            }
        }
    }
}
END {
    if (engine_main != "") {
        print ""
        print "### Engine End-to-End"
        print "| PyTorch | FlowEdge (Main) | FlowEdge (Current) | Speedup (vs Main) | Speedup (vs PT) |"
        print "|---------|-----------------|--------------------|-----------------------|-----------------------|"
        
        pt = "N/A"
        speedup_pt = "N/A"
        if (pt_fwd != "N/A" && pt_fwd != "") {
            pt = sprintf("%.4f %s", pt_fwd, engine_u)
            if (engine_cur > 0) speedup_pt = sprintf("%.2fx", pt_fwd / engine_cur)
        }
        
        printf "| %s | %.4f %s | %.4f %s | %.2fx | %s |\n", pt, engine_main, engine_u, engine_cur, engine_u, engine_speedup, speedup_pt
    }
}
' "$ROOT/build/main.csv" "$ROOT/build/current.csv" > "$ROOT/ab_results.md"


