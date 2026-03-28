#!/usr/bin/env bash
#
# bench-inference.sh — End-to-end inference benchmark driver
#
# Builds ironmill, compiles models with various optimization combos,
# and runs the Swift inference harness on each compiled model.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_DIR="$ROOT_DIR/benches/swift/InferenceBench"
HARNESS=""
ITERATIONS="${BENCH_ITERATIONS:-200}"
WARMUP="${BENCH_WARMUP:-20}"
WORK_DIR="$ROOT_DIR/target/bench-inference"

MODELS=(
    "mobilenetv2.onnx"
    "squeezenet1.1.onnx"
)

# Each config: "label|extra_args"
CONFIGS=(
    "No optimization|--no-fusion"
    "Default (always-on)|"
    "+ FP16|--quantize fp16"
    "+ INT8|--quantize int8"
    "+ Palettize 4-bit|--palettize 4"
)

# ─── Helpers ────────────────────────────────────────────────────────────────

log() { printf '\033[1;34m==> %s\033[0m\n' "$*"; }
err() { printf '\033[1;31mError: %s\033[0m\n' "$*" >&2; exit 1; }

cleanup() {
    if [[ -d "$WORK_DIR" ]]; then
        log "Cleaning up $WORK_DIR"
        rm -rf "$WORK_DIR"
    fi
}
trap cleanup EXIT

fmt_ms() {
    # Format a millisecond value for display; pass through non-numeric values
    case "$1" in
        ''|*[!0-9.]*) printf '%s' "$1"; return ;;
    esac
    awk "BEGIN { v = $1; if (v < 1) printf \"%.1f\302\265s\", v*1000; else if (v < 1000) printf \"%.1fms\", v; else printf \"%.2fs\", v/1000 }"
}

# ─── Step 1: Build ironmill ─────────────────────────────────────────────────

log "Building ironmill (release)…"
cd "$ROOT_DIR"
cargo build --release --quiet 2>&1
IRONMILL="$ROOT_DIR/target/release/ironmill"
[[ -x "$IRONMILL" ]] || err "ironmill binary not found at $IRONMILL"
log "ironmill ready: $IRONMILL"

# ─── Step 2: Build Swift harness ────────────────────────────────────────────

log "Building Swift inference harness (release)…"
cd "$BENCH_DIR"
swift build -c release 2>&1
HARNESS="$(swift build -c release --show-bin-path)/InferenceBench"
[[ -x "$HARNESS" ]] || err "Swift harness binary not found"
log "Harness ready: $HARNESS"

# ─── Step 3: Check for coremlcompiler ───────────────────────────────────────

HAS_COMPILER=false
if xcrun --find coremlcompiler &>/dev/null; then
    HAS_COMPILER=true
    log "coremlcompiler found"
else
    err "coremlcompiler not found — install Xcode command-line tools"
fi

# ─── Step 4: Run benchmarks ─────────────────────────────────────────────────

mkdir -p "$WORK_DIR"

# Locate test models
MODEL_DIR="$ROOT_DIR/tests/fixtures"
if [[ ! -d "$MODEL_DIR" ]]; then
    MODEL_DIR="$ROOT_DIR/tests/data"
fi
if [[ ! -d "$MODEL_DIR" ]]; then
    # Try downloading fixtures if a script exists
    if [[ -x "$SCRIPT_DIR/download-fixtures.sh" ]]; then
        log "Downloading test fixtures…"
        "$SCRIPT_DIR/download-fixtures.sh"
        MODEL_DIR="$ROOT_DIR/tests/fixtures"
    fi
fi

print_separator() {
    local w=$1
    local char=$2
    printf '%s' "$char"
    printf '%*s' "$w" '' | tr ' ' '─'
    printf '%s\n' "$char"
}

for model_file in "${MODELS[@]}"; do
    model_path=""
    # Search for the model in common locations
    for dir in "$MODEL_DIR" "$ROOT_DIR/tests/fixtures" "$ROOT_DIR/tests/data" "$ROOT_DIR/models"; do
        if [[ -f "$dir/$model_file" ]]; then
            model_path="$dir/$model_file"
            break
        fi
    done

    if [[ -z "$model_path" ]]; then
        log "Skipping $model_file (not found)"
        continue
    fi

    model_name="${model_file%.onnx}"
    log "Benchmarking: $model_name"

    # Collect results for the comparison table
    declare -a cfg_labels=()
    declare -a cfg_p50=()
    declare -a cfg_p95=()
    declare -a cfg_p99=()
    declare -a cfg_load=()

    for config_entry in "${CONFIGS[@]}"; do
        IFS='|' read -r label extra_args <<< "$config_entry"
        safe_label="$(echo "$label" | tr ' +' '__' | tr -cd 'a-zA-Z0-9_-')"
        out_dir="$WORK_DIR/${model_name}_${safe_label}"
        mlpackage="$out_dir/${model_name}.mlpackage"
        mlmodelc="$out_dir/${model_name}.mlmodelc"

        mkdir -p "$out_dir"

        log "  Compiling: $label"
        # shellcheck disable=SC2086
        if ! "$IRONMILL" compile "$model_path" -o "$mlpackage" $extra_args 2>&1; then
            log "  ⚠ ironmill compile failed for '$label', skipping"
            continue
        fi

        if [[ ! -d "$mlpackage" ]]; then
            log "  ⚠ .mlpackage not produced for '$label', skipping"
            continue
        fi

        # Compile to .mlmodelc
        if $HAS_COMPILER; then
            if ! xcrun coremlcompiler compile "$mlpackage" "$out_dir" 2>&1; then
                log "  ⚠ coremlcompiler failed for '$label', skipping"
                continue
            fi
        fi

        if [[ ! -d "$mlmodelc" ]]; then
            # coremlcompiler may place it under a different name
            found_mlmodelc="$(find "$out_dir" -name '*.mlmodelc' -maxdepth 2 | head -1)"
            if [[ -n "$found_mlmodelc" ]]; then
                mlmodelc="$found_mlmodelc"
            else
                log "  ⚠ .mlmodelc not found for '$label', skipping"
                continue
            fi
        fi

        log "  Running inference ($ITERATIONS iterations)…"
        json_out="$out_dir/results.json"
        if "$HARNESS" "$mlmodelc" --iterations "$ITERATIONS" --warmup "$WARMUP" --json > "$json_out" 2>&1; then
            p50=$(python3 -c "import json,sys; d=json.load(open('$json_out')); print(d.get('median_ms',0))" 2>/dev/null || echo "0")
            p95=$(python3 -c "import json,sys; d=json.load(open('$json_out')); print(d.get('p95_ms',0))" 2>/dev/null || echo "0")
            p99=$(python3 -c "import json,sys; d=json.load(open('$json_out')); print(d.get('p99_ms',0))" 2>/dev/null || echo "0")
            load=$(python3 -c "import json,sys; d=json.load(open('$json_out')); print(d.get('load_time_ms',0))" 2>/dev/null || echo "0")
        else
            log "  ⚠ Inference failed for '$label'"
            p50="—"; p95="—"; p99="—"; load="—"
        fi

        cfg_labels+=("$label")
        cfg_p50+=("$p50")
        cfg_p95+=("$p95")
        cfg_p99+=("$p99")
        cfg_load+=("$load")
    done

    # ─── Print comparison table ─────────────────────────────────────────────

    echo ""
    echo "Model: $model_name"

    col0=22  # Configuration
    col1=8   # p50
    col2=8   # p95
    col3=8   # p99
    col4=8   # Load

    # Top border
    printf '┌%s┬%s┬%s┬%s┬%s┐\n' \
        "$(printf '%*s' $col0 '' | tr ' ' '─')" \
        "$(printf '%*s' $col1 '' | tr ' ' '─')" \
        "$(printf '%*s' $col2 '' | tr ' ' '─')" \
        "$(printf '%*s' $col3 '' | tr ' ' '─')" \
        "$(printf '%*s' $col4 '' | tr ' ' '─')"

    # Header
    printf '│ %-*s│ %-*s│ %-*s│ %-*s│ %-*s│\n' \
        $((col0-1)) "Configuration" \
        $((col1-1)) "p50" \
        $((col2-1)) "p95" \
        $((col3-1)) "p99" \
        $((col4-1)) "Load"

    # Header separator
    printf '├%s┼%s┼%s┼%s┼%s┤\n' \
        "$(printf '%*s' $col0 '' | tr ' ' '─')" \
        "$(printf '%*s' $col1 '' | tr ' ' '─')" \
        "$(printf '%*s' $col2 '' | tr ' ' '─')" \
        "$(printf '%*s' $col3 '' | tr ' ' '─')" \
        "$(printf '%*s' $col4 '' | tr ' ' '─')"

    for i in "${!cfg_labels[@]}"; do
        p50_fmt=$(fmt_ms "${cfg_p50[$i]}")
        p95_fmt=$(fmt_ms "${cfg_p95[$i]}")
        p99_fmt=$(fmt_ms "${cfg_p99[$i]}")
        load_fmt=$(fmt_ms "${cfg_load[$i]}")

        printf '│ %-*s│ %-*s│ %-*s│ %-*s│ %-*s│\n' \
            $((col0-1)) "${cfg_labels[$i]}" \
            $((col1-1)) "$p50_fmt" \
            $((col2-1)) "$p95_fmt" \
            $((col3-1)) "$p99_fmt" \
            $((col4-1)) "$load_fmt"
    done

    # Bottom border
    printf '└%s┴%s┴%s┴%s┴%s┘\n' \
        "$(printf '%*s' $col0 '' | tr ' ' '─')" \
        "$(printf '%*s' $col1 '' | tr ' ' '─')" \
        "$(printf '%*s' $col2 '' | tr ' ' '─')" \
        "$(printf '%*s' $col3 '' | tr ' ' '─')" \
        "$(printf '%*s' $col4 '' | tr ' ' '─')"

    echo ""

    # Clear arrays for next model
    unset cfg_labels cfg_p50 cfg_p95 cfg_p99 cfg_load
done

log "Done."
