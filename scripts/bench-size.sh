#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IRONMILL="$REPO_ROOT/target/release/ironmill"
FIXTURE_DIR="$REPO_ROOT/tests/fixtures"

# ---------------------------------------------------------------------------
# 1. Build ironmill in release mode
# ---------------------------------------------------------------------------
echo "==> Building ironmill (release)..."
cargo build --release -p ironmill-cli --manifest-path "$REPO_ROOT/Cargo.toml" --quiet

# ---------------------------------------------------------------------------
# 2. Ensure test fixtures exist
# ---------------------------------------------------------------------------
if [ ! -f "$FIXTURE_DIR/mnist.onnx" ]; then
    echo "==> Downloading fixtures..."
    "$REPO_ROOT/scripts/download-fixtures.sh"
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Size of a file or directory in bytes (portable).
size_bytes() {
    if [ -d "$1" ]; then
        find "$1" -type f -exec cat {} + | wc -c | tr -d ' '
    else
        wc -c < "$1" | tr -d ' '
    fi
}

# Human-readable size via du -sh.
human_size() {
    du -sh "$1" | cut -f1 | tr -d ' '
}

# Compute percentage reduction relative to the original size.
reduction_pct() {
    local orig=$1 new=$2
    if [ "$orig" -eq 0 ]; then
        echo "0"
        return
    fi
    # Use awk for floating-point math, round to nearest integer.
    awk "BEGIN { printf \"%d\", (($orig - $new) / $orig) * 100 }"
}

# Print a table row.
row() {
    printf "│ %-28s │ %-8s │ %-9s │\n" "$1" "$2" "$3"
}

# ---------------------------------------------------------------------------
# 3-4. Compile each model with every optimization combo and print table
# ---------------------------------------------------------------------------

MODELS=("mnist.onnx" "squeezenet1.1.onnx")

CONFIGS=(
    "No optimization (--no-fusion)|--no-fusion"
    "Default (always-on)|"
    "+ FP16|--quantize fp16"
    "+ INT8|--quantize int8"
    "+ Palettize 4-bit|--palettize 4"
    "+ Palettize 6-bit|--palettize 6"
    "+ FP16 + Palettize 4-bit|--quantize fp16 --palettize 4"
)

for model in "${MODELS[@]}"; do
    model_path="$FIXTURE_DIR/$model"
    if [ ! -f "$model_path" ]; then
        echo "WARNING: $model_path not found, skipping."
        continue
    fi

    orig_size=$(human_size "$model_path")
    orig_bytes=$(size_bytes "$model_path")

    # Temp directory for compiled outputs
    work_dir=$(mktemp -d "${REPO_ROOT}/bench-size-XXXXXX")

    echo ""
    echo "Model: $model (original: $orig_size)"
    echo "┌──────────────────────────────┬──────────┬───────────┐"
    printf "│ %-28s │ %-8s │ %-9s │\n" "Configuration" "Size" "Reduction"
    echo "├──────────────────────────────┼──────────┼───────────┤"

    idx=0
    for entry in "${CONFIGS[@]}"; do
        label="${entry%%|*}"
        flags="${entry##*|}"

        out_dir="$work_dir/variant-$idx"
        mkdir -p "$out_dir"

        # shellcheck disable=SC2086
        "$IRONMILL" compile "$model_path" -o "$out_dir/output.mlpackage" $flags 2>/dev/null

        pkg="$out_dir/output.mlpackage"
        if [ -d "$pkg" ]; then
            pkg_size=$(human_size "$pkg")
            pkg_bytes=$(size_bytes "$pkg")
        else
            pkg_size="N/A"
            pkg_bytes=0
        fi

        pct=$(reduction_pct "$orig_bytes" "$pkg_bytes")
        if [ "$pct" -lt 0 ] 2>/dev/null; then
            pct_str="${pct}%"
        else
            pct_str="${pct}%"
        fi

        row "$label" "$pkg_size" "$pct_str"
        idx=$((idx + 1))
    done

    echo "└──────────────────────────────┴──────────┴───────────┘"

    # 5. Clean up temp directory
    rm -rf "$work_dir"
done

echo ""
echo "Done."
