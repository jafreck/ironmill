#!/bin/bash
# Capture Metal System Trace for MLX or ironmill decode.
#
# Usage:
#   ./scripts/capture_metal_trace.sh mlx       # Capture MLX decode
#   ./scripts/capture_metal_trace.sh ironmill   # Capture ironmill decode
#
# Produces .trace files that can be opened in Instruments or analyzed
# with xctrace export.

set -euo pipefail

MODE="${1:-mlx}"
DURATION="${2:-10}"  # seconds to capture
OUTDIR="/tmp/metal-traces"
mkdir -p "$OUTDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "═══════════════════════════════════════════════════════"
echo " Metal System Trace — $MODE"
echo " Duration: ${DURATION}s"
echo "═══════════════════════════════════════════════════════"

case "$MODE" in
  mlx)
    TRACE_FILE="$OUTDIR/mlx_decode_${TIMESTAMP}.trace"
    echo "Starting MLX decode benchmark..."
    echo "Will capture trace for ${DURATION}s"
    echo ""

    # Start the benchmark in background
    python3 scripts/mlx_decode_bench.py --tokens 100 &
    BENCH_PID=$!

    # Give it a moment to load and start generating
    sleep 3

    echo "Capturing Metal System Trace (PID: $BENCH_PID)..."
    xctrace record \
      --template "Metal System Trace" \
      --attach "$BENCH_PID" \
      --time-limit "${DURATION}s" \
      --output "$TRACE_FILE" 2>&1 || true

    # Wait for benchmark to finish
    wait $BENCH_PID 2>/dev/null || true

    echo ""
    echo "Trace saved to: $TRACE_FILE"
    ;;

  ironmill)
    TRACE_FILE="$OUTDIR/ironmill_decode_${TIMESTAMP}.trace"
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_ROOT="$(dirname "$SCRIPT_DIR")"

    echo "Building ironmill-bench (release)..."
    cargo build --release -p ironmill-bench --features metal 2>/dev/null

    echo "Starting ironmill decode benchmark..."
    cargo run --release -p ironmill-bench --features metal -- \
      --config configs/qwen35-4b-decode-perf.toml \
      --suite decode &
    BENCH_PID=$!

    # Give it time to load model
    sleep 35

    echo "Capturing Metal System Trace (PID: $BENCH_PID)..."
    xctrace record \
      --template "Metal System Trace" \
      --attach "$BENCH_PID" \
      --time-limit "${DURATION}s" \
      --output "$TRACE_FILE" 2>&1 || true

    wait $BENCH_PID 2>/dev/null || true

    echo ""
    echo "Trace saved to: $TRACE_FILE"
    ;;

  analyze)
    # Analyze existing trace files
    TRACE_FILE="${2:?Usage: $0 analyze <trace_file>}"
    echo "Analyzing: $TRACE_FILE"
    echo ""

    # Export trace data as XML for analysis
    xctrace export --input "$TRACE_FILE" --xpath '/trace-toc' 2>&1 | head -50
    echo ""
    echo "For detailed analysis, open in Instruments.app:"
    echo "  open $TRACE_FILE"
    ;;

  *)
    echo "Usage: $0 {mlx|ironmill|analyze} [duration_seconds|trace_file]"
    exit 1
    ;;
esac

echo ""
echo "To analyze, open in Instruments.app:"
echo "  open $TRACE_FILE"
echo ""
echo "Or export with xctrace:"
echo "  xctrace export --input $TRACE_FILE --xpath '/trace-toc'"
