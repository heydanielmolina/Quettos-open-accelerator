#!/usr/bin/env bash
# Record the `make demo` token stream (stage 5) with asciinema and render it
# to a GIF (<5 MB) with agg, plus a PNG fallback.
#
# Arrives with the demo recording step (see docs/ROADMAP.md); until then it exits 1.
#
# Planned usage:
#   scripts/record_gif.sh [MODEL=qwen|smollm2] [OUT=docs/diagrams/demo.gif]
#
# Planned behaviour:
#   asciinema rec docs/demo.cast --command "make demo MODEL=$MODEL" --overwrite
#   agg --cols 100 --rows 24 --font-size 16 docs/demo.cast "$OUT"
#   check that "$OUT" is under 5 MB; render a PNG of the final frame as fallback.
set -euo pipefail
echo "record_gif.sh: arrives with the demo recording step (see docs/ROADMAP.md)" >&2
exit 1
