#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 FOLD SEED [--resume]" >&2; exit 2
fi
fold=$1; seed=$2; resume=${3:-}
[[ "$fold" =~ ^[123]$ ]] || { echo "fold must be 1, 2, or 3" >&2; exit 2; }
[[ "$seed" =~ ^(42|1337|2026)$ ]] || { echo "seed must be 42, 1337, or 2026" >&2; exit 2; }
[[ -z "$resume" || "$resume" == --resume ]] || { echo "third argument may only be --resume" >&2; exit 2; }
repo=$(cd "$(dirname "$0")/../.." && pwd)
run="inceptiontime_f${fold}_s${seed}_l100"
out="$repo/results/baselines/inceptiontime_four_domain/$run"
if [[ -f "$out/state.json" ]] && [[ $(jq -r '.status // ""' "$out/state.json") == COMPLETE ]]; then
  for required in best.pt test_predictions.csv test_report.json; do
    [[ -s "$out/$required" ]] || { echo "COMPLETE state is missing $required; audit required" >&2; exit 1; }
  done
  echo "$run already COMPLETE; refusing to overwrite"
  exit 0
fi
if [[ -e "$out" && -z "$resume" ]]; then
  echo "$run has an incomplete directory; inspect it and pass --resume only if scientifically valid" >&2
  exit 3
fi
if [[ "$resume" == --resume ]]; then
  [[ -s "$out/last.pt" ]] || { echo "cannot resume without last.pt" >&2; exit 3; }
  [[ ! -e "$out/test_seal.json" ]] || { echo "cannot resume after TEST seal; audit required" >&2; exit 3; }
fi
cd "$repo"
exec .venv/bin/python scripts/baselines/run_inceptiontime.py --fold "$fold" --seed "$seed" ${resume:+--resume}
