#!/usr/bin/env bash
# Package the code-submission zip in the same layout as DACON's baseline_submit.zip:
#   script.py, requirements.txt, model/vessel_matcher.py, model/bundle.pkl
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f model/bundle.pkl ] || { echo "model/bundle.pkl missing: run python train.py first" >&2; exit 1; }
rm -f submission.zip
zip -q submission.zip script.py requirements.txt model/vessel_matcher.py model/bundle.pkl
unzip -l submission.zip
