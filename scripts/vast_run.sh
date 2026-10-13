#!/usr/bin/env bash
# Rent a many-core vast.ai box, run the 5-fold CV and the final training there, pull results back.
# Run this on YOUR machine (needs: pip install vastai, an SSH key registered at vast.ai, git access to the repo).
#
#   export VAST_API_KEY=...            # never commit this
#   scripts/vast_run.sh search         # list cheap high-core offers
#   scripts/vast_run.sh up <OFFER_ID>  # create instance, push data, start training
#   scripts/vast_run.sh pull <INSTANCE_ID>   # fetch logs + model/bundle.pkl
#   scripts/vast_run.sh down <INSTANCE_ID>   # destroy (stops billing)
set -euo pipefail
cd "$(dirname "$0")/.."
BRANCH=${BRANCH:-claude/ecstatic-archimedes-g2uglk}
REPO=${REPO:-https://github.com/Topabaem05/gamboa.git}
IMAGE=${IMAGE:-python:3.11-slim}
DISK=${DISK:-20}

case "${1:-}" in
  search)
    # CPU-bound workload: many effective cores + RAM matter, the GPU does not.
    vastai search offers 'cpu_cores_effective>=32 cpu_ram>=64 disk_space>=20 reliability>0.95 inet_down>200 rentable=true' \
      -o 'dph' | head -25
    ;;
  up)
    OFFER=${2:?offer id}
    ONSTART=$(mktemp)
    cat > "$ONSTART" <<ONS
#!/bin/bash
set -x
cd /workspace
apt-get update -qq && apt-get install -y -qq git zip >/dev/null
git clone -b $BRANCH $REPO gamboa && cd gamboa
pip install -q -r requirements.txt
mkdir -p data output
# wait until data has been copied in (see 'vastai copy' below)
while [ ! -f data/train.csv ] || [ ! -f data/vessel_catalog.csv ]; do sleep 10; done
NJ=\$(nproc)
( for k in 0 1 2 3 4; do python -W ignore evaluate_local.py --folds 5 --only-fold \$k --skip-replay > cv_fold\$k.log 2>&1 & done; wait ) &
python -W ignore train.py > train.log 2>&1
scripts/make_submission.sh > package.log 2>&1
wait
touch DONE
ONS
    vastai create instance "$OFFER" --image "$IMAGE" --disk "$DISK" --ssh --direct --onstart "$ONSTART" --label gamboa-train
    echo "Now copy the data in:  vastai copy data/ <INSTANCE_ID>:/workspace/gamboa/data/"
    ;;
  pull)
    ID=${2:?instance id}
    mkdir -p vast_results
    vastai copy "$ID:/workspace/gamboa/cv_fold0.log" vast_results/ || true
    for k in 1 2 3 4; do vastai copy "$ID:/workspace/gamboa/cv_fold$k.log" vast_results/ || true; done
    vastai copy "$ID:/workspace/gamboa/train.log" vast_results/ || true
    vastai copy "$ID:/workspace/gamboa/model/bundle.pkl" model/bundle.pkl
    vastai copy "$ID:/workspace/gamboa/submission.zip" submission.zip
    grep -h "cv fold\|replay" vast_results/*.log || true
    ;;
  down)
    vastai destroy instance "${2:?instance id}"
    ;;
  *)
    echo "usage: $0 search | up <offer> | pull <instance> | down <instance>" >&2; exit 1
    ;;
esac
