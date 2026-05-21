#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
python paper_revision/scripts/create_label_noise_splits.py
echo "Created label-noise splits. Training requires torch/GPU; run corresponding configs in paper_revision/configs/label_noise_*_config.py."
python paper_revision/scripts/build_label_noise_table.py || true
