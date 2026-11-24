#!/usr/bin/env bash
# Runs the whole project end to end on a machine with an NVIDIA GPU.
#   ./run_all.sh          full sweep, N = 2^10 .. 2^26   (a few minutes)
#   ./run_all.sh quick    smaller sweep, N = 2^10 .. 2^22
set -e
cd "$(dirname "$0")"
MAX_EXP=26; [ "$1" = "quick" ] && MAX_EXP=22

echo "### 1. Build";                     make
echo "### 2. Device information";        ./bin/vector_add info
echo "### 3. One run, every CUDA step";  ./bin/vector_add run --n 16777216 --strategy naive --block 256
echo "### 4. Experiment sweep";          ./bin/vector_add sweep --out results --max-exp $MAX_EXP
echo "### 5. Train the AI model";        python3 ai/train_model.py
echo "### 6. Charts and tables";         python3 analysis/plots.py
echo "### 7. AI-selected vs default";    python3 ai/predict.py --n 10000000 --run
echo "### 8. CUDA program in auto mode"; ./bin/vector_add auto --n 5000000
echo "Done. See results/RESULTS.md and results/figures/"
