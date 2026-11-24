# Case Study 15: CUDA Vector Addition, Kernel Optimization and AI Configuration

This project does vector addition `C = A + B` in four ways:

- sequentially on the CPU,
- with OpenMP on several CPU cores,
- with three CUDA kernel strategies across many thread, block and grid configurations,
- with a configuration chosen by a machine-learning model that predicts the fastest CUDA setup for a given vector size.

The full write-up, covering the 10 workflow steps and the prompts used, is in **[REPORT.md](REPORT.md)**.
Once you have run the project, the measured numbers are in `results/RESULTS.md` and the charts are in `results/figures/`.

## Quick start (no GPU needed on your own computer)

1. Open the notebook in Google Colab:
   **https://colab.research.google.com/github/TejusDHegde/pc-project/blob/claude/cuda-vector-addition-ai-4t14v6/CUDA_Vector_Add_AI.ipynb**
   If that link doesn't open, go to colab.research.google.com, choose **File → Upload notebook** and upload `CUDA_Vector_Add_AI.ipynb` from this repository.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Choose **Runtime → Run all**. It takes about 5 minutes. The last cell downloads `results.zip`.

## On any Linux machine with an NVIDIA GPU and the CUDA toolkit

```bash
pip install -r requirements.txt
./run_all.sh            # full run;  ./run_all.sh quick  for a shorter one
```

## Commands

| Command | What it does |
|---|---|
| `make` | builds `bin/vector_add` |
| `./bin/vector_add info` | shows GPU properties and the peak memory bandwidth |
| `./bin/vector_add run --n 16777216 --strategy naive --block 256 [--grid G] [--pageable]` | runs one configuration and prints every step: allocation, H2D copy, kernel, D2H copy, verification, speedup, waiting time and utilisation |
| `./bin/vector_add sweep --out results` | collects the dataset: 33 sizes × 48 GPU configurations, plus CPU runs on 1…all cores |
| `python3 ai/train_model.py` | trains and compares 4 ML models, saves the best one and the recommendation table |
| `python3 analysis/plots.py` | writes the charts and `results/RESULTS.md` |
| `python3 ai/predict.py --n 10000000 --run` | the AI picks a configuration, then runs it and the default configuration on the GPU for comparison |
| `./bin/vector_add auto --n 5000000` | the CUDA program reads the AI's recommendation table and runs that configuration |

## Layout

```
src/vector_add.cu          CPU baseline, OpenMP, 3 CUDA kernels, timing, sweep and auto modes
ai/common.py               features and candidate configurations
ai/train_model.py          model training, cross-validation and the recommendation table
ai/predict.py              AI-driven configuration, run on the GPU
analysis/plots.py          charts and results tables
CUDA_Vector_Add_AI.ipynb   one-click Colab notebook
run_all.sh, Makefile       local pipeline
REPORT.md                  project report (objectives, prompts, method, metrics, conclusions)
```
