"""
Step 9 - Use the AI model to configure the CUDA program.

    python3 ai/predict.py --n 10000000            # show the recommended configuration
    python3 ai/predict.py --n 10000000 --run      # ...and run it vs. the default config

The model predicts the kernel time of every candidate (strategy, block, grid),
picks the fastest, and launches bin/vector_add with that configuration.
"""
import argparse
import os
import subprocess
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEFAULT_CONFIG, candidate_configs  # noqa: E402
from train_model import recommend  # noqa: E402


def run_binary(binary, n, strategy, block, grid=0):
    cmd = [binary, "run", "--n", str(n), "--strategy", strategy, "--block", str(block)]
    if grid:
        cmd += ["--grid", str(grid)]
    print("\n$ " + " ".join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True)
    print(out.stdout, end="")
    if out.returncode != 0:
        print(out.stderr, file=sys.stderr)
        sys.exit(out.returncode)
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT,")][-1].split(",")
    keys = ["n", "strategy", "block", "grid", "cpu_ms", "h2d_ms", "kernel_ms", "d2h_ms", "total_ms", "ok"]
    return dict(zip(keys, line[1:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True, help="vector size")
    ap.add_argument("--model", default="models/kernel_time_model.joblib")
    ap.add_argument("--binary", default="bin/vector_add")
    ap.add_argument("--run", action="store_true", help="run the AI and default configs on the GPU")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model, num_sms = bundle["model"], bundle["num_sms"]
    ranked = recommend(model, candidate_configs(args.n, num_sms), bundle["peak_bw_gbs"])
    print(f"Model: {bundle['model_name']} trained on {bundle['gpu']} "
          f"(N = {bundle['n_min']:,} .. {bundle['n_max']:,})")
    if not bundle["n_min"] <= args.n <= bundle["n_max"]:
        print("Note: N is outside the training range - the prediction is an extrapolation.")
    print(f"\nTop 5 predicted configurations for N = {args.n:,}:")
    print(ranked.head(5).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best = ranked.iloc[0]
    if not args.run:
        return
    ai = run_binary(args.binary, args.n, best["strategy"], int(best["block_size"]), int(best["grid_size"]))
    de = run_binary(args.binary, args.n, *DEFAULT_CONFIG)
    k_ai, k_de = float(ai["kernel_ms"]), float(de["kernel_ms"])
    print("\n================ AI-selected vs default configuration ================")
    print(f"{'config':40s} {'predicted ms':>12s} {'measured ms':>12s}")
    ai_label = f"AI: {best['strategy']} block={int(best['block_size'])} grid={int(best['grid_size'])}"
    print(f"{ai_label:40s} {best['predicted_kernel_ms']:12.4f} {k_ai:12.4f}")
    print(f"{'default: naive block=256':40s} {'':>12s} {k_de:12.4f}")
    print(f"Kernel speed-up of AI choice over default: {k_de / k_ai:.2f}x  "
          f"(prediction error {100 * abs(best['predicted_kernel_ms'] - k_ai) / k_ai:.1f}%)")


if __name__ == "__main__":
    main()
