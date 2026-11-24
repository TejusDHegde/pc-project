"""
Step 8 - Build and train the AI model.

The model learns the kernel execution time ("burst time") of a CUDA vector-add
configuration from (vector size, strategy, block size, grid size). Internally it
predicts the slowdown relative to the ideal time at peak memory bandwidth
(log10(kernel_ms / ideal_ms)); this keeps the model focused on what the
configuration changes instead of on the obvious "bigger N takes longer". To recommend
a configuration for a new vector size we predict the time of every candidate
configuration and pick the fastest one.

Evaluation uses GroupKFold over vector sizes, so every test size is completely
unseen during training (an honest test of "predict for a new N").

Inputs : results/gpu_configs.csv, results/device_info.csv   (from `vector_add sweep`)
Outputs: models/kernel_time_model.joblib
         models/recommendation_table.csv   (read by `vector_add auto`)
         results/model_comparison.csv, results/cv_predictions.csv,
         results/ai_vs_default.csv
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEFAULT_CONFIG, FEATURES, candidate_configs, ideal_ms, make_features  # noqa: E402

MODELS = {
    "LinearRegression": lambda: LinearRegression(),
    "DecisionTree": lambda: DecisionTreeRegressor(max_depth=10, random_state=0),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1),
    "GradientBoosting": lambda: GradientBoostingRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, random_state=0),
}


def predict_ms(model, configs, peak_bw):
    return 10 ** model.predict(make_features(configs)) * ideal_ms(configs["n"], peak_bw)


def recommend(model, configs, peak_bw):
    """Predict kernel time for each candidate and return them sorted fastest-first."""
    pred = predict_ms(model, configs, peak_bw)
    out = configs.copy()
    out["predicted_kernel_ms"] = pred
    return out.sort_values("predicted_kernel_ms").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--models", default="models")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.results, "gpu_configs.csv"))
    dev = pd.read_csv(os.path.join(args.results, "device_info.csv")).iloc[0]
    num_sms, peak_bw = int(dev["num_sms"]), float(dev["measured_peak_bw_gbs"])
    df = df[df["verified"] == 1].reset_index(drop=True)
    ideal = ideal_ms(df["n"], peak_bw)
    X, y, groups = make_features(df), np.log10(df["kernel_ms"] / ideal), df["n"]
    print(f"Dataset: {len(df)} measured configurations, {df['n'].nunique()} vector sizes, GPU = {dev['gpu']}")

    # ---------- compare models with cross-validation over unseen vector sizes ----------
    gkf = GroupKFold(n_splits=args.folds)
    comparison, cv_rows, choice_rows = [], [], []
    for name, make in MODELS.items():
        pred = np.zeros(len(df))
        for tr, te in gkf.split(X, y, groups):
            m = make().fit(X.iloc[tr], y.iloc[tr])
            pred[te] = m.predict(X.iloc[te])
        pred_ms = 10 ** pred * ideal
        # Recommendation quality: for each unseen N, run the config the model ranks fastest.
        eff, top1, within5 = [], 0, 0
        for n, g in df.assign(pred_ms=pred_ms).groupby("n"):
            oracle = g["kernel_ms"].min()
            chosen = g.loc[g["pred_ms"].idxmin()]
            default = g[(g["strategy"] == DEFAULT_CONFIG[0]) & (g["block_size"] == DEFAULT_CONFIG[1])]["kernel_ms"].iloc[0]
            eff.append(oracle / chosen["kernel_ms"])
            top1 += chosen["kernel_ms"] == oracle
            within5 += chosen["kernel_ms"] <= 1.05 * oracle
            choice_rows.append({
                "model": name, "n": n, "oracle_kernel_ms": oracle, "ai_kernel_ms": chosen["kernel_ms"],
                "default_kernel_ms": default, "ai_strategy": chosen["strategy"],
                "ai_block": int(chosen["block_size"]), "ai_grid": int(chosen["grid_size"]),
            })
        comparison.append({
            "model": name,
            "r2_log_time": r2_score(np.log10(df["kernel_ms"]), np.log10(pred_ms)),
            "mape_pct": 100 * mean_absolute_percentage_error(df["kernel_ms"], pred_ms),
            "recommendation_efficiency_pct": 100 * np.mean(eff),
            "top1_accuracy_pct": 100 * top1 / df["n"].nunique(),
            "within_5pct_of_best_pct": 100 * within5 / df["n"].nunique(),
        })
        cv_rows.append(pd.DataFrame({"model": name, "n": df["n"], "strategy": df["strategy"],
                                     "actual_ms": df["kernel_ms"], "predicted_ms": pred_ms}))

    comp = pd.DataFrame(comparison).sort_values("recommendation_efficiency_pct", ascending=False)
    best_name = comp.iloc[0]["model"]
    print("\nModel comparison (5-fold CV, test vector sizes never seen in training):")
    print(comp.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
    print(f"\nSelected model: {best_name}")

    os.makedirs(args.models, exist_ok=True)
    comp.to_csv(os.path.join(args.results, "model_comparison.csv"), index=False)
    pd.concat(cv_rows).to_csv(os.path.join(args.results, "cv_predictions.csv"), index=False)
    choices = pd.DataFrame(choice_rows)
    choices[choices["model"] == best_name].drop(columns="model").to_csv(
        os.path.join(args.results, "ai_vs_default.csv"), index=False)

    # ---------- final model on all data ----------
    model = MODELS[best_name]().fit(X, y)
    joblib.dump({"model": model, "model_name": best_name, "features": FEATURES, "num_sms": num_sms,
                 "peak_bw_gbs": peak_bw, "gpu": dev["gpu"], "n_min": int(df["n"].min()), "n_max": int(df["n"].max())},
                os.path.join(args.models, "kernel_time_model.joblib"))

    if hasattr(model, "feature_importances_"):
        imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda t: -t[1])
        print("Feature importances: " + ", ".join(f"{f}={v:.3f}" for f, v in imp))

    # ---------- step 9: recommendation table used by the CUDA program ----------
    lo, hi = np.log2(df["n"].min()), np.log2(df["n"].max())
    rows = []
    for e in np.arange(lo, hi + 1e-9, 0.25):
        n = int(round(2 ** e))
        best = recommend(model, candidate_configs(n, num_sms), peak_bw).iloc[0]
        rows.append([n, best["strategy"], int(best["block_size"]), int(best["grid_size"]), best["predicted_kernel_ms"]])
    table = pd.DataFrame(rows, columns=["n", "strategy", "block_size", "grid_size", "predicted_kernel_ms"])
    table.to_csv(os.path.join(args.models, "recommendation_table.csv"), index=False)
    with open(os.path.join(args.models, "model_info.json"), "w") as f:
        json.dump({"selected_model": best_name, "gpu": dev["gpu"], "num_sms": num_sms,
                   "cv": comp.to_dict(orient="records")}, f, indent=2)
    print(f"\nSaved model and recommendation table ({len(table)} sizes) to {args.models}/")
    print(table.iloc[:: max(1, len(table) // 10)].to_string(index=False))


if __name__ == "__main__":
    main()
