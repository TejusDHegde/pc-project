"""
Step 10 - Compare, visualise and summarise.

Reads results/*.csv (from the sweep and the model training) and writes
  results/figures/*.png   - the charts
  results/RESULTS.md      - tables of the measured numbers
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK2, "ytick.color": INK2,
    "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False,
})


def size_label(n):
    for unit, v in (("M", 2 ** 20), ("K", 2 ** 10)):
        if n >= v:
            x = n / v
            return f"{x:g}{unit}" if x == int(x) else f"{x:.1f}{unit}"
    return str(n)


def log2_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Vector size N (elements)")


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name))
    plt.close(fig)
    print("  wrote", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    R = args.results
    out = os.path.join(R, "figures")
    os.makedirs(out, exist_ok=True)

    dev = pd.read_csv(os.path.join(R, "device_info.csv")).iloc[0]
    s = pd.read_csv(os.path.join(R, "summary.csv")).sort_values("n")
    g = pd.read_csv(os.path.join(R, "gpu_configs.csv"))
    cores = pd.read_csv(os.path.join(R, "cpu_cores.csv"))
    s["gpu_total_pinned_ms"] = s.h2d_pinned_ms + s.best_kernel_ms + s.d2h_pinned_ms
    s["gpu_total_pageable_ms"] = s.h2d_pageable_ms + s.best_kernel_ms + s.d2h_pageable_ms
    s["waiting_ms"] = s.h2d_pinned_ms + s.d2h_pinned_ms
    s["speedup_kernel"] = s.cpu_seq_ms / s.best_kernel_ms
    s["speedup_e2e"] = s.cpu_seq_ms / s.gpu_total_pinned_ms
    s["speedup_omp"] = s.cpu_seq_ms / s.cpu_omp_ms
    best_rows = g.loc[g.groupby("n")["kernel_ms"].idxmin()].set_index("n")
    s["throughput_gelem_s"] = s.n.map(best_rows["throughput_gelem_s"])
    s["bw_util_pct"] = s.n.map(best_rows["bw_util_pct"])
    s["occupancy_pct"] = s.n.map(best_rows["occupancy_pct"])

    # 1 - execution time
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    series = [("CPU sequential", s.cpu_seq_ms), (f"CPU OpenMP (best of 1..{int(dev.cpu_threads)} threads)", s.cpu_omp_ms),
              ("GPU kernel only (best config)", s.best_kernel_ms),
              ("GPU end-to-end, pinned memory", s.gpu_total_pinned_ms),
              ("GPU end-to-end, pageable memory", s.gpu_total_pageable_ms)]
    for i, (lab, y) in enumerate(series):
        ax.plot(s.n, y, marker="o", color=C[i], label=lab)
    log2_axis(ax)
    ax.set_yscale("log")
    ax.set_ylabel("Time (ms, log scale)")
    ax.set_title("Execution time vs vector size")
    ax.legend(fontsize=8)
    save(fig, out, "fig1_execution_time.png")

    # 2 - speedup
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, (lab, y) in enumerate([("GPU kernel only", s.speedup_kernel), ("GPU end-to-end (incl. transfers)", s.speedup_e2e),
                                  ("CPU OpenMP", s.speedup_omp)]):
        ax.plot(s.n, y, marker="o", color=C[[2, 3, 1][i]], label=lab)
    ax.axhline(1, color=INK2, lw=1, ls="--")
    ax.text(s.n.iloc[0], 1.08, "CPU sequential = 1x", color=INK2, fontsize=8)
    log2_axis(ax)
    ax.set_yscale("log")
    ax.set_ylabel("Speedup over CPU sequential (log scale)")
    ax.set_title("Speedup over the sequential baseline")
    ax.legend(fontsize=8)
    save(fig, out, "fig2_speedup.png")

    # 3 - time breakdown (waiting time = transfers)
    pick = [n for n in s.n if np.log2(n) % 4 == 2 or n == s.n.max()]
    sb = s[s.n.isin(pick)]
    tot = sb.gpu_total_pinned_ms
    parts = [("Host->Device copy", sb.h2d_pinned_ms / tot * 100, C[0]), ("Kernel", sb.best_kernel_ms / tot * 100, C[2]),
             ("Device->Host copy", sb.d2h_pinned_ms / tot * 100, C[1])]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    left = np.zeros(len(sb))
    labels = [size_label(n) for n in sb.n]
    for lab, v, col in parts:
        ax.barh(labels, v, left=left, color=col, label=lab, edgecolor="white", linewidth=2, height=0.6)
        left += v.values
    ax.set_xlabel("Share of GPU end-to-end time (%)")
    ax.set_ylabel("Vector size N")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.set_title("Where the GPU time goes (pinned host memory)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
    save(fig, out, "fig3_time_breakdown.png")

    # 4 - block size effect (naive kernel)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ns = [n for n in sorted(g.n.unique()) if np.log2(n) in (12, 18, 22, 26)] or sorted(g.n.unique())[::8]
    for i, n in enumerate(ns):
        d = g[(g.n == n) & (g.strategy == "naive")].sort_values("block_size")
        ax.plot(d.block_size, d.kernel_ms / d.kernel_ms.min(), marker="o", color=C[i], label=f"N = {size_label(n)}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 64, 128, 256, 512, 1024], ["32", "64", "128", "256", "512", "1024"])
    ax.set_xlabel("Threads per block")
    ax.set_ylabel("Kernel time / best block size")
    ax.set_title("Effect of block size (naive kernel)")
    ax.legend(fontsize=8)
    save(fig, out, "fig4_block_size.png")

    # 5 - strategies
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    best_all = g.groupby("n").kernel_ms.min()
    for i, st in enumerate(["naive", "gridstride", "float4"]):
        b = g[g.strategy == st].groupby("n").kernel_ms.min()
        ax.plot(b.index, b / best_all.loc[b.index], marker="o", color=C[i], label=st)
    log2_axis(ax)
    ax.set_ylabel("Best time of strategy / best overall")
    ax.set_title("Kernel strategies compared (1.0 = fastest)")
    ax.legend(fontsize=8)
    save(fig, out, "fig5_strategies.png")

    # 6 - utilisation
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(s.n, s.bw_util_pct, marker="o", color=C[0], label="Memory-bandwidth utilisation")
    ax.plot(s.n, s.occupancy_pct, marker="s", color=C[1], label="Theoretical SM occupancy")
    log2_axis(ax)
    ax.set_ylabel("% (best configuration)")
    ax.set_ylim(0, 110)
    ax.set_title(f"GPU utilisation (peak = {dev.measured_peak_bw_gbs:.0f} GB/s measured)")
    ax.legend(fontsize=8)
    save(fig, out, "fig6_utilisation.png")

    # 7 - CPU cores
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cn = [n for n in sorted(cores.n.unique()) if np.log2(n) in (14, 20, 26)] or sorted(cores.n.unique())[-1:]
    for i, n in enumerate(cn):
        d = cores[cores.n == n]
        ax.plot(d.threads, d.speedup, marker="o", color=C[i], label=f"N = {size_label(n)}")
    t = sorted(cores.threads.unique())
    ax.plot(t, t, color=INK2, lw=1, ls="--", label="ideal (linear)")
    ax.set_xticks(t)
    ax.set_xlabel("CPU threads (cores)")
    ax.set_ylabel("Speedup over sequential")
    ax.set_title("CPU scaling with number of cores (OpenMP)")
    ax.legend(fontsize=8)
    save(fig, out, "fig7_cpu_cores.png")

    # 8/9 - AI model
    mc_path = os.path.join(R, "model_comparison.csv")
    have_ai = os.path.exists(mc_path)
    if have_ai:
        mc = pd.read_csv(mc_path)
        best_model = mc.iloc[0].model
        cv = pd.read_csv(os.path.join(R, "cv_predictions.csv"))
        cv = cv[cv.model == best_model]
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        for i, st in enumerate(["naive", "gridstride", "float4"]):
            d = cv[cv.strategy == st]
            ax.scatter(d.actual_ms, d.predicted_ms, s=12, color=C[i], alpha=0.7, label=st, edgecolor="white", linewidth=0.4)
        lim = [cv[["actual_ms", "predicted_ms"]].min().min() * 0.8, cv[["actual_ms", "predicted_ms"]].max().max() * 1.2]
        ax.plot(lim, lim, color=INK2, lw=1, ls="--", label="perfect prediction")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Measured kernel time (ms)")
        ax.set_ylabel("Predicted kernel time (ms)")
        ax.set_title(f"{best_model}: predicted vs measured\n(cross-validated, unseen sizes)")
        ax.legend(fontsize=8)
        save(fig, out, "fig8_prediction_accuracy.png")

        av = pd.read_csv(os.path.join(R, "ai_vs_default.csv")).sort_values("n")
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        ax.plot(av.n, av.default_kernel_ms / av.oracle_kernel_ms, marker="o", color=C[1], label="Default (naive, block 256)")
        ax.plot(av.n, av.ai_kernel_ms / av.oracle_kernel_ms, marker="o", color=C[0], label="AI-recommended")
        ax.axhline(1, color=INK2, lw=1, ls="--")
        log2_axis(ax)
        ax.set_ylabel("Kernel time / best possible")
        ax.set_title("AI-selected configuration vs default (1.0 = optimal)")
        ax.legend(fontsize=8)
        save(fig, out, "fig9_ai_vs_default.png")

    # ---------------- RESULTS.md ----------------
    L = [f"# Measured results\n",
         f"GPU: **{dev.gpu}** (compute capability {dev.compute_capability}, {int(dev.num_sms)} SMs, "
         f"{dev.global_mem_gb:.1f} GB) | theoretical bandwidth {dev.theoretical_bw_gbs:.0f} GB/s, "
         f"measured peak {dev.measured_peak_bw_gbs:.0f} GB/s | CPU threads: {int(dev.cpu_threads)}\n",
         "## Performance per vector size (best GPU configuration, pinned host memory)\n",
         "| N | CPU seq (ms) | CPU OpenMP (ms) | H2D+D2H wait (ms) | kernel (ms) | GPU total (ms) | speedup kernel | speedup end-to-end | throughput (G elem/s) | BW util (%) | best config |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s[np.log2(s.n) % 2 == 0].itertuples():
        L.append(f"| {size_label(r.n)} | {r.cpu_seq_ms:.4f} | {r.cpu_omp_ms:.4f} | {r.waiting_ms:.4f} | {r.best_kernel_ms:.4f} | "
                 f"{r.gpu_total_pinned_ms:.4f} | {r.speedup_kernel:.1f}x | {r.speedup_e2e:.2f}x | {r.throughput_gelem_s:.2f} | "
                 f"{r.bw_util_pct:.0f} | {r.best_strategy}/{r.best_block}/{r.best_grid} |")
    big = s.iloc[-1]
    L += ["", f"At N = {size_label(big.n)}: data transfers take **{100 * big.waiting_ms / big.gpu_total_pinned_ms:.0f}%** "
              f"of the GPU end-to-end time; pinned memory makes transfers "
              f"{(big.h2d_pageable_ms + big.d2h_pageable_ms) / big.waiting_ms:.1f}x faster than pageable memory.", ""]
    if have_ai:
        L += ["## AI model comparison (5-fold cross-validation over unseen vector sizes)\n",
              "| model | R² (log time) | MAPE (%) | recommendation efficiency (%) | top-1 (%) | within 5% of best (%) |",
              "|---|---|---|---|---|---|"]
        for r in mc.itertuples():
            L.append(f"| {r.model} | {r.r2_log_time:.3f} | {r.mape_pct:.1f} | {r.recommendation_efficiency_pct:.1f} | "
                     f"{r.top1_accuracy_pct:.0f} | {r.within_5pct_of_best_pct:.0f} |")
        gm = lambda x: float(np.exp(np.mean(np.log(x))))  # noqa: E731
        L += ["", f"AI-selected config vs default (naive, 256): geometric-mean kernel speed-up "
                  f"**{gm(av.default_kernel_ms / av.ai_kernel_ms):.2f}x**; max {(av.default_kernel_ms / av.ai_kernel_ms).max():.2f}x. "
                  f"AI choice reaches on average {100 * np.mean(av.oracle_kernel_ms / av.ai_kernel_ms):.1f}% of the best possible "
                  f"speed, the default {100 * np.mean(av.oracle_kernel_ms / av.default_kernel_ms):.1f}%.", ""]
    L += ["## Figures\n"] + [f"![{f}](figures/{f})" for f in sorted(os.listdir(out)) if f.endswith(".png")]
    with open(os.path.join(R, "RESULTS.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("  wrote RESULTS.md")


if __name__ == "__main__":
    main()
