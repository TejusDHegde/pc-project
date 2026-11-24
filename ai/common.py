"""Shared helpers for the AI part: feature engineering and candidate configs."""
import numpy as np
import pandas as pd

STRATEGIES = ["naive", "gridstride", "float4"]
BLOCKS = [32, 64, 128, 256, 512, 1024]
SM_MULTS = [1, 2, 4, 8, 16, 32]
DEFAULT_CONFIG = ("naive", 256)  # the "textbook" configuration we compare against

FEATURES = [
    "log2_n",
    "is_naive",
    "is_gridstride",
    "is_float4",
    "log2_block",
    "log2_grid",
    "log2_elems_per_thread",
]


def ideal_ms(n, peak_bw_gbs):
    """Time the kernel would take at the measured peak bandwidth (read a, b; write c)."""
    return 12.0 * np.asarray(n, dtype=float) / (peak_bw_gbs * 1e6)


def grid_for(strategy, n, block, num_sms, sm_mult=8):
    """Grid size used by the CUDA program for a given strategy (mirrors default_grid)."""
    if strategy == "naive":
        return (n + block - 1) // block
    if strategy == "float4":
        return max(1, (n // 4 + block - 1) // block)
    return num_sms * sm_mult


def make_features(df):
    """Turn (n, strategy, block_size, grid_size) rows into model features."""
    n = df["n"].astype(float)
    block = df["block_size"].astype(float)
    grid = df["grid_size"].astype(float)
    X = pd.DataFrame(
        {
            "log2_n": np.log2(n),
            "is_naive": (df["strategy"] == "naive").astype(int),
            "is_gridstride": (df["strategy"] == "gridstride").astype(int),
            "is_float4": (df["strategy"] == "float4").astype(int),
            "log2_block": np.log2(block),
            "log2_grid": np.log2(grid),
            "log2_elems_per_thread": np.log2(n / (grid * block)),
        }
    )
    return X[FEATURES]


def candidate_configs(n, num_sms):
    """Every configuration the sweep measures, for an arbitrary vector size n."""
    rows = []
    for b in BLOCKS:
        rows.append(("naive", b, grid_for("naive", n, b, num_sms)))
        rows.append(("float4", b, grid_for("float4", n, b, num_sms)))
        for m in SM_MULTS:
            rows.append(("gridstride", b, num_sms * m))
    df = pd.DataFrame(rows, columns=["strategy", "block_size", "grid_size"])
    df.insert(0, "n", n)
    return df
