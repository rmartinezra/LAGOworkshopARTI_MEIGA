#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D scatter from CORSIKA/ARTI .sec:
- X = x' [cm], Y = y' [cm], Z = t [arb. units]
- Colors: muons (IDs 5,6), electrons (IDs 2,3), gammas (ID 1)

Expected 12-column format (comments with '#'):
# CorsikaId px py pz x' y' t shower_id prm_id prm_energy prm_theta prm_phi

python3 EAS3d.py -i 999900.sec -o arti_meiga_corsika_plots --max-points 600000
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

COLS = ["CorsikaId","px","py","pz","xprime","yprime","t",
        "shower_id","prm_id","prm_energy","prm_theta","prm_phi"]

def load_sec(path):
    df = pd.read_csv(
        path,
        comment="#",
        delim_whitespace=True,
        header=None,
        names=COLS,
        engine="python"
    )
    # Ensure numeric types where needed
    for c in ["CorsikaId","xprime","yprime","t"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["CorsikaId","xprime","yprime","t"])
    df["CorsikaId"] = df["CorsikaId"].astype(int)
    return df

def main():
    ap = argparse.ArgumentParser(description="3D scatter x', y', t colored by species groups.")
    ap.add_argument("-i","--input", required=True, help="Path to .sec file")
    ap.add_argument("-o","--outdir", default="plots_3d", help="Output directory")
    ap.add_argument("--max-points", type=int, default=60000, help="Subsample cap for plotting")
    ap.add_argument("--dpi", type=int, default=220, help="Figure DPI")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_sec(args.input)
    if df.empty:
        raise SystemExit("No valid rows found (check input file and columns).")

    # Subsample for clarity/performance
    N = len(df)
    sample_n = min(args.max_points, N)
    sample = df.sample(sample_n, random_state=42) if N > sample_n else df

    # Species groups
    mu_mask = sample["CorsikaId"].isin([5, 6])   # muons μ±
    e_mask  = sample["CorsikaId"].isin([2, 3])   # electrons e±
    g_mask  = sample["CorsikaId"] == 1           # gammas γ

    # 3D scatter: X = x', Y = y', Z = t
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(sample.loc[mu_mask, "xprime"],
               sample.loc[mu_mask, "yprime"],
               sample.loc[mu_mask, "t"],
               s=2, alpha=0.35, label="muons (μ±)")

    ax.scatter(sample.loc[e_mask, "xprime"],
               sample.loc[e_mask, "yprime"],
               sample.loc[e_mask, "t"],
               s=2, alpha=0.35, label="electrons (e±)")

    ax.scatter(sample.loc[g_mask, "xprime"],
               sample.loc[g_mask, "yprime"],
               sample.loc[g_mask, "t"],
               s=3, alpha=0.5, label="gammas (γ)")

    ax.set_xlabel("x' [cm]")
    ax.set_ylabel("y' [cm]")
    ax.set_zlabel("t [arb. units]")
    ax.set_title("3D Scatter: x', y', t (colored by species groups)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

    # Axis limits (tight to observed range)
    ax.set_xlim(float(sample["xprime"].min()), float(sample["xprime"].max()))
    ax.set_ylim(float(sample["yprime"].min()), float(sample["yprime"].max()))
    ax.set_zlim(float(sample["t"].min()), float(sample["t"].max()))

    outpath = os.path.join(args.outdir, "scatter3d_species_groups.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=args.dpi)
    plt.close()
    print("Saved:", outpath)

if __name__ == "__main__":
    main()
