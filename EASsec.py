#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORSIKA/ARTI .sec -> p^2 histograms
- Overlay by species (names, not IDs), log-log, common log bins
- Total histogram (all species combined), log-log

Input format (12 cols, commented header line):
# CorsikaId px py pz x' y' t shower_id prm_id prm_energy prm_theta prm_phi

python EASsec.py -i 999900.sec -o arti_meiga_corsika_plots --nbins 80 --min-count 1
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Map CORSIKA IDs to particle names (extend if needed) ----
ID2NAME = {
    1:  "gamma",  # γ
    2:  "e+",
    3:  "e-",
    5:  "mu+",
    6:  "mu-",
    7:  "pi0",
    8:  "pi+",
    9:  "pi-",
    13: "neutron",
    14: "proton",
}

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
    # numeric
    for c in COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["CorsikaId","px","py","pz"])
    # name
    df["name"] = df["CorsikaId"].astype(int).map(ID2NAME).fillna("other")
    # p^2 in (GeV/c)^2
    df["p2"] = df["px"]**2 + df["py"]**2 + df["pz"]**2
    # keep only positive for log x
    df = df[df["p2"] > 0]
    return df

def main():
    ap = argparse.ArgumentParser(description="p^2 histograms (overlay by species + total) from CORSIKA .sec")
    ap.add_argument("-i","--input", required=True, help="Path to .sec file")
    ap.add_argument("-o","--outdir", default="p2_hists", help="Output directory")
    ap.add_argument("--nbins", type=int, default=80, help="Number of log-spaced bins")
    ap.add_argument("--min-count", type=int, default=10, help="Skip species with fewer counts than this")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_sec(args.input)
    if df.empty:
        raise SystemExit("No valid rows after reading the file.")

    # --- Build common log-spaced bins for p2 ---
    p2 = df["p2"].to_numpy()
    p2min, p2max = float(np.min(p2)), float(np.max(p2))
    # guard
    if p2min <= 0:
        p2min = np.nextafter(0, 1)
    bins = np.logspace(np.log10(p2min), np.log10(p2max), args.nbins)

    # --- 1) Overlay by species (step plots), log-log ---
    plt.figure(figsize=(9.5,6.2))
    for name, sub in df.groupby("name"):
        if len(sub) < args.min_count:
            continue
        h, edges = np.histogram(sub["p2"].values, bins=bins)
        centers = 0.5*(edges[1:] + edges[:-1])
        # avoid all-zero series
        if np.all(h == 0):
            continue
        plt.step(centers, h, where="mid", label=f"{name} (n={len(sub)})", linewidth=1)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$p^2 \; [(\mathrm{GeV}/c)^2]$")
    plt.ylabel("Counts per bin")
    plt.title("Energy proxy by species (overlay) — $p^2$")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(title="Species", fontsize=8, ncol=2)
    out_overlay = os.path.join(args.outdir, "p2_overlay_by_species.png")
    plt.tight_layout()
    plt.savefig(out_overlay, dpi=220)
    plt.close()

    # --- 2) Total histogram (all species combined), log-log ---
    plt.figure(figsize=(9.5,6.2))
    plt.hist(p2, bins=bins, histtype="stepfilled", alpha=0.6)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$p^2 \; [(\mathrm{GeV}/c)^2]$")
    plt.ylabel("Counts per bin")
    plt.title("All species combined — $p^2$")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    out_total = os.path.join(args.outdir, "p2_total_all_species.png")
    plt.tight_layout()
    plt.savefig(out_total, dpi=220)
    plt.close()

    # Simple CSV summary of counts by species
    counts_csv = os.path.join(args.outdir, "counts_by_species.csv")
    df["name"].value_counts().sort_index().to_csv(counts_csv, header=["count"])

    print("Saved:")
    print(" -", out_overlay)
    print(" -", out_total)
    print(" -", counts_csv)

if __name__ == "__main__":
    main()
