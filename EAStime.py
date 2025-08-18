#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORSIKA/ARTI Secondaries: counts vs time (t)

Reads a 12-column .sec file with commented header:
# CorsikaId px py pz x' y' t shower_id prm_id prm_energy prm_theta prm_phi

Outputs:
  - counts_vs_t.png                (linear y-axis)
  - counts_vs_t_log.png            (log y-axis)
  - counts_vs_t.csv                (t, count)
Optional:
  - counts_by_id_vs_t.png          (one curve per CorsikaId)
  - binned_counts_vs_t.png/.csv    (if --bins or --bin-width used)

USE
python counts_vs_t.py -i 999900.sec -o arti_meiga_corsika_plots --by-id

"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


COLS = [
    "CorsikaId","px","py","pz","xprime","yprime","t",
    "shower_id","prm_id","prm_energy","prm_theta","prm_phi"
]


def read_sec(path):
    # Read strictly 12 columns ignoring commented lines
    df = pd.read_csv(
        path,
        comment="#",
        delim_whitespace=True,
        header=None,
        names=COLS,
        engine="python"
    )
    # Ensure numeric
    for c in COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop rows without t
    df = df.dropna(subset=["t"])
    return df


def counts_vs_t(df):
    # Exact counts per unique t value (no binning)
    c = df.groupby("t").size().reset_index(name="count").sort_values("t")
    return c


def binned_counts_vs_t(df, bins=None, bin_width=None):
    t = df["t"].to_numpy()
    tmin, tmax = float(np.nanmin(t)), float(np.nanmax(t))

    if bin_width is not None:
        # Build uniform bins from min to max
        nbins = int(np.ceil((tmax - tmin) / bin_width)) if bin_width > 0 else 1
        edges = np.linspace(tmin, tmax, max(nbins, 1) + 1)
    elif bins is not None:
        # Use specified number of bins
        edges = np.linspace(tmin, tmax, max(int(bins), 1) + 1)
    else:
        # Fallback: 100 bins
        edges = np.linspace(tmin, tmax, 101)

    hist, edges = np.histogram(t, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return pd.DataFrame({"t_center": centers, "count": hist, "t_left": edges[:-1], "t_right": edges[1:]})


def plot_counts_vs_t(df_counts, out_path_png, logy=False, title="Particle counts vs time t (all species)"):
    plt.figure(figsize=(9, 5.5))
    x = df_counts.iloc[:, 0].values
    y = df_counts.iloc[:, 1].values
    if logy:
        plt.semilogy(x, y, marker='o', linestyle='-', markersize=2)
        plt.ylabel("Number of particles (log scale)")
    else:
        plt.plot(x, y, marker='o', linestyle='-', markersize=2)
        plt.ylabel("Number of particles")

    plt.xlabel("t [arb. units]")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)

    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmin == xmax:
        eps = 1e-9 if xmin == 0 else abs(xmin) * 1e-9
        xmin -= eps; xmax += eps
    plt.xlim(xmin, xmax)

    plt.tight_layout()
    plt.savefig(out_path_png, dpi=180)
    plt.close()


def plot_by_id(df, out_path_png):
    """One curve per CorsikaId (may be heavy if many IDs)."""
    plt.figure(figsize=(9, 5.5))
    for pid, sub in df.groupby("CorsikaId"):
        series = sub.groupby("t").size().reset_index(name="count").sort_values("t")
        if len(series) == 0:
            continue
        plt.plot(series["t"].values, series["count"].values, linewidth=1, label=str(int(pid)))
    plt.xlabel("t [arb. units]")
    plt.ylabel("Number of particles")
    plt.title("Particle counts vs time t by CorsikaId")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="CorsikaId", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Counts of CORSIKA/ARTI secondaries vs time t.")
    ap.add_argument("-i", "--input", required=True, help="Path to .sec file")
    ap.add_argument("-o", "--outdir", default="plots_counts_vs_t", help="Output directory")
    ap.add_argument("--bins", type=int, default=None, help="Number of bins for t (optional)")
    ap.add_argument("--bin-width", type=float, default=None, help="Bin width for t (optional)")
    ap.add_argument("--by-id", action="store_true", help="Also plot one curve per CorsikaId")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Read data
    df = read_sec(args.input)
    if df.empty:
        print("No data read. Check the input file.", file=sys.stderr)
        sys.exit(1)

    # Exact counts per unique t
    c_exact = counts_vs_t(df)
    c_exact_csv = os.path.join(args.outdir, "counts_vs_t.csv")
    c_exact.to_csv(c_exact_csv, index=False)

    # Plots exact counts
    plot_counts_vs_t(
        c_exact,
        os.path.join(args.outdir, "counts_vs_t.png"),
        logy=False,
        title="Particle counts vs time t (all species)"
    )
    plot_counts_vs_t(
        c_exact,
        os.path.join(args.outdir, "counts_vs_t_log.png"),
        logy=True,
        title="Particle counts vs time t (all species, log scale)"
    )

    # Optional: binned counts
    if args.bins is not None or args.bin_width is not None:
        c_binned = binned_counts_vs_t(df, bins=args.bins, bin_width=args.bin_width)
        c_binned_csv = os.path.join(args.outdir, "binned_counts_vs_t.csv")
        c_binned.to_csv(c_binned_csv, index=False)
        plot_counts_vs_t(
            c_binned[["t_center", "count"]],
            os.path.join(args.outdir, "binned_counts_vs_t.png"),
            logy=False,
            title="Binned particle counts vs time t (all species)"
        )
        plot_counts_vs_t(
            c_binned[["t_center", "count"]],
            os.path.join(args.outdir, "binned_counts_vs_t_log.png"),
            logy=True,
            title="Binned particle counts vs time t (all species, log scale)"
        )

    # Optional: one curve per particle ID
    if args.by_id:
        plot_by_id(df, os.path.join(args.outdir, "counts_by_id_vs_t.png"))

    print("Done.")
    print(f"Output dir: {args.outdir}")
    print("* counts_vs_t.png / counts_vs_t_log.png")
    if args.bins or args.bin_width:
        print("* binned_counts_vs_t.png / binned_counts_vs_t_log.png")
    if args.by_id:
        print("* counts_by_id_vs_t.png")


if __name__ == "__main__":
    main()
