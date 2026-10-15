"""
02_plot_rate_distortion_curves_fig4.py: Rate-Distortion & Quality Metrics (Figure 4)
=====================================================================================
Replicates Figure 4 from ICMLCN 2024:
'Figure 4: Comparison of the introduced DeepJSCC-WZ method with the baselines
on the KITTIStereo and Cityscape datasets for BRs rho in {1/16; 1/32}.'

Plots all 6 subplots comparing PSNR, MS-SSIM, and LPIPS across all channel SNRs:
- Datasets: Cityscape and KITTIStereo
- Bandwidth ratios: rho = 1/16 (solid lines) and rho = 1/32 (dashed lines)
- Evaluated schemes:
  * DeepJSCC-Cond (Upper bound, green circle)
  * DeepJSCC-WZ (Proposed Wyner-Ziv, orange square)
  * DeepJSCC-WZ-sm (Proposed parameter-shared Wyner-Ziv, blue triangle)
  * DeepJSCC (Point-to-point without side info, pink diamond)
  * DeepNIC [10] + Capacity (Separation-based baseline, light green pentagon)
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# DeepNIC [10] + Capacity data points from paper/Mital et al.
DEEPNIC_DATA = {
    "Cityscape": {
        16: {
            "snr": np.array([0.14, 0.42, 0.88, 1.15, 1.80, 3.65, 4.80]),
            "psnr": np.array([27.40, 28.93, 29.35, 29.80, 30.15, 29.85, 30.00]),
            "msssim": np.array([0.9630, 0.9755, 0.9795, 0.9822, 0.9828, 0.9830, 0.9835]),
            "lpips": np.array([0.0695, 0.0682, 0.0674, 0.0667, 0.0660, 0.0658, 0.0653]),
        },
        32: {
            "snr": np.array([0.14, 1.05, 2.05, 3.80, 4.80]),
            "psnr": np.array([27.20, 28.93, 29.35, 29.80, 29.95]),
            "msssim": np.array([0.9630, 0.9755, 0.9795, 0.9822, 0.9828]),
            "lpips": np.array([0.0695, 0.0682, 0.0674, 0.0667, 0.0660]),
        },
    },
    "KittiStereo": {
        16: {
            "snr": np.array([0.15, 0.40, 0.70, 1.30, 2.30, 3.35, 4.75]),
            "psnr": np.array([20.00, 21.30, 22.30, 22.90, 23.80, 24.20, 24.90]),
            "msssim": np.array([0.8980, 0.8980, 0.9320, 0.9370, 0.9460, 0.9500, 0.9580]),
            "lpips": np.array([0.2700, 0.2680, 0.2550, 0.2480, 0.2380, 0.2300, 0.2200]),
        },
        32: {
            "snr": np.array([1.05, 1.95, 3.10, 4.45, 4.95]),
            "psnr": np.array([21.30, 22.30, 22.90, 23.80, 24.00]),
            "msssim": np.array([0.8980, 0.9320, 0.9370, 0.9460, 0.9480]),
            "lpips": np.array([0.2680, 0.2550, 0.2480, 0.2380, 0.2350]),
        },
    },
}

MODEL_CONFIGS = {
    "DeepJSCC-Cond": {
        "file_key": "DeepJSCCWZJoint2",
        "color": "#66c2a5",
        "marker": "o",
        "label": "DeepJSCC-Cond",
    },
    "DeepJSCC-WZ": {
        "file_key": "DeepJSCCWZBaseline2",
        "color": "#fc8d62",
        "marker": "s",
        "label": "DeepJSCC-WZ",
    },
    "DeepJSCC-WZ-sm": {
        "file_key": "DeepJSCCWZ",
        "color": "#8da0cb",
        "marker": "^",
        "label": "DeepJSCC-WZ-sm",
    },
    "DeepJSCC": {
        "file_key": "DeepJSCCWZBaseline",
        "color": "#e78ac3",
        "marker": "d",
        "label": "DeepJSCC",
    },
}


def load_results_data(results_dir: str = "results"):
    data = {}
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")

    for f in csv_files:
        df = pd.read_csv(f)
        dataset = df["data/name"].iloc[0]
        model_name = df["model/_target_"].iloc[0]
        bw_factor = int(df["model/bw_factor"].iloc[0])

        if dataset not in data:
            data[dataset] = {}
        if bw_factor not in data[dataset]:
            data[dataset][bw_factor] = {}

        data[dataset][bw_factor][model_name] = df.sort_values("model/snr")

    return data


def plot_rate_distortion_curves_fig4(results_dir: str = "results", output_dir: str = "paper_evaluation_results"):
    os.makedirs(output_dir, exist_ok=True)
    data = load_results_data(results_dir)

    print("=" * 85)
    print("FIGURE 4: Plotting Performance Metric Curves across SNRs and Bandwidth Ratios")
    print("=" * 85)

    # IEEE style formatting
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 13,
        "axes.grid": True,
        "grid.color": "#d3d3d3",
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
    })

    fig, axs = plt.subplots(2, 3, figsize=(15, 9.5))
    plt.subplots_adjust(hspace=0.45, wspace=0.25, top=0.92, bottom=0.10)

    datasets = ["Cityscape", "KittiStereo"]
    metrics = [
        ("psnr", "PSNR (dB) \u2191", "test/psnr", "test/psnr_std"),
        ("msssim", "MS-SSIM \u2191", "test/msssim", "test/msssim_std"),
        ("lpips", "LPIPS \u2193", "test/lpips", "test/lpips_std"),
    ]

    for row_idx, dataset in enumerate(datasets):
        for col_idx, (metric_key, metric_label, col_val, col_std) in enumerate(metrics):
            ax = axs[row_idx, col_idx]

            # 1. Plot our 4 DeepJSCC models for rho = 1/16 (solid) and rho = 1/32 (dashed)
            for model_name, cfg in MODEL_CONFIGS.items():
                file_key = cfg["file_key"]
                color = cfg["color"]
                marker = cfg["marker"]

                # rho = 1/16 (solid)
                if 16 in data.get(dataset, {}) and file_key in data[dataset][16]:
                    df_16 = data[dataset][16][file_key]
                    x_16 = df_16["model/snr"].values
                    y_16 = df_16[col_val].values
                    err_16 = df_16[col_std].values
                    ax.errorbar(
                        x_16, y_16, yerr=err_16,
                        color=color, linestyle="-", marker=marker,
                        capsize=2, capthick=0.8, elinewidth=0.8,
                        alpha=0.95
                    )

                # rho = 1/32 (dashed)
                if 32 in data.get(dataset, {}) and file_key in data[dataset][32]:
                    df_32 = data[dataset][32][file_key]
                    x_32 = df_32["model/snr"].values
                    y_32 = df_32[col_val].values
                    err_32 = df_32[col_std].values
                    ax.errorbar(
                        x_32, y_32, yerr=err_32,
                        color=color, linestyle="--", marker=marker,
                        capsize=2, capthick=0.8, elinewidth=0.8,
                        alpha=0.95
                    )

            # 2. Plot DeepNIC [10] + Capacity baseline
            if dataset in DEEPNIC_DATA:
                dn_16 = DEEPNIC_DATA[dataset][16]
                if metric_key in dn_16:
                    ax.plot(
                        dn_16["snr"], dn_16[metric_key],
                        color="#a6d854", linestyle="-", marker="p",
                        label="DeepNIC + Capacity (\u03c1=1/16)",
                        alpha=0.95
                    )

                dn_32 = DEEPNIC_DATA[dataset][32]
                if metric_key in dn_32:
                    ax.plot(
                        dn_32["snr"], dn_32[metric_key],
                        color="#a6d854", linestyle="--", marker="p",
                        label="DeepNIC + Capacity (\u03c1=1/32)",
                        alpha=0.95
                    )

            # Axis configurations
            ax.set_title(f"{dataset} - \u03c1 \u2208 {{1/16; 1/32}}", pad=8)
            ax.set_xlabel("SNR$_{\\mathrm{test}}$ (dB)")
            ax.set_ylabel(metric_label)
            ax.set_xlim(-5.2, 5.2)
            ax.set_xticks([-4, -2, 0, 2, 4])

            # Limits tailored to match paper exactly
            if dataset == "Cityscape":
                if metric_key == "psnr":
                    ax.set_ylim(27.2, 38.0)
                    ax.set_yticks([28, 30, 32, 34, 36])
                elif metric_key == "msssim":
                    ax.set_ylim(0.898, 1.002)
                    ax.set_yticks([0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
                elif metric_key == "lpips":
                    ax.set_ylim(-0.01, 0.31)
                    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
            elif dataset == "KittiStereo":
                if metric_key == "psnr":
                    ax.set_ylim(19.8, 27.2)
                    ax.set_yticks([20, 22, 24, 26])
                elif metric_key == "msssim":
                    ax.set_ylim(0.868, 0.972)
                    ax.set_yticks([0.88, 0.90, 0.92, 0.94, 0.96])
                elif metric_key == "lpips":
                    ax.set_ylim(0.14, 0.41)
                    ax.set_yticks([0.15, 0.20, 0.25, 0.30, 0.35, 0.40])

    # Central Legend Box
    legend_elements = [
        Line2D([0], [0], color=MODEL_CONFIGS["DeepJSCC-Cond"]["color"], lw=1.6, marker='o', label='DeepJSCC-Cond'),
        Line2D([0], [0], color=MODEL_CONFIGS["DeepJSCC-WZ"]["color"], lw=1.6, marker='s', label='DeepJSCC-WZ'),
        Line2D([0], [0], color=MODEL_CONFIGS["DeepJSCC-WZ-sm"]["color"], lw=1.6, marker='^', label='DeepJSCC-WZ-sm'),
        Line2D([0], [0], color=MODEL_CONFIGS["DeepJSCC"]["color"], lw=1.6, marker='d', label='DeepJSCC'),
        Line2D([0], [0], color='#a6d854', lw=1.6, marker='p', label='DeepNIC [10] + Capacity'),
        Line2D([0], [0], color='black', lw=1.6, linestyle='-', label='\u03c1 = 1/16'),
        Line2D([0], [0], color='black', lw=1.6, linestyle='--', label='\u03c1 = 1/32'),
    ]

    fig.legend(
        handles=legend_elements,
        loc="center",
        bbox_to_anchor=(0.5, 0.51),
        ncol=7,
        frameon=True,
        edgecolor="#333333",
        facecolor="white",
        framealpha=1.0,
        fontsize=10.5,
    )

    out_png = os.path.join(output_dir, "fig4_performance_curves.png")
    out_pdf = os.path.join(output_dir, "fig4_performance_curves.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print(f"[Saved Figure 4 PNG (300 DPI)] -> {out_png}")
    print(f"[Saved Figure 4 PDF (Vector)]  -> {out_pdf}")
    return out_png


if __name__ == "__main__":
    plot_rate_distortion_curves_fig4()
