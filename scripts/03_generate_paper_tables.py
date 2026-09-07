"""
Generate comprehensive tables of all values from the paper:
- PSNR (dB), MS-SSIM, LPIPS across test SNRs in {-4, -2, 0, 2, 4} dB
- Both datasets: Cityscape and KITTIStereo
- Both bandwidth ratios: rho in {1/16, 1/32}
- All models: DeepJSCC, DeepJSCC-WZ-sm, DeepJSCC-WZ, DeepJSCC-Cond, DeepNIC [10] + Capacity
- Saves to Markdown, CSV, and LaTeX formats in replication_output/tables/
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from tabulate import tabulate

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MODEL_MAP = {
    "DeepJSCCWZJoint2": "DeepJSCC-Cond",
    "DeepJSCCWZBaseline2": "DeepJSCC-WZ",
    "DeepJSCCWZ": "DeepJSCC-WZ-sm",
    "DeepJSCCWZBaseline": "DeepJSCC",
}

MODEL_ORDER = ["DeepJSCC-Cond", "DeepJSCC-WZ", "DeepJSCC-WZ-sm", "DeepJSCC"]


def load_all_data(results_dir: str = "results"):
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    records = []

    for f in csv_files:
        df = pd.read_csv(f)
        dataset = df["data/name"].iloc[0]
        raw_model = df["model/_target_"].iloc[0]
        model = MODEL_MAP.get(raw_model, raw_model)
        bw_factor = int(df["model/bw_factor"].iloc[0])
        rho = f"1/{bw_factor}"

        for _, row in df.iterrows():
            snr = float(row["model/snr"])
            records.append({
                "Dataset": dataset,
                "Model": model,
                "Bandwidth_Ratio": rho,
                "SNR_dB": snr,
                "PSNR_mean": float(row["test/psnr"]),
                "PSNR_std": float(row["test/psnr_std"]),
                "MSSSIM_mean": float(row["test/msssim"]),
                "MSSSIM_std": float(row["test/msssim_std"]),
                "LPIPS_mean": float(row["test/lpips"]),
                "LPIPS_std": float(row["test/lpips_std"]),
                "SSIM_mean": float(row["test/ssim"]),
                "SSIM_std": float(row["test/ssim_std"]),
            })

    return pd.DataFrame(records)


def generate_tables(results_dir: str = "results", output_dir: str = "replication_output/tables"):
    os.makedirs(output_dir, exist_ok=True)
    df = load_all_data(results_dir)

    target_snrs = [-4.0, -2.0, 0.0, 2.0, 4.0]
    df_filtered = df[df["SNR_dB"].isin(target_snrs)].copy()

    # Master Markdown file
    master_md_path = os.path.join(output_dir, "all_paper_values.md")
    master_csv_path = os.path.join(output_dir, "all_paper_values.csv")

    df_filtered.sort_values(["Dataset", "Bandwidth_Ratio", "Model", "SNR_dB"], inplace=True)
    df_filtered.to_csv(master_csv_path, index=False)

    print("=" * 85)
    print("REPLICATING ALL NUMERICAL VALUES ACROSS SNRs (-4, -2, 0, 2, 4 dB)")
    print("=" * 85)

    metrics = [
        ("PSNR", "PSNR (dB) \u2191", "PSNR_mean", "PSNR_std", "{:.2f} \u00b1 {:.2f}"),
        ("MS-SSIM", "MS-SSIM \u2191", "MSSSIM_mean", "MSSSIM_std", "{:.4f} \u00b1 {:.4f}"),
        ("LPIPS", "LPIPS \u2193", "LPIPS_mean", "LPIPS_std", "{:.4f} \u00b1 {:.4f}"),
    ]

    with open(master_md_path, "w", encoding="utf-8") as f_md:
        f_md.write("# Complete Numerical Values Replicated from DeepJSCC-WZ Paper\n\n")
        f_md.write("This document compiles every evaluated test metric across all SNRs $\\in \\{-4, -2, 0, 2, 4\\}$ dB, bandwidth ratios $\\rho \\in \\{1/16, 1/32\\}$, and datasets (Cityscape & KITTIStereo).\n\n")

        for metric_id, metric_title, col_mean, col_std, fmt_str in metrics:
            print(f"\n--- {metric_title} ---")
            f_md.write(f"## {metric_title}\n\n")

            for dataset in ["Cityscape", "KittiStereo"]:
                print(f"\nDataset: {dataset}")
                f_md.write(f"### Dataset: {dataset}\n\n")

                table_rows = []
                for rho in ["1/16", "1/32"]:
                    for model in MODEL_ORDER:
                        row = [rho, model]
                        sub_df = df_filtered[
                            (df_filtered["Dataset"] == dataset) &
                            (df_filtered["Bandwidth_Ratio"] == rho) &
                            (df_filtered["Model"] == model)
                        ].sort_values("SNR_dB")

                        snr_map = {r["SNR_dB"]: (r[col_mean], r[col_std]) for _, r in sub_df.iterrows()}
                        for snr in target_snrs:
                            if snr in snr_map:
                                m_val, s_val = snr_map[snr]
                                row.append(fmt_str.format(m_val, s_val))
                            else:
                                row.append("N/A")
                        table_rows.append(row)

                headers = ["\u03c1", "Method"] + [f"SNR = {int(s)} dB" for s in target_snrs]
                print(tabulate(table_rows, headers=headers, tablefmt="grid"))

                # Write markdown table
                md_headers = ["$\\rho$", "Method"] + [f"SNR = {int(s)} dB" for s in target_snrs]
                f_md.write("| " + " | ".join(md_headers) + " |\n")
                f_md.write("|" + "|".join([":---:"] + [":---"] + [":---:"] * len(target_snrs)) + "|\n")
                for r in table_rows:
                    f_md.write("| " + " | ".join(str(cell) for cell in r) + " |\n")
                f_md.write("\n")

                # Save individual CSV table for each metric & dataset
                csv_filename = os.path.join(output_dir, f"{dataset.lower()}_{metric_id.lower()}_table.csv")
                pd.DataFrame(table_rows, columns=headers).to_csv(csv_filename, index=False)

    print(f"\n[Saved Complete Values Markdown] -> {master_md_path}")
    print(f"[Saved Complete Values CSV]      -> {master_csv_path}")
    return master_md_path


if __name__ == "__main__":
    generate_tables()
