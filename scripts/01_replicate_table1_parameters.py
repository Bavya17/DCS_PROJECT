"""
Replicate Table I from the paper:
'Table I: Number of parameters for the compared methods.'

This script instantiates each model architecture directly on CPU, counts the
total number of trainable parameters, and compares against the published Table I values.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from tabulate import tabulate

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Prevent downloading large pretrained VGG weights for LPIPS metric during parameter counting
from src.models.base_deepjscc import BaseDeepJSCC
BaseDeepJSCC.init_metrics = lambda self: None

from src.models.components.channels import ComplexAWGNChannel
from src.models.components.power_constraints import ComplexAveragePowerConstraint
from src.models.wz.deepjscc_wz_baseline import DeepJSCCWZBaseline
from src.models.wz.deepjscc_wz import DeepJSCCWZ
from src.models.wz.deepjscc_wz_baseline2 import DeepJSCCWZBaseline2
from src.models.wz.deepjscc_wz_joint2 import DeepJSCCWZJoint2


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def replicate_table1(output_dir: str = "replication_output/tables"):
    os.makedirs(output_dir, exist_ok=True)

    dummy_loss = lambda: nn.MSELoss()
    dummy_csi = lambda: lambda *args, **kwargs: torch.zeros(1, 1)

    base_kwargs = {
        "channel": ComplexAWGNChannel,
        "power_constraint": lambda: ComplexAveragePowerConstraint(1.0),
        "optimizer": lambda params: torch.optim.Adam(params, lr=1e-4),
        "scheduler": None,
        "loss": dummy_loss,
        "csi": dummy_csi,
        "input_dims": (3, 128, 256),
        "N": 256,
    }

    models_info = [
        ("DeepJSCC", DeepJSCCWZBaseline, "31.6M", "31.0M"),
        ("DeepJSCC-WZ-sm", DeepJSCCWZ, "39.8M", "38.6M"),
        ("DeepJSCC-WZ", DeepJSCCWZBaseline2, "48.9M", "47.4M"),
        ("DeepJSCC-Cond", DeepJSCCWZJoint2, "59.1M", "57.4M"),
    ]

    results = []
    print("=" * 75)
    print("REPLICATING TABLE I: Number of Parameters for the Compared Methods")
    print("Computing directly from PyTorch architectures on CPU...")
    print("=" * 75)

    for paper_name, model_cls, expected_16, expected_32 in models_info:
        # rho = 1/16 (bw_factor = 16)
        m_16 = model_cls(bw_factor=16, **base_kwargs)
        p_16 = count_parameters(m_16)
        p_16_str = f"{p_16 / 1e6:.1f}M"

        # rho = 1/32 (bw_factor = 32)
        m_32 = model_cls(bw_factor=32, **base_kwargs)
        p_32 = count_parameters(m_32)
        p_32_str = f"{p_32 / 1e6:.1f}M"

        match_16 = "✓ Match" if p_16_str == expected_16 else "✗ Diff"
        match_32 = "✓ Match" if p_32_str == expected_32 else "✗ Diff"

        results.append({
            "Method": paper_name,
            "rho = 1/16 (Computed)": p_16_str,
            "rho = 1/16 (Exact Params)": f"{p_16:,}",
            "rho = 1/16 (Paper)": expected_16,
            "rho = 1/32 (Computed)": p_32_str,
            "rho = 1/32 (Exact Params)": f"{p_32:,}",
            "rho = 1/32 (Paper)": expected_32,
            "Status": f"{match_16} | {match_32}"
        })

    # Summary table for paper reproduction
    paper_table = [
        [r["Method"], r["rho = 1/16 (Computed)"], r["rho = 1/32 (Computed)"], r["rho = 1/16 (Paper)"], r["rho = 1/32 (Paper)"]]
        for r in results
    ]
    headers = ["Method", "rho = 1/16 (Computed)", "rho = 1/32 (Computed)", "rho = 1/16 (Paper)", "rho = 1/32 (Paper)"]
    print(tabulate(paper_table, headers=headers, tablefmt="grid"))
    print("\nDetailed exact parameter counts:")
    detail_table = [
        [r["Method"], r["rho = 1/16 (Exact Params)"], r["rho = 1/32 (Exact Params)"], r["Status"]]
        for r in results
    ]
    print(tabulate(detail_table, headers=["Method", "rho = 1/16 (Exact Count)", "rho = 1/32 (Exact Count)", "Status"], tablefmt="simple"))

    # Save to Markdown
    md_path = os.path.join(output_dir, "table1_parameters.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Table I: Number of parameters for the compared methods\n\n")
        f.write("| Method | $\\rho = 1/16$ (Computed) | $\\rho = 1/32$ (Computed) | $\\rho = 1/16$ (Paper) | $\\rho = 1/32$ (Paper) | Exact Count ($\\rho=1/16$) | Exact Count ($\\rho=1/32$) |\n")
        f.write("|:-------|:------------------------:|:------------------------:|:---------------------:|:---------------------:|:--------------------------:|:--------------------------:|\n")
        for r in results:
            f.write(f"| {r['Method']} | **{r['rho = 1/16 (Computed)']}** | **{r['rho = 1/32 (Computed)']}** | {r['rho = 1/16 (Paper)']} | {r['rho = 1/32 (Paper)']} | {r['rho = 1/16 (Exact Params)']} | {r['rho = 1/32 (Exact Params)']} |\n")
    print(f"\n[Saved Markdown] -> {md_path}")

    # Save to CSV
    csv_path = os.path.join(output_dir, "table1_parameters.csv")
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[Saved CSV]      -> {csv_path}")

    # Save to LaTeX
    tex_path = os.path.join(output_dir, "table1_parameters.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Number of parameters for the compared methods.}\n")
        f.write("\\label{tab:num_parameters}\n")
        f.write("\\begin{tabular}{lcc}\n\\toprule\n")
        f.write("Method & $\\rho = 1/16$ & $\\rho = 1/32$ \\\\\n\\midrule\n")
        for r in results:
            f.write(f"{r['Method']} & {r['rho = 1/16 (Computed)']} & {r['rho = 1/32 (Computed)']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"[Saved LaTeX]    -> {tex_path}")

    return results


if __name__ == "__main__":
    replicate_table1()
