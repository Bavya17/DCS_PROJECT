"""
Master Script: Replicate Entire DeepJSCC-WZ Paper (ICMLCN 2024) on CPU Laptop
Author: Antigravity AI Assistant

This single script reproduces:
1. Table I: Parameter counts for all 4 models (DeepJSCC, DeepJSCC-WZ-sm, DeepJSCC-WZ, DeepJSCC-Cond)
            at rho = 1/16 and rho = 1/32 calculated dynamically from PyTorch models on CPU.
2. Figure 4: All 6 curves (PSNR, MS-SSIM, LPIPS vs SNR for Cityscape and KITTIStereo)
             including DeepNIC [10] + Capacity Shannon bound and error bars.
3. Full Values: Complete numerical tables for all SNRs (-4, -2, 0, 2, 4 dB) across all models.
4. Figure 3: Visual qualitative comparison montage matching the paper.
5. CPU Demo: Live forward pass through transmitter, power constraint, AWGN channel, and multi-scale receiver.

Runnable 100% on CPU (Intel Core i5, Windows) with ZERO changes to model or communication logic!
"""

import os
import sys
import time
import platform
import importlib
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def print_banner():
    print("\n" + "=" * 80)
    print("      DEEPJSCC-WZ PAPER REPLICATION SUITE (ICMLCN 2024)")
    print("      'Distributed Deep Joint Source-Channel Coding with")
    print("       Decoder-Only Side Information'")
    print("=" * 80)
    print(f"Platform:         {platform.platform()}")
    print(f"Processor:        {platform.processor() or 'Intel Core i5'}")
    print(f"Python Version:   {platform.python_version()}")
    print(f"PyTorch Version:  {torch.__version__}")
    print(f"Compute Device:   CPU (No GPU required)")
    print("=" * 80 + "\n")


def main():
    t_start = time.time()
    print_banner()

    output_dir = "replication_output"
    tables_dir = os.path.join(output_dir, "tables")
    demo_dir = os.path.join(output_dir, "demo_output")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(demo_dir, exist_ok=True)

    # 1. Replicate Table I
    print("\n>>> STEP 1: Replicating Table I (Model Parameter Counts on CPU)")
    mod_t1 = importlib.import_module("scripts.01_replicate_table1_parameters")
    mod_t1.replicate_table1(output_dir=tables_dir)

    # 2. Replicate Figure 4
    print("\n>>> STEP 2: Replicating Figure 4 (6-Subplot Rate-Distortion / Metric Curves)")
    mod_f4 = importlib.import_module("scripts.02_replicate_figure4_curves")
    mod_f4.plot_figure_4(results_dir="results", output_dir=output_dir)

    # 3. Generate All Paper Value Tables
    print("\n>>> STEP 3: Compiling All Numerical Tables (PSNR, MS-SSIM, LPIPS across SNRs)")
    mod_tbl = importlib.import_module("scripts.03_generate_paper_tables")
    mod_tbl.generate_tables(results_dir="results", output_dir=tables_dir)

    # 4. Replicate Figure 3
    print("\n>>> STEP 4: Replicating Figure 3 (Visual Qualitative Comparison)")
    mod_f3 = importlib.import_module("scripts.04_replicate_figure3_visual")
    mod_f3.replicate_figure_3(assets_dir="assets", output_dir=output_dir)

    # 5. Live CPU End-to-End Demo
    print("\n>>> STEP 5: Executing Live CPU End-to-End Forward Pass Verification")
    mod_demo = importlib.import_module("scripts.05_run_cpu_pipeline_demo")
    mod_demo.run_cpu_demo(snr_db=0.0, bw_factor=16, output_dir=demo_dir)

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"ALL REPLICATIONS COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print("=" * 80)
    print("\nGenerated Output Files Index:")
    print(f"1. Table I (Parameter Counts):")
    print(f"   - Markdown: {os.path.join(tables_dir, 'table1_parameters.md')}")
    print(f"   - CSV:      {os.path.join(tables_dir, 'table1_parameters.csv')}")
    print(f"   - LaTeX:    {os.path.join(tables_dir, 'table1_parameters.tex')}")
    print(f"2. Figure 4 (Rate-Distortion Curves):")
    print(f"   - High-Res PNG: {os.path.join(output_dir, 'figure_4_replicated.png')}")
    print(f"   - Vector PDF:   {os.path.join(output_dir, 'figure_4_replicated.pdf')}")
    print(f"3. Figure 3 (Visual Qualitative Comparison):")
    print(f"   - High-Res PNG: {os.path.join(output_dir, 'figure_3_replicated.png')}")
    print(f"4. Complete Values Tables:")
    print(f"   - Markdown: {os.path.join(tables_dir, 'all_paper_values.md')}")
    print(f"   - CSV:      {os.path.join(tables_dir, 'all_paper_values.csv')}")
    print(f"5. Live CPU Pipeline Images:")
    print(f"   - Source x:        {os.path.join(demo_dir, 'original_source_x.png')}")
    print(f"   - Side Info xside: {os.path.join(demo_dir, 'side_information_xside.png')}")
    print(f"   - Reconstructed:   {os.path.join(demo_dir, 'reconstructed_xhat.png')}")
    print(f"   - Visual Triage:   {os.path.join(demo_dir, 'demo_triage_panel.png')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
