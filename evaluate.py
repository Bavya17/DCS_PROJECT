"""
run_full_paper_evaluation.py: Master Evaluation Runner
======================================================
Reproduces the complete empirical evaluation suite for:
"Distributed Deep Joint Source-Channel Coding with Decoder-Only Side Information" (IEEE ICMLCN 2024)

Pipeline Stages:
1. Model Complexity: Computes parameter counts across all models for Table I on CPU.
2. Performance Curves: Plots PSNR, MS-SSIM, and LPIPS vs SNR curves for Figure 4.
3. Performance Matrices: Tabulates all numerical metrics across SNRs (-4 to 4 dB).
4. Qualitative Comparison: Generates visual reconstruction comparison panel for Figure 3.
5. End-to-End Simulation: Simulates live transmission pipeline (Tx -> AWGN -> Rx) on CPU.

Execution: 100% CPU-compatible (Intel Core i5, Windows/Linux/macOS).
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
    print("\n" + "=" * 85)
    print("      DEEPJSCC-WZ COMPREHENSIVE PAPER EVALUATION SUITE (IEEE ICMLCN 2024)")
    print("      'Distributed Deep Joint Source-Channel Coding with")
    print("       Decoder-Only Side Information'")
    print("=" * 85)
    print(f"Platform:         {platform.platform()}")
    print(f"Processor:        {platform.processor() or 'Intel Core i5'}")
    print(f"Python Version:   {platform.python_version()}")
    print(f"PyTorch Version:  {torch.__version__}")
    print(f"Compute Device:   CPU (Zero GPU dependency)")
    print("=" * 85 + "\n")


def main():
    t_start = time.time()
    print_banner()

    output_dir = "paper_evaluation_results"
    tables_dir = os.path.join(output_dir, "tables")
    demo_dir = os.path.join(output_dir, "transmission_demo")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(demo_dir, exist_ok=True)

    # 1. Model Complexity Evaluation (Table I)
    print("\n>>> STAGE 1: Evaluating Model Complexity & Parameter Counts (Table I)")
    mod_t1 = importlib.import_module("scripts.compute_model_complexity")
    mod_t1.compute_model_complexity_table1(output_dir=tables_dir)

    # 2. Rate-Distortion & Quality Metrics Curves (Figure 4)
    print("\n>>> STAGE 2: Plotting Rate-Distortion & Perceptual Quality Curves (Figure 4)")
    mod_f4 = importlib.import_module("scripts.plot_rate_distortion_curves")
    mod_f4.plot_rate_distortion_curves_fig4(results_dir="results", output_dir=output_dir)

    # 3. Comprehensive Performance Metrics Matrix
    print("\n>>> STAGE 3: Tabulating Full Numerical Performance Metrics")
    mod_tbl = importlib.import_module("scripts.tabulate_paper_metrics")
    mod_tbl.tabulate_performance_metrics(results_dir="results", output_dir=tables_dir)

    # 4. Visual Reconstruction Comparison (Figure 3)
    print("\n>>> STAGE 4: Generating Visual Qualitative Reconstructions (Figure 3)")
    mod_f3 = importlib.import_module("scripts.generate_visual_reconstructions")
    mod_f3.generate_visual_comparison_fig3(assets_dir="assets", output_dir=output_dir)

    # 5. Live Transmission Pipeline Simulation on CPU
    print("\n>>> STAGE 5: Simulating Live DeepJSCC-WZ Forward Transmission Pipeline on CPU")
    mod_demo = importlib.import_module("scripts.simulate_wireless_transmission")
    mod_demo.simulate_deepjscc_wz_transmission(snr_db=0.0, bw_factor=16, output_dir=demo_dir)

    total_time = time.time() - t_start
    print("\n" + "=" * 85)
    print(f"FULL PAPER EVALUATION SUITE COMPLETED IN {total_time:.2f} SECONDS!")
    print("=" * 85)
    print("\nGenerated Evaluation Files Index:")
    print(f"1. Model Complexity (Table I):")
    print(f"   - Markdown: {os.path.join(tables_dir, 'table1_model_complexity.md')}")
    print(f"   - CSV:      {os.path.join(tables_dir, 'table1_model_complexity.csv')}")
    print(f"   - LaTeX:    {os.path.join(tables_dir, 'table1_model_complexity.tex')}")
    print(f"2. Rate-Distortion Curves (Figure 4):")
    print(f"   - High-Res PNG: {os.path.join(output_dir, 'fig4_performance_curves.png')}")
    print(f"   - Vector PDF:   {os.path.join(output_dir, 'fig4_performance_curves.pdf')}")
    print(f"3. Visual Reconstructions (Figure 3):")
    print(f"   - High-Res PNG: {os.path.join(output_dir, 'fig3_visual_reconstructions.png')}")
    print(f"4. Performance Metrics Summary:")
    print(f"   - Markdown: {os.path.join(tables_dir, 'performance_metrics_summary.md')}")
    print(f"   - CSV:      {os.path.join(tables_dir, 'performance_metrics_summary.csv')}")
    print(f"5. Simulated Transmission Images:")
    print(f"   - Source x:        {os.path.join(demo_dir, 'source_image_x.png')}")
    print(f"   - Side Info xside: {os.path.join(demo_dir, 'decoder_side_information_xside.png')}")
    print(f"   - Reconstructed:   {os.path.join(demo_dir, 'reconstructed_image_xhat.png')}")
    print(f"   - Visual Panel:    {os.path.join(demo_dir, 'transmission_comparison_panel.png')}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
