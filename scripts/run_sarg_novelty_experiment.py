import os
import sys
import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.base_deepjscc import BaseDeepJSCC
BaseDeepJSCC.init_metrics = lambda self: None

from src.models.components.channels import ComplexAWGNChannel
from src.models.components.power_constraints import ComplexAveragePowerConstraint
from src.models.wz.deepjscc_wz_baseline2 import DeepJSCCWZBaseline2
from src.models.wz.deepjscc_wz_sarg import DeepJSCCWZ_SARG
from src.data.stereo import PairKittiCityscape

def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

def compute_ssim_tensor(img1, img2):
    from torchmetrics.functional.image import structural_similarity_index_measure
    return structural_similarity_index_measure(img1, img2, data_range=(0.0, 1.0)).item()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_base_kwargs():
    return {
        "channel": ComplexAWGNChannel,
        "power_constraint": lambda: ComplexAveragePowerConstraint(1.0),
        "optimizer": lambda p: torch.optim.Adam(p, lr=5e-5),
        "scheduler": None,
        "loss": lambda: nn.MSELoss(),
        "csi": lambda: lambda *args, **kwargs: torch.zeros(1, 1),
        "input_dims": (3, 64, 128),
        "N": 64,
        "bw_factor": 16
    }

def train_model(model, train_loader, val_loader, num_epochs=48, lr=5e-5, name="model", save_dir="checkpoints_sarg", seed=42):
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, f"{name}_best.pth")
    
    set_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float("inf")
    history = []
    
    print(f"\n[{name.upper()}] Starting {num_epochs}-Epoch Run (Seed: {seed}, LR: {lr})...", flush=True)
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for (l_img, r_img), _ in train_loader:
            B = l_img.shape[0]
            # Sample continuous SNR in [-5, 5] dB during training
            snr = torch.empty(B, 1).uniform_(-5.0, 5.0)
            
            optimizer.zero_grad()
            out = model(((l_img, r_img), snr))
            loss = criterion(out, l_img)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
            
        train_loss /= max(1, n_batches)
        
        # Validation on Val Set only
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        n_val = 0
        with torch.no_grad():
            for (l_img, r_img), _ in val_loader:
                B = l_img.shape[0]
                # Fixed SNR 0 dB for stable validation checkpointing
                snr = torch.zeros(B, 1)
                torch.manual_seed(1000 + n_val)
                out = model(((l_img, r_img), snr)).clip(0, 1)
                v_loss = criterion(out, l_img).item()
                v_psnr = compute_psnr(out, l_img)
                val_loss += v_loss
                val_psnr += v_psnr
                n_val += 1
                
        val_loss /= max(1, n_val)
        val_psnr /= max(1, n_val)
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt_path)
            
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "is_best": is_best
        })
        
        star = " *" if is_best else ""
        if epoch % 2 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"[{name.upper()}] Epoch {epoch:02d}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val PSNR: {val_psnr:.2f} dB{star}", flush=True)
        
    total_time = time.time() - start_time
    print(f"[{name.upper()}] {num_epochs} Epochs completed in {total_time:.1f}s. Best Val Loss: {best_val_loss:.6f} saved to {best_ckpt_path}", flush=True)
    return best_ckpt_path, pd.DataFrame(history)

def run_experiment():
    print("=" * 80, flush=True)
    print("RUNNING 48-EPOCH CONTENT-AWARE RELIABILITY GATE (SARG) EXPERIMENT", flush=True)
    print("Protocol: Fixed noise seeds, 20 draws/img/SNR, validation-only checkpointing", flush=True)
    print("=" * 80, flush=True)
    
    # 1. Dataset setup
    train_ds = PairKittiCityscape("data", "KittiStereo", "train", resize=(64, 128))
    val_ds = PairKittiCityscape("data", "KittiStereo", "val", resize=(64, 128))
    test_ds = PairKittiCityscape("data", "KittiStereo", "test", resize=(64, 128))
    
    print(f"Dataset Loaded: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}", flush=True)
    
    # Same data order for both runs via generator with fixed seed
    g_train = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, generator=g_train)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    
    # 2. Base checkpoint to initialize from
    base_ckpt_path = "logs/deepjscc_wz_cpu_small_scale/runs/2026-09-09_20-16-02/checkpoints/last.ckpt"
    if not os.path.exists(base_ckpt_path):
        raise FileNotFoundError(f"Base checkpoint not found at {base_ckpt_path}")
    print(f"Loading Base 48-Epoch Checkpoint: {base_ckpt_path}", flush=True)
    ckpt_dict = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    
    base_kwargs = get_base_kwargs()
    
    # 3. Model A: Control (DeepJSCCWZBaseline2)
    m_control = DeepJSCCWZBaseline2(**base_kwargs)
    m_control.load_state_dict(ckpt_dict, strict=False)
    
    # 4. Model B: Novelty (DeepJSCCWZ_SARG)
    m_sarg = DeepJSCCWZ_SARG(**base_kwargs)
    m_sarg.load_state_dict(ckpt_dict, strict=False)
    
    # Run Training: Control for 48 epochs
    control_ckpt, df_control = train_model(m_control, train_loader, val_loader, num_epochs=48, lr=5e-5, name="control", seed=42)
    
    # Reset train generator for exact identical data order
    g_train.manual_seed(42)
    train_loader_sarg = DataLoader(train_ds, batch_size=4, shuffle=True, generator=g_train)
    
    # Run Training: SARG Novelty for 48 epochs
    sarg_ckpt, df_sarg = train_model(m_sarg, train_loader_sarg, val_loader, num_epochs=48, lr=5e-5, name="sarg", seed=42)
    
    # 5. Rigorous Evaluation Phase
    print("\n" + "=" * 80, flush=True)
    print("EVALUATING ON 60 TEST SCENES (20 NOISE DRAWS PER IMAGE PER SNR)", flush=True)
    print("=" * 80, flush=True)
    
    m_control.load_state_dict(torch.load(control_ckpt, map_location="cpu", weights_only=False))
    m_sarg.load_state_dict(torch.load(sarg_ckpt, map_location="cpu", weights_only=False))
    m_control.eval()
    m_sarg.eval()
    
    snr_list = [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0]
    num_noise_draws = 20
    
    eval_results = []
    alpha_records = []
    
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    for snr_val in snr_list:
        print(f"Testing SNR = {snr_val:+5.1f} dB ({len(test_ds)} images x {num_noise_draws} noise realizations)...", flush=True)
        snr_tensor = torch.tensor([[snr_val]])
        
        psnr_ctrl_list = []
        psnr_sarg_list = []
        ssim_ctrl_list = []
        ssim_sarg_list = []
        diff_psnr_list = []
        diff_ssim_list = []
        
        alphas_at_snr = [[] for _ in range(5)]
        
        with torch.no_grad():
            for img_idx, ((l_img, r_img), _) in enumerate(test_loader):
                for noise_seed in range(num_noise_draws):
                    # Identical noise draw for both models
                    torch.manual_seed(noise_seed * 1000 + img_idx)
                    out_ctrl = m_control(((l_img, r_img), snr_tensor)).clip(0, 1)
                    
                    torch.manual_seed(noise_seed * 1000 + img_idx)
                    out_sarg, alphas = m_sarg(((l_img, r_img), snr_tensor), return_alphas=True)
                    out_sarg = out_sarg.clip(0, 1)
                    
                    p_c = compute_psnr(out_ctrl, l_img)
                    p_s = compute_psnr(out_sarg, l_img)
                    s_c = compute_ssim_tensor(out_ctrl, l_img)
                    s_s = compute_ssim_tensor(out_sarg, l_img)
                    
                    psnr_ctrl_list.append(p_c)
                    psnr_sarg_list.append(p_s)
                    ssim_ctrl_list.append(s_c)
                    ssim_sarg_list.append(s_s)
                    diff_psnr_list.append(p_s - p_c)
                    diff_ssim_list.append(s_s - s_c)
                    
                    if noise_seed == 0:  # record alphas once per image
                        for s_idx, a in enumerate(alphas):
                            alphas_at_snr[s_idx].append(a.mean().item())
                            
        # Compute mean and standard deviations
        mean_p_ctrl = np.mean(psnr_ctrl_list)
        std_p_ctrl = np.std(psnr_ctrl_list)
        mean_p_sarg = np.mean(psnr_sarg_list)
        std_p_sarg = np.std(psnr_sarg_list)
        
        mean_diff_p = np.mean(diff_psnr_list)
        std_diff_p = np.std(diff_psnr_list)
        
        mean_s_ctrl = np.mean(ssim_ctrl_list)
        mean_s_sarg = np.mean(ssim_sarg_list)
        mean_diff_s = np.mean(diff_ssim_list)
        std_diff_s = np.std(diff_ssim_list)
        
        # Paired t-statistic
        n_samples = len(diff_psnr_list)
        t_stat = mean_diff_p / (std_diff_p / math.sqrt(n_samples) + 1e-9)
        
        mean_alphas = [np.mean(a) for a in alphas_at_snr]
        for s_idx, ma in enumerate(mean_alphas):
            alpha_records.append({
                "SNR_dB": snr_val,
                "Scale": s_idx,
                "Scale_Name": ["Bottleneck (96ch)", "Scale 1/8 (64ch)", "Scale 1/4 (64ch)", "Scale 1/2 (64ch)", "Scale 1/1 (3ch)"][s_idx],
                "Mean_Alpha": ma
            })
            
        print(f"  --> Control PSNR: {mean_p_ctrl:.2f} +/- {std_p_ctrl:.2f} dB | SARG PSNR: {mean_p_sarg:.2f} +/- {std_p_sarg:.2f} dB", flush=True)
        print(f"  --> Paired Delta PSNR: {mean_diff_p:+.3f} +/- {std_diff_p:.3f} dB (t = {t_stat:.2f})", flush=True)
        print(f"  --> Paired Delta SSIM: {mean_diff_s:+.4f} +/- {std_diff_s:.4f}", flush=True)
        print(f"  --> Learned Mean Alphas across scales: {[round(x, 3) for x in mean_alphas]}", flush=True)
        
        eval_results.append({
            "SNR_dB": snr_val,
            "Control_PSNR_mean": round(mean_p_ctrl, 3),
            "Control_PSNR_std": round(std_p_ctrl, 3),
            "SARG_PSNR_mean": round(mean_p_sarg, 3),
            "SARG_PSNR_std": round(std_p_sarg, 3),
            "Delta_PSNR_mean": round(mean_diff_p, 3),
            "Delta_PSNR_std": round(std_diff_p, 3),
            "Control_SSIM_mean": round(mean_s_ctrl, 4),
            "SARG_SSIM_mean": round(mean_s_sarg, 4),
            "Delta_SSIM_mean": round(mean_diff_s, 4),
            "Delta_SSIM_std": round(std_diff_s, 4),
            "Paired_t_stat": round(t_stat, 2),
            "Scale0_Bottleneck_Alpha": round(mean_alphas[0], 3),
            "Scale1_Alpha": round(mean_alphas[1], 3),
            "Scale2_Alpha": round(mean_alphas[2], 3),
            "Scale3_Alpha": round(mean_alphas[3], 3),
            "Scale4_Alpha": round(mean_alphas[4], 3)
        })
        
    # Save CSV & Markdown
    df_results = pd.DataFrame(eval_results)
    df_alphas = pd.DataFrame(alpha_records)
    
    os.makedirs("paper_evaluation_results/tables", exist_ok=True)
    csv_path = "paper_evaluation_results/tables/sarg_experiment_paired_results.csv"
    md_path = "paper_evaluation_results/tables/sarg_experiment_paired_results.md"
    alpha_csv = "paper_evaluation_results/tables/sarg_learned_alphas.csv"
    
    df_results.to_csv(csv_path, index=False)
    df_alphas.to_csv(alpha_csv, index=False)
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# SARG (Content-Aware Multi-Scale Reliability Gate) vs Control: 48-Epoch Paired Test Results\n\n")
        f.write("Evaluated on 60 KITTI Stereo test scenes across 20 fixed noise draws per image per SNR.\n\n")
        f.write(df_results.to_markdown(index=False))
        f.write("\n\n### Learned Gate Alpha Values Across Scales:\n\n")
        f.write(df_alphas.to_markdown(index=False))
        f.write("\n")
        
    print(f"\nResults saved to {csv_path} and {md_path}", flush=True)
    
    # 6. Plot Learned Alpha vs SNR
    plt.figure(figsize=(8, 5))
    scale_names = ["Bottleneck (96ch)", "Scale 1/8 (64ch)", "Scale 1/4 (64ch)", "Scale 1/2 (64ch)", "Scale 1/1 (3ch)"]
    for s_idx in range(5):
        s_data = df_alphas[df_alphas["Scale"] == s_idx]
        plt.plot(s_data["SNR_dB"], s_data["Mean_Alpha"], marker="o", linewidth=2, label=scale_names[s_idx])
    plt.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Equal Trust (Init α=0.5)")
    plt.xlabel("Channel SNR (dB)", fontsize=12, fontweight="bold")
    plt.ylabel("Learned Reliability Weight α (Trust on Wireless Latent)", fontsize=12, fontweight="bold")
    plt.title("Learned Multi-Scale Reliability Gate Weight α vs Channel SNR (48 Epochs)", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    alpha_plot_path = "paper_evaluation_results/sarg_learned_alpha_vs_snr.png"
    plt.savefig(alpha_plot_path, dpi=300)
    plt.close()
    print(f"Alpha plot saved to {alpha_plot_path}", flush=True)
    
    # 7. Plot Rate-Distortion / PSNR vs SNR Curve
    plt.figure(figsize=(8, 5))
    plt.plot(df_results["SNR_dB"], df_results["Control_PSNR_mean"], "o--", color="#1E293B", linewidth=2, label="Control (Fine-Tuned 48-Epoch WZ)")
    plt.plot(df_results["SNR_dB"], df_results["SARG_PSNR_mean"], "s-", color="#D97706", linewidth=2.5, label="Ours: DeepJSCC-WZ + SARG (Gated)")
    plt.xlabel("Channel SNR (dB)", fontsize=12, fontweight="bold")
    plt.ylabel("Test PSNR (dB)", fontsize=12, fontweight="bold")
    plt.title("Test PSNR vs Channel SNR: Control vs Content-Aware SARG (48 Epochs)", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    psnr_plot_path = "paper_evaluation_results/sarg_psnr_comparison_curves.png"
    plt.savefig(psnr_plot_path, dpi=300)
    plt.close()
    print(f"PSNR comparison plot saved to {psnr_plot_path}", flush=True)
    
    # 8. Visual Reconstruction Panel
    (l_test, r_test), _ = test_ds[0]
    l_test = l_test.unsqueeze(0)
    r_test = r_test.unsqueeze(0)
    snr_demo = torch.tensor([[0.0]])
    
    with torch.no_grad():
        torch.manual_seed(42)
        recon_ctrl = m_control(((l_test, r_test), snr_demo)).clip(0, 1)
        torch.manual_seed(42)
        recon_sarg = m_sarg(((l_test, r_test), snr_demo)).clip(0, 1)
        
    p_ctrl_demo = compute_psnr(recon_ctrl, l_test)
    p_sarg_demo = compute_psnr(recon_sarg, l_test)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(l_test.squeeze().permute(1, 2, 0).numpy())
    axes[0].set_title("Source Image X (Left GT)", fontweight="bold")
    axes[0].axis("off")
    
    axes[1].imshow(r_test.squeeze().permute(1, 2, 0).numpy())
    axes[1].set_title("Side Information X_side (Right GT)", fontweight="bold")
    axes[1].axis("off")
    
    axes[2].imshow(recon_ctrl.squeeze().permute(1, 2, 0).numpy())
    axes[2].set_title(f"Control WZ (PSNR: {p_ctrl_demo:.2f} dB)", fontweight="bold")
    axes[2].axis("off")
    
    axes[3].imshow(recon_sarg.squeeze().permute(1, 2, 0).numpy())
    axes[3].set_title(f"Proposed SARG (PSNR: {p_sarg_demo:.2f} dB)", fontweight="bold", color="#D97706")
    axes[3].axis("off")
    
    plt.tight_layout()
    panel_path = "paper_evaluation_results/transmission_demo/sarg_visual_comparison_panel.png"
    plt.savefig(panel_path, dpi=300)
    plt.close()
    print(f"Visual reconstruction panel saved to {panel_path}", flush=True)
    
    print("\n" + "=" * 80, flush=True)
    print("ALL 48-EPOCH EXPERIMENTAL RUNS AND EVALUATIONS COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 80, flush=True)

if __name__ == '__main__':
    run_experiment()
