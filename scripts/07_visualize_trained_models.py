import os
import sys
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.components.channels import ComplexAWGNChannel
from src.models.components.power_constraints import ComplexAveragePowerConstraint
from src.models.wz.deepjscc_wz_baseline2 import DeepJSCCWZBaseline2
from src.models.wz.deepjscc_wz_baseline import DeepJSCCWZBaseline
import torchvision.transforms.functional as TF

def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

def main():
    print("=" * 80)
    print("VISUALIZING RECONSTRUCTIONS FROM OUR LOCALLY TRAINED 10-EPOCH MODELS")
    print("=" * 80)

    # 1. Paths to checkpoints
    ckpt_wz = "logs/deepjscc_wz_cpu_small_scale/runs/2026-09-08_18-30-20/checkpoints/epoch_009.ckpt"
    ckpt_base = "logs/deepjscc_baseline_cpu_small_scale/runs/2026-09-08_18-37-40/checkpoints/epoch_009.ckpt"

    if not os.path.exists(ckpt_wz) or not os.path.exists(ckpt_base):
        print("Trained checkpoints not found!")
        return

    # 2. Instantiate models
    base_kwargs = {
        "channel": ComplexAWGNChannel,
        "power_constraint": lambda: ComplexAveragePowerConstraint(1.0),
        "optimizer": lambda p: torch.optim.Adam(p, lr=2e-4),
        "scheduler": None,
        "loss": lambda: nn.MSELoss(),
        "csi": lambda: lambda *args, **kwargs: torch.tensor([[4.0]]),
        "input_dims": (3, 64, 128),
        "N": 64,
        "bw_factor": 16
    }

    print("\n[1/4] Loading trained DeepJSCC-WZ model...")
    m_wz = DeepJSCCWZBaseline2(**base_kwargs)
    c_wz = torch.load(ckpt_wz, map_location="cpu", weights_only=False)
    m_wz.load_state_dict(c_wz["state_dict"], strict=False)
    m_wz.eval()

    print("[2/4] Loading trained DeepJSCC Baseline model...")
    m_base = DeepJSCCWZBaseline(**base_kwargs)
    c_base = torch.load(ckpt_base, map_location="cpu", weights_only=False)
    m_base.load_state_dict(c_base["state_dict"], strict=False)
    m_base.eval()

    # 3. Load an unseen test pair from KITTI stereo test set
    print("[3/4] Loading test stereo pair...")
    idx_test = "data/KITTI_stereo_test.txt"
    with open(idx_test) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    left_rel, right_rel = lines[0], lines[1]
    left_path = os.path.join("data", left_rel)
    right_path = os.path.join("data", right_rel)

    t_crop = transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.ToTensor()
    ])

    img_x = t_crop(Image.open(left_path).convert("RGB")).unsqueeze(0)
    img_xs = t_crop(Image.open(right_path).convert("RGB")).unsqueeze(0)
    csi = torch.tensor([[4.0]], dtype=torch.float32)

    # 4. Forward passes
    print("[4/4] Executing inference over AWGN channel at SNR = 4.0 dB...")
    with torch.no_grad():
        # Baseline (takes ((x, _), csi))
        recon_base = m_base.forward(((img_x, img_xs), csi))
        recon_base = torch.clamp(recon_base, 0.0, 1.0)

        # Proposed WZ (takes ((x, xs), csi))
        recon_wz = m_wz.forward(((img_x, img_xs), csi))
        recon_wz = torch.clamp(recon_wz, 0.0, 1.0)

    psnr_base = compute_psnr(recon_base, img_x)
    psnr_wz = compute_psnr(recon_wz, img_x)

    print(f"      Baseline Reconstruction PSNR: {psnr_base:.2f} dB")
    print(f"      DeepJSCC-WZ Reconstruction PSNR: {psnr_wz:.2f} dB (+{psnr_wz - psnr_base:.2f} dB Gain)")

    # 5. Save visual comparison panel
    os.makedirs("paper_evaluation_results/transmission_demo", exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    panels = [
        (img_x.squeeze(0).permute(1, 2, 0).numpy(), "(a) Source Image (x)\n[Ground Truth Target]", "#1E293B"),
        (img_xs.squeeze(0).permute(1, 2, 0).numpy(), "(b) Side Information (x_s)\n[Decoder Stereo View]", "#0284C7"),
        (recon_base.squeeze(0).permute(1, 2, 0).numpy(), f"(c) DeepJSCC Baseline\nPSNR: {psnr_base:.2f} dB (No Side Info)", "#7570B3"),
        (recon_wz.squeeze(0).permute(1, 2, 0).numpy(), f"(d) DeepJSCC-WZ (Proposed)\nPSNR: {psnr_wz:.2f} dB (With Side Info)", "#D95F02")
    ]

    for ax, (img_np, title, col) in zip(axes, panels):
        ax.imshow(img_np)
        ax.set_title(title, fontsize=11, fontweight="bold", color=col, pad=8)
        ax.axis("off")

    plt.suptitle("Our Locally Trained 10-Epoch CPU Models: Test Image Reconstruction Comparison", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_panel = "paper_evaluation_results/transmission_demo/locally_trained_reconstruction_panel.png"
    plt.savefig(out_panel, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSuccessfully generated visual comparison panel -> {out_panel}")

    # Also save individual reconstructed images
    TF.to_pil_image(recon_base.squeeze(0)).save("paper_evaluation_results/transmission_demo/test_recon_baseline.png")
    TF.to_pil_image(recon_wz.squeeze(0)).save("paper_evaluation_results/transmission_demo/test_recon_deepjscc_wz.png")
    print("Saved test_recon_baseline.png and test_recon_deepjscc_wz.png")

if __name__ == "__main__":
    main()
