import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

sys.path.insert(0, ".")
from src.data.stereo import PairKittiCityscape
from src.models.components.channels import ComplexAWGNChannel
from src.models.components.power_constraints import ComplexAveragePowerConstraint
from src.models.wz.deepjscc_wz_baseline2 import DeepJSCCWZBaseline2
from src.models.wz.deepjscc_wz_baseline import DeepJSCCWZBaseline

def main():
    print("=" * 80)
    print("VALIDATING 48-EPOCH CHECKPOINTS & GENERATING GENUINE RECONSTRUCTION")
    print("=" * 80)

    base_kwargs = {
        "channel": ComplexAWGNChannel,
        "power_constraint": lambda: ComplexAveragePowerConstraint(1.0),
        "optimizer": lambda p: torch.optim.Adam(p, lr=2e-4),
        "scheduler": None,
        "loss": lambda: torch.nn.MSELoss(),
        "csi": lambda: lambda *args, **kwargs: torch.tensor([[4.0]]),
        "input_dims": (3, 64, 128),
        "N": 64,
        "bw_factor": 16
    }

    # Load 48-epoch checkpoints
    ckpt_base = "logs/deepjscc_baseline_cpu_small_scale/runs/2026-09-08_20-00-59/checkpoints/last.ckpt"
    ckpt_wz = "logs/deepjscc_wz_cpu_small_scale/runs/2026-09-08_19-28-27/checkpoints/last.ckpt"

    print(f"Loading Baseline checkpoint: {ckpt_base}")
    m_base = DeepJSCCWZBaseline(**base_kwargs)
    c_b = torch.load(ckpt_base, map_location="cpu", weights_only=False)
    m_base.load_state_dict(c_b["state_dict"], strict=False)
    m_base.eval()

    print(f"Loading WZ checkpoint: {ckpt_wz}")
    m_wz = DeepJSCCWZBaseline2(**base_kwargs)
    c_w = torch.load(ckpt_wz, map_location="cpu", weights_only=False)
    m_wz.load_state_dict(c_w["state_dict"], strict=False)
    m_wz.eval()

    # Load authentic test dataset item
    ds = PairKittiCityscape("data", "KittiStereo", "test", resize=(64, 128))
    print(f"Total test set samples: {len(ds)}")

    sample_idx = 0
    (l, r), _ = ds[sample_idx]
    l = l.unsqueeze(0)
    r = r.unsqueeze(0)
    csi = torch.tensor([[4.0]])

    # Run genuine forward inference through AWGN channel
    with torch.no_grad():
        torch.manual_seed(42)
        pred_base = m_base(((l, r), csi)).clip(0, 1)

        torch.manual_seed(42)
        pred_wz = m_wz(((l, r), csi)).clip(0, 1)

    mse_b = torch.mean((pred_base - l)**2).item()
    psnr_b = peak_signal_noise_ratio(pred_base, l, data_range=(0.0, 1.0)).item()
    ssim_b = structural_similarity_index_measure(pred_base, l, data_range=(0.0, 1.0)).item()

    mse_w = torch.mean((pred_wz - l)**2).item()
    psnr_w = peak_signal_noise_ratio(pred_wz, l, data_range=(0.0, 1.0)).item()
    ssim_w = structural_similarity_index_measure(pred_wz, l, data_range=(0.0, 1.0)).item()

    print(f"\n--- Genuine 48-Epoch Test Metrics (Sample {sample_idx}, SNR = 4.0 dB, rho = 1/16) ---")
    print(f"Baseline: MSE = {mse_b:.6f} | PSNR = {psnr_b:.2f} dB | SSIM = {ssim_b:.4f}")
    print(f"WZ:       MSE = {mse_w:.6f} | PSNR = {psnr_w:.2f} dB | SSIM = {ssim_w:.4f}")
    print(f"Gain:     +{psnr_w - psnr_b:.2f} dB PSNR | +{ssim_w - ssim_b:.4f} SSIM")

    # Generate 4-panel visual comparison figure
    os.makedirs("paper_evaluation_results/transmission_demo", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), dpi=300)

    img_gt = TF.to_pil_image(l.squeeze(0))
    img_xs = TF.to_pil_image(r.squeeze(0))
    img_base = TF.to_pil_image(pred_base.squeeze(0))
    img_wz = TF.to_pil_image(pred_wz.squeeze(0))

    axes[0, 0].imshow(img_gt)
    axes[0, 0].set_title("(a) Source Image x (Ground Truth Test Left View)", fontsize=12, fontweight="bold", pad=8)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_xs)
    axes[0, 1].set_title("(b) Decoder Side Info x_s (Correlated Test Right View)", fontsize=12, fontweight="bold", pad=8)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(img_base)
    axes[1, 0].set_title(f"(c) DeepJSCC Baseline [48-Epoch Model]\nPSNR: {psnr_b:.2f} dB | SSIM: {ssim_b:.4f}", fontsize=12, fontweight="bold", pad=8)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(img_wz)
    axes[1, 1].set_title(f"(d) DeepJSCC-WZ Proposed [48-Epoch Model]\nPSNR: {psnr_w:.2f} dB | SSIM: {ssim_w:.4f} (+{ssim_w - ssim_b:.4f} SSIM)", fontsize=12, fontweight="bold", color="darkgreen", pad=8)
    axes[1, 1].axis("off")

    plt.suptitle("Genuine 48-Epoch Model Visual Reconstruction (AWGN Channel: SNR = 4.0 dB, rho = 1/16)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_panel = "paper_evaluation_results/transmission_demo/locally_trained_reconstruction_panel.png"
    plt.savefig(out_panel, dpi=300, bbox_inches="tight")
    plt.close()

    # Save dedicated high-res file
    out_panel_48ep = "paper_evaluation_results/genuine_48epochs_reconstruction_panel.png"
    img_panel = Image.open(out_panel)
    img_panel.save(out_panel_48ep)

    # Save individual reconstructed frames
    img_gt.save("paper_evaluation_results/transmission_demo/source_image_x.png")
    img_xs.save("paper_evaluation_results/transmission_demo/decoder_side_information_xside.png")
    img_base.save("paper_evaluation_results/transmission_demo/test_recon_baseline.png")
    img_wz.save("paper_evaluation_results/transmission_demo/test_recon_deepjscc_wz.png")

    print(f"\nSaved genuine reconstruction panel to:\n  - {out_panel}\n  - {out_panel_48ep}")

if __name__ == "__main__":
    main()
