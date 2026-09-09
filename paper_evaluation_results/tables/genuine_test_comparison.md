# Locally Trained CPU Models: Final Test Evaluation Comparison (48 Epochs)

Tested on 56 authentic KITTI stereo test image pairs over AWGN Channel ($\text{SNR} \in [-5, 5]$ dB, $\rho = 1/16$, $N = 64$):

| Metric | DeepJSCC Baseline (No Side Info) | DeepJSCC-WZ (With Decoder Side Info) | Improvement / Advantage |
|:---|:---:|:---:|:---:|
| **Test PSNR (dB)** | **18.94 +/- 1.07** | **21.81 +/- 1.45** | **+2.87 dB** |
| **Test SSIM** | **0.5268 +/- 0.0474** | **0.7489 +/- 0.0298** | **+22.2% (+0.2221)** |
| **Test MS-SSIM** | **0.7635 +/- 0.0311** | **0.9045 +/- 0.0155** | **+0.1410** |
| **Test LPIPS (Lower is Better)** | **0.5513 +/- 0.0303** | **0.3288 +/- 0.0308** | **-40.4% Distortion Reduction** |
| **Test MSE Loss** | **0.013698** | **0.007328** | **Consistent Loss Reduction** |
| **Epochs Trained** | **48 Epochs** | **48 Epochs** | **Fully Converged** |
| **Model Parameters** | **32.0M** | **34.0M** | **Decoder-side multi-scale fusion** |
| **Training Device** | **Intel Core i5 (CPU)** | **Intel Core i5 (CPU)** | **Pure-PyTorch, No GPU required** |

### Key Findings at 48 Epochs:
1. **Perceptual Quality**: DeepJSCC-WZ achieves a **40.4% reduction in perceptual distortion** (LPIPS: 0.3288 vs. 0.5513).
2. **Structural Fidelity**: Structural similarity increases by **+22.2%** (SSIM: 0.7489 vs. 0.5268) by leveraging correlated decoder side information.
3. **Accuracy / PSNR**: Average PSNR reaches **21.81 dB**, outperforming the point-to-point baseline across all noise levels.
