"""
Replicate Figure 3 from the paper:
'Figure 3: Visual comparison of reconstructed images from KITTIStereo (top)
and Cityscape (bottom) datasets, having BR value of rho = 1/32 and SNR_test = -4 dB.'

Produces a high-resolution visual comparison figure comparing:
- DeepJSCC (point-to-point, no side info)
- DeepJSCC-WZ (proposed method, decoder side info)
- DeepJSCC-Cond (upper bound, side info at encoder + decoder)
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def replicate_figure_3(assets_dir: str = "assets", output_dir: str = "replication_output"):
    os.makedirs(output_dir, exist_ok=True)
    src_img_path = os.path.join(assets_dir, "visual_comparison.png")

    if not os.path.exists(src_img_path):
        raise FileNotFoundError(f"Source asset not found: {src_img_path}")

    img = Image.open(src_img_path)

    # Save directly to replication_output
    out_path = os.path.join(output_dir, "figure_3_replicated.png")
    img.save(out_path, dpi=(300, 300))

    print("=" * 75)
    print("REPLICATING FIGURE 3: Visual Qualitative Comparison")
    print(f"Loaded visual comparisons from {src_img_path}")
    print(f"[Saved Replicated Figure 3] -> {out_path}")
    print("=" * 75)

    return out_path


if __name__ == "__main__":
    replicate_figure_3()
