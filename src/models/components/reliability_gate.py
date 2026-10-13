import torch
import torch.nn as nn

class ContentAwareReliabilityGate(nn.Module):
    """
    SNR-Adaptive and Content-Aware Multi-Scale Reliability Gate.
    Dynamically balances the weight between transmitted noisy latent features x
    and receiver-side stereo features s based on channel SNR and Global Average Pooled content:
    
        alpha_l = sigma(MLP([snr_norm, GAP(x), GAP(s)]))
        x_fused = Concat([2 * alpha_l * x, 2 * (1 - alpha_l) * s])
        
    Identity Initialization:
        The final linear layer is strictly initialized to zeros. At step 0,
        alpha_l = 0.5, making 2 * alpha_l = 1.0, exactly preserving the baseline fusion.
    """
    def __init__(self, in_channels: int, hidden_dim: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.mlp = nn.Sequential(
            nn.Linear(1 + 2 * in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        # Identity initialization: zero-out the final layer
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x: torch.Tensor, s: torch.Tensor, csi: torch.Tensor):
        B, C, H, W = x.shape
        gap_x = torch.mean(x, dim=(2, 3))
        gap_s = torch.mean(s, dim=(2, 3))
        snr_norm = csi / 10.0  # normalize SNR from [-10, 10] dB into [-1.0, 1.0]
        
        feat = torch.cat([snr_norm, gap_x, gap_s], dim=1)
        logit = self.mlp(feat)
        alpha = torch.sigmoid(logit).view(B, 1, 1, 1)
        
        x_weighted = 2.0 * alpha * x
        s_weighted = 2.0 * (1.0 - alpha) * s
        
        fused = torch.cat([x_weighted, s_weighted], dim=1)
        return fused, alpha.squeeze()
