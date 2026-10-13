import torch
import torch.nn as nn
from compressai.layers import ResidualBlockWithStride
from src.models.components.afmodule import AFModule
from src.models.components.reliability_gate import ContentAwareReliabilityGate
from src.models.wz.deepjscc_wz_baseline2 import DeepJSCCWZBaseline2
from src.models.nets.utils import calculate_num_filters

class DeepJSCCWZ_SARG(DeepJSCCWZBaseline2):
    """
    DeepJSCC-WZ with Content-Aware Multi-Scale Reliability Gating (SARG).
    Replaces static feature concatenation with dynamic, SNR- and content-aware gating
    across all 5 multi-scale decoder fusion points.
    """
    def __init__(self, N=64, bw_factor=16, **kwargs):
        super().__init__(N=N, bw_factor=bw_factor, **kwargs)
        
        M = calculate_num_filters(4, bw_factor)
        
        # 5 multi-scale fusion points in g_s: idx in [0, 3, 6, 10, 13]
        # in_channels: M=96, N=64, N=64, N=64, 3
        self.gates = nn.ModuleList([
            ContentAwareReliabilityGate(in_channels=M, hidden_dim=32),
            ContentAwareReliabilityGate(in_channels=N, hidden_dim=32),
            ContentAwareReliabilityGate(in_channels=N, hidden_dim=32),
            ContentAwareReliabilityGate(in_channels=N, hidden_dim=32),
            ContentAwareReliabilityGate(in_channels=3, hidden_dim=16),
        ])

    def forward(self, batch, return_alphas=False):
        (x, xs), csi = batch
        
        csi_transmitter = csi
        csi_sideinfo = csi
        
        for layer in self.g_a:
            if isinstance(layer, AFModule):
                x = layer((x, csi_transmitter))
            else:
                x = layer(x)
        
        x = self.power_constraint(x)
        x = self.channel((x, csi))
        
        xs_list = []
        for idx, layer in enumerate(self.g_a2):
            if isinstance(layer, ResidualBlockWithStride):
                xs_list.append(xs)
            if isinstance(layer, AFModule):
                xs = layer((xs, csi_sideinfo))
            else:
                xs = layer(xs)
        xs_list.append(xs)
        
        alphas = []
        gate_idx = 0
        for idx, layer in enumerate(self.g_s):
            if idx in [0, 3, 6, 10, 13]:
                last_xs = xs_list.pop()
                x, alpha_val = self.gates[gate_idx](x, last_xs, csi)
                alphas.append(alpha_val)
                gate_idx += 1
            
            if isinstance(layer, AFModule):
                x = layer((x, csi))
            else:
                x = layer(x)
        
        if return_alphas:
            return x, alphas
        return x

    def step(self, batch):
        x, csi = batch
        x_hat = self.forward((x, csi))
        target = x[0]
        loss = self.loss(x_hat, target)
        return loss, x_hat, target
