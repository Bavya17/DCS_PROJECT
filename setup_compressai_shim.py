import os
import shutil

base_dir = os.path.join(".venv", "Lib", "site-packages", "compressai")
os.makedirs(os.path.join(base_dir, "layers"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "ops"), exist_ok=True)

with open(os.path.join(base_dir, "__init__.py"), "w", encoding="utf-8") as f:
    f.write('__version__ = "1.2.8"\n')

with open(os.path.join(base_dir, "ops", "__init__.py"), "w", encoding="utf-8") as f:
    f.write("from .parametrizers import NonNegativeParametrizer\nfrom .bound_ops import LowerBound\n")

with open(os.path.join(base_dir, "ops", "bound_ops.py"), "w", encoding="utf-8") as f:
    f.write('''import torch
import torch.nn as nn
from torch import Tensor

class LowerBoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return torch.max(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        pass_through_if = (x >= bound) | (grad_output < 0)
        return grad_output * pass_through_if.float(), None

class LowerBound(nn.Module):
    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([bound]))

    def forward(self, x: Tensor) -> Tensor:
        return LowerBoundFunction.apply(x, self.bound)
''')

shutil.copy("compressai_param_reference.py", os.path.join(base_dir, "ops", "parametrizers.py"))
shutil.copy("compressai_gdn_reference.py", os.path.join(base_dir, "layers", "gdn.py"))
shutil.copy("compressai_layers_reference.py", os.path.join(base_dir, "layers", "layers.py"))

with open(os.path.join(base_dir, "layers", "__init__.py"), "w", encoding="utf-8") as f:
    f.write('''from .layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv1x1,
    conv3x3,
    subpel_conv3x3,
)
from .gdn import GDN, GDN1
''')

print("CompressAI pure-PyTorch layer package created successfully!")
