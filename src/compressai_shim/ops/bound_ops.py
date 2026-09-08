import torch
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
