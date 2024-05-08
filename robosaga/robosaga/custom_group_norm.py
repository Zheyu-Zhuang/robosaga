import torch
import torch.nn as nn


class CustomGroupNorm(nn.Module):
    """
    A simple wrapper around nn.GroupNorm to compute fullgrad_bias that is dependent on the input features
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super(CustomGroupNorm, self).__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels, eps, affine)
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.bias = torch.zeros(num_channels)
        self.fullgrad_bias = None

    def forward(self, x):
        if self.training:
            return self.group_norm(x)
        return self.eval_forward(x)

    def eval_forward(self, x):
        out = self.group_norm(x)
        with torch.no_grad():
            B, C, H, W = x.size()
            x = x.view(B, self.num_groups, -1)
            # important to detach the mean and var from the graph, ensuring the completeness of fullgrad
            mean = x.mean(dim=-1, keepdim=True).detach()
            var = x.var(dim=-1, unbiased=False, keepdim=True).detach()
            w, b = self.group_norm.weight.view(1, C, 1, 1), self.group_norm.bias.view(1, C, 1, 1)
            m = mean.repeat_interleave(C // self.num_groups, dim=1).view(B, C, 1, 1)
            v = var.repeat_interleave(C // self.num_groups, dim=1).view(B, C, 1, 1)
            self.fullgrad_bias = -(m * w / torch.sqrt(v + self.eps)) + b
        return out
