import torch
from torch import nn


class MovingAverageNormalzier(nn.Module):
    def __init__(self, alpha, dims, fixed=True):
        super(MovingAverageNormalzier, self).__init__()
        self.alpha = alpha
        self.mean = nn.Parameter(torch.zeros(1, *dims), requires_grad=False)
        self.var = nn.Parameter(torch.ones(1, *dims), requires_grad=False)
        self.fixed = fixed

    def forward(self, x):
        out = (x - self.mean) / torch.sqrt(self.var + 1e-12)
        if self.training and not self.fixed:
            batch_mean = torch.mean(x, dim=0, keepdims=True).data
            self.mean.data = (
                1.0 - self.alpha
            ) * batch_mean + self.alpha * self.mean.data
            batch_var = torch.sum((x - self.mean) ** 2, dim=0, keepdims=True).data
            self.var.data = (1.0 - self.alpha) * batch_var + self.alpha * self.var.data

        return out
