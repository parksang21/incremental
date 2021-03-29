import torch
import torch.nn as nn


class CosineSim(nn.Module):
    def __init__(self, eps=1e-8):
        super(CosineSim, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eps = eps

    def forward(self, a, b):
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, self.eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, self.eps * torch.ones_like(b_n))

        return torch.mm(a_norm, b_norm.transpose(0, 1))
