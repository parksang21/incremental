import torch
import torch.nn as nn
import torch.nn.functional as F

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


def normalize(x, dim=-1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def cosine_sim(x):
    x = normalize(x)
    return torch.mm(x, x.T)


def center_cosine(c, x):
    # input shape c: n x d, x: b x d
    c = F.normalize(c, p=2, dim=-1, eps=1e-8).unsqueeze(1)
    x = F.normalize(x, p=2, dim=-1, eps=1e-8)
    return torch.sum(c * x, dim=-1)


def L2_dist(c, x):
    return torch.cdist(c, x) ** 2
