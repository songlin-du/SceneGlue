import torch
from torch import nn
from copy import deepcopy
import torch.nn.functional as F
INF = 1E9


class TransLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):
        return self.ln(x.transpose(1,2)).transpose(1,2)


def MLP(channels: list, do_bn=True, act=nn.GELU):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(act())
    return nn.Sequential(*layers)


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class Wave_position_encoder(nn.Module):
    def __init__(self, dim: int, x_ratio, m_ratio, act=nn.ReLU):
        super().__init__()
        self.x_proj = nn.Sequential(
            nn.Conv1d(dim, int(x_ratio*dim), 1),
            nn.BatchNorm1d(int(x_ratio*dim)),
            act(),
            nn.Conv1d(int(x_ratio*dim), dim, 1),)

        self.theta_proj = nn.Sequential(
            nn.Conv1d(3, dim // 2, 1),
            nn.BatchNorm1d(dim // 2),
            act(),
            nn.Conv1d(dim // 2, dim, 1),
            nn.BatchNorm1d(dim),
            act(),)

        self.merge = nn.Sequential(
            nn.Conv1d(2 * dim, int(m_ratio*dim), 1),
            nn.BatchNorm1d(int(m_ratio*dim)),
            act(),
            nn.Conv1d(int(m_ratio*dim), dim, 1),)

    def forward(self, desc, kpt):
        x = self.x_proj(desc)  # [B, C, N]
        theta = self.theta_proj(kpt)  # [B, C, N]
        x = torch.cat([x * torch.cos(theta), x * torch.sin(theta)], dim=1)  # [B, 2C, N]
        x = desc + self.merge(x)  # [B, C, N]
        return x


class attention_self(nn.Module):
    def __init__(self, dim: int, attention_dropout=0.):
        super().__init__()
        self.scale = 1 / dim**.5
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, query, key, value):
        scores = torch.einsum('bdhn,bdhm->bhnm', query, key) * self.scale
        prob = F.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        return torch.einsum('bhnm,bdhm->bdhn', prob, value)


class attention_cross(nn.Module):
    def __init__(self, dim: int, attention_dropout=0.):
        super().__init__()
        self.scale = 1 / dim**.5
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, query0, key1, value0, value1):
        scores0to1 = torch.einsum('bdhn,bdhm->bhnm', query0, key1) * self.scale
        scores1to0 = scores0to1.transpose(2, 3)
        prob0to1 = F.softmax(scores0to1, dim=-1)
        prob1to0 = F.softmax(scores1to0, dim=-1)
        prob0to1 = self.dropout(prob0to1)
        prob1to0 = self.dropout(prob1to0)
        return torch.einsum('bhnm,bdhm->bdhn', prob0to1, value1), torch.einsum('bhmn,bdhn->bdhm', prob1to0, value0)


class AttnBlock_ln(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int, attention_dropout=0.0, mlp_ratio=2, act=nn.ReLU):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads

        self.attention_self = attention_self(d_model, attention_dropout=attention_dropout)
        self.attention_cross = attention_cross(d_model, attention_dropout=attention_dropout)

        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)

        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

        self.mlp = nn.Sequential(
            nn.Conv1d(d_model * 3, int(mlp_ratio * d_model), 1),
            TransLN(int(mlp_ratio * d_model)),
            nn.GELU(),
            nn.Conv1d(int(mlp_ratio * d_model), d_model, 1),
        )

    def forward(self, desc0, desc1):
        batch_dim = desc0.size(0)

        query0, key0, value0 = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                                for l, x in zip(self.proj, (desc0, desc0, desc0))]
        query1, key1, value1 = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                                for l, x in zip(self.proj, (desc1, desc1, desc1))]

        x_self0 = self.attention_self(query0, key0, value0)
        x_self1 = self.attention_self(query1, key1, value1)
        x_cross0, x_cross1 = self.attention_cross(query0, key1, value0, value1)

        x_self0 = self.merge(x_self0.contiguous().view(batch_dim, self.dim * self.num_heads, -1))
        x_self1 = self.merge(x_self1.contiguous().view(batch_dim, self.dim * self.num_heads, -1))
        x_cross0 = self.merge(x_cross0.contiguous().view(batch_dim, self.dim * self.num_heads, -1))
        x_cross1 = self.merge(x_cross1.contiguous().view(batch_dim, self.dim * self.num_heads, -1))

        desc0 = desc0 + self.mlp(torch.cat([desc0, x_self0, x_cross0], dim=1))
        desc1 = desc1 + self.mlp(torch.cat([desc1, x_self1, x_cross1], dim=1))
        return desc0, desc1

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class group_decoder_sf(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.merge = nn.Conv1d(dim, dim, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        self.mlp = MLP([dim*2, dim*2, dim], do_bn=False)
        nn.init.constant_(self.mlp[-1].bias, 0.0)
        self.norm0 = TransLN(dim)
        self.norm1 = TransLN(dim)
        self.grouping = nn.Sequential(
            nn.Conv1d(dim, 64, 1),
            TransLN(64),
            # nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 16, 1),
            TransLN(16),
            # nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 4, 1),
            TransLN(4),
            # nn.BatchNorm1d(4),
            nn.GELU(),
            nn.Conv1d(4, 2, 1),
            nn.Softmax(1)
        )
        self.mlp_s = Mlp(2, int(0.5*dim), act_layer=nn.GELU)
        self.mlp_c = Mlp(dim, int(4*dim), act_layer=nn.GELU)

    def forward(self, group_token, x):
        b = x.size(0)
        group_token = group_token + self.mlp_s(group_token.transpose(1,2)).transpose(1,2)
        group_token = group_token + self.mlp_c(group_token)
        query, key, value = [l(x).view(b, self.head_dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (x, group_token, group_token))]
        attn = torch.einsum('bdhn,bdhm->bhnm', query, key) * self.scale
        attn = attn.softmax(-1)
        out = torch.einsum('bhnm,bdhm->bdhn', attn, value)
        out = self.merge(out.contiguous().view(b, self.head_dim*self.num_heads, -1))
        x_decode = x + self.norm1(self.mlp(torch.cat([x, self.norm0(out)], dim=1)))
        group_sf = self.grouping(x_decode)
        return group_sf[:, 0, :]

