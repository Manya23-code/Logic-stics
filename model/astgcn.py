"""
astgcn.py — Attention-based Spatial-Temporal Graph Convolutional Network.

Architecture:
  - Spatial Attention: learns dynamic node importance
  - Temporal Attention: learns dynamic timestep importance
  - Chebyshev Graph Convolution + Temporal Convolution
  - Stacked ST-Blocks with residual connections
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_laplacian(adj: np.ndarray) -> np.ndarray:
    """Compute the scaled graph Laplacian ~L = 2L/λ_max - I."""
    N = adj.shape[0]
    d = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(d + 1e-8, -0.5)
    d_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(N) - d_inv_sqrt @ adj @ d_inv_sqrt

    try:
        from scipy.sparse.linalg import eigsh
        lambda_max = eigsh(L, k=1, which="LM", return_eigenvectors=False)[0]
    except Exception:
        lambda_max = 2.0

    return (2.0 * L / lambda_max) - np.eye(N)


def cheb_polynomials(L: np.ndarray, K: int) -> list[torch.Tensor]:
    """Compute Chebyshev polynomials T_0 … T_{K-1} of the scaled Laplacian."""
    N = L.shape[0]
    L = L.astype(np.float32)  # ensure float32 throughout
    polys = [np.eye(N, dtype=np.float32)]
    if K > 1:
        polys.append(L.copy())
    for k in range(2, K):
        polys.append((2.0 * L @ polys[-1] - polys[-2]).astype(np.float32))
    return [torch.from_numpy(p.astype(np.float32)).float() for p in polys]


# ── Attention Blocks ─────────────────────────────────────────────────────────
class SpatialAttention(nn.Module):
    """Learns attention scores across nodes for each time step."""

    def __init__(self, num_nodes: int, num_features: int, num_timesteps: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(num_timesteps, dtype=torch.float32))
        self.W2 = nn.Parameter(torch.empty(num_features, num_timesteps, dtype=torch.float32))
        self.W3 = nn.Parameter(torch.empty(num_features, dtype=torch.float32))
        self.bs = nn.Parameter(torch.empty(num_nodes, num_nodes, dtype=torch.float32))
        self.Vs = nn.Parameter(torch.empty(num_nodes, num_nodes, dtype=torch.float32))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            nn.init.uniform_(p, -0.1, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, F, T) → attention (B, N, N)"""
        # (B, N, F, T) × W1(T,) → (B, N, F)
        lhs = torch.einsum("bnft,t->bnf", x, self.W1)
        # (B, N, F) × W2(F,T) → (B, N, T)
        lhs = lhs @ self.W2

        # (B, N, F, T) × W3(F,) → (B, N, T)
        rhs = torch.einsum("bnft,f->bnt", x, self.W3)

        # (B, N, T) × (B, T, N) → (B, N, N)
        product = torch.bmm(lhs, rhs.transpose(1, 2))

        S = self.Vs @ torch.sigmoid(product + self.bs)
        # Normalize
        S = F.softmax(S, dim=-1)
        return S


class TemporalAttention(nn.Module):
    """Learns attention scores across timesteps for each node."""

    def __init__(self, num_nodes: int, num_features: int, num_timesteps: int):
        super().__init__()
        self.U1 = nn.Parameter(torch.empty(num_nodes, dtype=torch.float32))
        self.U2 = nn.Parameter(torch.empty(num_features, num_nodes, dtype=torch.float32))
        self.U3 = nn.Parameter(torch.empty(num_features, dtype=torch.float32))
        self.be = nn.Parameter(torch.empty(1, num_timesteps, num_timesteps, dtype=torch.float32))
        self.Ve = nn.Parameter(torch.empty(num_timesteps, num_timesteps, dtype=torch.float32))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            nn.init.uniform_(p, -0.1, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, F, T) → attention (B, T, T)"""
        B, N, F_dim, T = x.shape

        # lhs: (B, T, F, N) × (N,) → (B, T, F) × (F, N) → (B, T, N)
        lhs = torch.matmul(x.permute(0, 3, 2, 1), self.U1)  # (B, T, F)
        lhs = torch.matmul(lhs, self.U2)                      # (B, T, N)

        # rhs: (F,) × (B*N, F, T) → (B*N, T) → (B, N, T)
        rhs = torch.matmul(self.U3, x.reshape(-1, F_dim, T))  # (B*N, T)
        rhs = rhs.reshape(B, N, T)                             # (B, N, T)

        # (B, T, N) × (B, N, T) → (B, T, T)
        product = torch.bmm(lhs, rhs)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))
        E = F.softmax(E, dim=-1)
        return E


# ── Chebyshev Graph Convolution ──────────────────────────────────────────────
class ChebConv(nn.Module):
    """Chebyshev spectral graph convolution."""

    def __init__(self, K: int, in_channels: int, out_channels: int):
        super().__init__()
        self.K = K
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Parameter(
            torch.empty(K, in_channels, out_channels, dtype=torch.float32))
        nn.init.xavier_uniform_(self.Theta)

    def forward(self, x: torch.Tensor, cheb_polys: list[torch.Tensor]) -> torch.Tensor:
        """
        x: (B, N, F_in)
        cheb_polys: list of K tensors, each (N, N)
        Returns: (B, N, F_out)
        """
        B, N, F_in = x.shape
        outputs = []
        for k in range(self.K):
            T_k = cheb_polys[k].to(device=x.device, dtype=x.dtype)  # (N, N) — match dtype
            # (N, N) × (B, N, F_in) → (B, N, F_in) — batch matmul
            rhs = torch.einsum("mn,bnf->bmf", T_k, x)
            outputs.append(rhs @ self.Theta[k])  # (B, N, F_out)
        return sum(outputs)


# ── Spatio-Temporal Block ────────────────────────────────────────────────────
class STBlock(nn.Module):
    """
    One Spatio-Temporal block:
      SpatialAttention → ChebConv → ReLU →
      TemporalAttention → TemporalConv → LayerNorm + Residual
    """

    def __init__(self, num_nodes: int, in_features: int, out_features: int,
                 num_timesteps: int, K: int):
        super().__init__()
        self.spatial_attn  = SpatialAttention(num_nodes, in_features, num_timesteps)
        self.temporal_attn = TemporalAttention(num_nodes, out_features, num_timesteps)
        self.cheb_conv     = ChebConv(K, in_features, out_features)
        self.temporal_conv = nn.Conv2d(out_features, out_features,
                                       kernel_size=(1, 3), padding=(0, 1))
        self.norm          = nn.LayerNorm(out_features)
        self.residual      = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x: torch.Tensor, cheb_polys: list[torch.Tensor]) -> torch.Tensor:
        """x: (B, N, C, T)"""
        B, N, C, T = x.shape  # C not F — avoid shadowing torch.nn.functional

        # Spatial attention + graph conv
        S_attn = self.spatial_attn(x)  # (B, N, N)
        spatial_out = []
        for t in range(T):
            x_t = x[:, :, :, t]  # (B, N, C)
            # Apply spatial attention
            x_t = torch.bmm(S_attn, x_t)  # (B, N, C)
            x_t = self.cheb_conv(x_t, cheb_polys)  # (B, N, C_out)
            x_t = F.relu(x_t)
            spatial_out.append(x_t)
        # Stack → (B, N, C_out, T)
        h = torch.stack(spatial_out, dim=-1)

        # Temporal attention
        E_attn = self.temporal_attn(h)  # (B, T, T)
        h = torch.einsum("bnft,bts->bnfs", h, E_attn)

        # Temporal convolution: needs (B, C_out, N, T) layout
        h_perm = h.permute(0, 2, 1, 3)  # (B, C_out, N, T)
        h_perm = self.temporal_conv(h_perm)
        h = h_perm.permute(0, 2, 1, 3)  # back to (B, N, C_out, T)

        # Residual + LayerNorm over feature dim
        res = self.residual(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)  # (B, N, C_out, T)
        combined = (h + res).permute(0, 1, 3, 2)  # (B, N, T, C_out)
        out = self.norm(combined).permute(0, 1, 3, 2)  # (B, N, C_out, T)
        return out


# ── Full ASTGCN Model ────────────────────────────────────────────────────────
class ASTGCN(nn.Module):
    """
    Attention-based Spatial-Temporal Graph Convolutional Network.

    Input:  (B, T_in, N, F)   — batch of historical traffic
    Output: (B, T_out, N, 1)  — predicted future traffic (speed)
    """

    def __init__(self,
                 num_nodes: int,
                 in_features: int = 1,
                 hidden_dim: int = 64,
                 out_features: int = 1,
                 num_timesteps_in: int = 12,
                 num_timesteps_out: int = 12,
                 K: int = 3,
                 num_blocks: int = 2):
        super().__init__()
        self.num_nodes = num_nodes
        self.T_in  = num_timesteps_in
        self.T_out = num_timesteps_out

        # Build ST blocks
        blocks = []
        for i in range(num_blocks):
            f_in = in_features if i == 0 else hidden_dim
            blocks.append(STBlock(num_nodes, f_in, hidden_dim, num_timesteps_in, K))
        self.st_blocks = nn.ModuleList(blocks)

        # Output projection: (B, N, hidden, T_in) → (B, T_out, N, 1)
        self.output_conv = nn.Conv2d(hidden_dim, out_features,
                                      kernel_size=(1, num_timesteps_in))
        self.fc_out = nn.Linear(1, num_timesteps_out)

    def forward(self, x: torch.Tensor, cheb_polys: list[torch.Tensor]) -> torch.Tensor:
        """
        x: (B, T_in, N, F)
        Returns: (B, T_out, N, 1)
        """
        # Rearrange to (B, N, F, T)
        h = x.permute(0, 2, 3, 1)

        for block in self.st_blocks:
            h = block(h, cheb_polys)

        # (B, N, H, T) → (B, H, N, T) for Conv2d
        h = h.permute(0, 2, 1, 3)
        # Conv over time → (B, 1, N, 1)
        h = self.output_conv(h)
        # (B, 1, N, 1) → (B, N, 1) → (B, N, T_out)
        h = h.squeeze(1).squeeze(-1)  # (B, N)
        h = self.fc_out(h.unsqueeze(-1))  # (B, N, T_out)
        # → (B, T_out, N, 1)
        return h.permute(0, 2, 1).unsqueeze(-1)


def build_model(adj_matrix: np.ndarray,
                num_nodes: int = 225,
                in_features: int = 1,
                hidden_dim: int = 64,
                T_in: int = 12,
                T_out: int = 12,
                K: int = 3,
                num_blocks: int = 2,
                device: str = "cpu") -> tuple:
    """
    Convenience function to build ASTGCN + Chebyshev polynomials.
    Returns (model, cheb_polys).
    """
    L_scaled = scaled_laplacian(adj_matrix)
    polys = cheb_polynomials(L_scaled, K)
    polys = [p.to(device) for p in polys]

    model = ASTGCN(
        num_nodes=num_nodes,
        in_features=in_features,
        hidden_dim=hidden_dim,
        num_timesteps_in=T_in,
        num_timesteps_out=T_out,
        K=K,
        num_blocks=num_blocks,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[ASTGCN] Built model: {num_nodes} nodes, {total_params:,} parameters")
    return model, polys
