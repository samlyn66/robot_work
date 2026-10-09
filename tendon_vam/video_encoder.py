"""共享视觉编码器 + 局部去噪（Partial Denoising）潜在视觉计划。

对应技术方案：
  "视频生成流并不执行全时域的完全去噪，而是前向演化至中间流时间处即行中止。
   此时提取的潜在中间状态表征即被定义为'潜在视觉计划'……在高维流形空间中
   深度编码了组织形变场与器械运动轨迹的未来瞬时动力学趋势。"

工程实现：完整视频扩散模型在 18 条视频的小数据上不可行，因此采用
"latent rectified flow"——在自监督视觉编码器的时序潜在空间中训练一个
条件流预测器，推理时只积分到中间流时间 s = partial_denoise_t (<1)，
得到潜在视觉计划 z_plan，供 Flow Matching 动作解码器条件使用。

可选升级点：将 `visual_backbone` 换成 VideoMAE / V-JEPA 预训练权重
（transformers 库），即获得"手术视频模型动力学先验"。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class FrameEncoder(nn.Module):
    """单帧视觉编码（ImageNet 预训练 CNN → 投影到 latent_dim）。"""

    def __init__(self, backbone: str = "efficientnet_b0", latent_dim: int = 256):
        super().__init__()
        if backbone == "efficientnet_b0":
            weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
            net = torchvision.models.efficientnet_b0(weights=weights)
            feat_dim = 1280
        elif backbone == "efficientnet_b3":
            weights = torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1
            net = torchvision.models.efficientnet_b3(weights=weights)
            feat_dim = 1536
        else:
            raise ValueError(f"unknown backbone {backbone}")
        self.cnn = net.features  # 预训练权重保留（可选冻结）
        self.proj = nn.Linear(feat_dim, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, 3, H, W)
        f = self.cnn(x).mean(dim=(2, 3))
        return self.proj(f)


class TemporalEncoder(nn.Module):
    """时序因果 Transformer：对连续视觉特征谱系建模（技术方案）。"""

    def __init__(self, dim: int, depth: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.pos = nn.Parameter(torch.zeros(1, 64, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) —— 时间因果（只看过去）
        B, T, D = x.shape
        x = x + self.pos[:, :T]
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        return self.enc(x, mask=mask)


class LatentPlanPredictor(nn.Module):
    """潜在视觉计划的 rectified flow 预测器（局部去噪）。

    训练：给定当前潜在状态 z0 与未来潜在目标 z1，学习速度场
          v(z_s, s | cond) ≈ z1 - z0 （rectified flow）。
    推理：从噪声出发只积分到 s = partial_denoise_t < 1，即"前向演化至
          中间流时间处即行中止"，输出潜在视觉计划。
    """

    def __init__(self, latent_dim: int, hidden: int = 512, depth: int = 4):
        super().__init__()
        self.t_mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        layers = []
        for i in range(depth):
            layers += [nn.Linear(latent_dim + hidden if i == 0 else hidden, hidden),
                       nn.SiLU()]
        self.net = nn.Sequential(*layers, nn.Linear(hidden, latent_dim))
        self.latent_dim = latent_dim

    def velocity(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # z: (B, D), s: (B,) in [0,1]
        ts = self.t_mlp(s[:, None])
        return self.net(torch.cat([z, ts], dim=-1))

    def loss(self, z_ctx: torch.Tensor, z_future: torch.Tensor) -> torch.Tensor:
        B = z_ctx.shape[0]
        s = torch.rand(B, device=z_ctx.device)
        noise = torch.randn_like(z_ctx)
        z_s = (1 - s[:, None]) * noise + s[:, None] * z_future
        v = self.velocity(z_s, s)
        return F.mse_loss(v, z_future - noise)

    @torch.no_grad()
    def sample_partial(self, cond: torch.Tensor, s_end: float, steps: int = 8,
                       generator: torch.Generator | None = None) -> torch.Tensor:
        """局部去噪：只积分到 s_end < 1（cond 作为初始潜在状态的引导偏置）。"""
        z = torch.randn(cond.shape[0], self.latent_dim,
                        device=cond.device, generator=generator)
        for i in range(steps):
            s = torch.full((cond.shape[0],), s_end * i / steps,
                           device=cond.device)
            z = z + self.velocity(z, s) * (s_end / steps)
        return z + cond  # 以当前潜在状态为基底（残差式计划）


class VideoBackbone(nn.Module):
    """共享视觉编码器 + 时序建模 + 潜在视觉计划头。"""

    def __init__(self, backbone: str, latent_dim: int, partial_t: float = 0.6):
        super().__init__()
        self.frame_enc = FrameEncoder(backbone, latent_dim)
        self.temporal = TemporalEncoder(latent_dim)
        self.plan = LatentPlanPredictor(latent_dim)
        self.partial_t = partial_t

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 3, H, W) 左腕相机序列（控制频率）→ (B, T, D) 时序特征。"""
        B, T = x.shape[:2]
        f = self.frame_enc(x.reshape(B * T, *x.shape[2:])).reshape(B, T, -1)
        return self.temporal(f)

    def forward(self, x: torch.Tensor, horizon: int = 1) -> torch.Tensor:
        """返回潜在视觉计划 z_plan: (B, D)。

        训练阶段调用 `.encode` + `.plan.loss`；
        推理阶段调用本函数做局部去噪得到 z_plan。
        """
        feats = self.encode(x)
        ctx = feats[:, -1]
        return self.plan.sample_partial(ctx, self.partial_t)

    def plan_loss(self, x: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        """x: (B, T_past, 3,H,W) 过去帧；x_future: (B, T_f, 3,H,W) 未来帧。"""
        past = self.encode(x)[:, -1]
        with torch.no_grad():
            fut = self.encode(x_future)[:, -1].detach()
        return self.plan.loss(past, fut)
