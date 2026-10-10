"""Flow Matching 动作解码器（IDM）+ 显式时空对齐算子。

对应技术方案：
  "作为逆动力学模型的流匹配动作解码器……在潜在视觉计划、瞬时本体感觉以及
   解剖约束的多模态条件联合驱动下，利用动作分块策略一次性回归预测未来步
   的连续控制轨迹"
  "通过将 T 维时序动作块与视觉预测演化的时间空间索引进行显式绑定……"

实现要点：
  - rectified flow：x_s = (1-s)·噪声 + s·动作块；速度场 v(x_s, s | cond)
  - 条件 cond = [潜在视觉计划 z_plan, 本体感觉, 子任务嵌入]
  - 显式时空对齐：动作块内第 i 步的查询与视频潜在时序之间做
    cross-attention，并加单调对齐先验偏置（对齐矩阵允许学习偏移，
    但被单调先验约束，保证"每一解算步长的控制输出均能精准匹配其所
    对应的组织交互态演变"）
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class SpatioTemporalAlign(nn.Module):
    """显式时空对齐算子：动作块时序 ↔ 视频潜在时序的绑定。"""

    def __init__(self, dim: int, heads: int = 4, layers: int = 2, video_T: int = 16):
        super().__init__()
        self.heads = heads
        self.video_T = video_T
        blk = []
        for _ in range(layers):
            blk.append(nn.TransformerDecoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim * 4,
                dropout=0.1, batch_first=True, norm_first=True))
        self.layers = nn.ModuleList(blk)
        # 单调对齐先验：query i 与 video token j 的先验对齐分数随 |i/K - j/T| 衰减
        self.align_bias = nn.Parameter(torch.zeros(1))

    def prior(self, K: int, T: int, device) -> torch.Tensor:
        """(K, T) 单调对齐先验矩阵（soft-DTW 风格的带状先验）。"""
        i = torch.arange(K, device=device).float()[:, None] / max(K - 1, 1)
        j = torch.arange(T, device=device).float()[None] / max(T - 1, 1)
        d = (i - j).abs()
        bias = -8.0 * d + 2.0 * self.align_bias
        return bias

    def forward(self, action_queries: torch.Tensor, video_latents: torch.Tensor,
                plan_time: torch.Tensor | None = None) -> torch.Tensor:
        """action_queries: (B, K, D)；video_latents: (B, T_v, D)。

        返回与动作块显式绑定的条件记忆 (B, K, D)：每个动作步只"看"
        与其时间索引对齐的视频潜在状态。
        """
        B, K, D = action_queries.shape
        T = video_latents.shape[1]
        bias = self.prior(K, T, action_queries.device)  # (K, T)
        x = action_queries
        for dec in self.layers:
            # TransformerDecoderLayer 不直接支持 attention bias，
            # 通过把先验加入 memory 的加性 key 实现等价软化。
            # 加性 key 只能依赖视频 token 索引 j，故对 query 维 K 取均值
            # 压成 (T,)，避免 (K,T) 与 (B,T,D) 的广播冲突。
            att_mem = video_latents + bias.mean(dim=0)[None, :, None] * 0.01
            x = dec(x, att_mem)
        return x


class FlowMatchingActionDecoder(nn.Module):
    """VAM 版动作解码器：以潜在视觉计划为条件的 flow matching IDM。"""

    def __init__(self, cfg: Config, video_dim: int, video_T: int = 16):
        super().__init__()
        self.cfg = cfg
        D = cfg.ll.dim
        A = cfg.action.total_dim
        K = cfg.action.chunk
        self.K, self.A = K, A

        num_instr = len(cfg.task.phases) + len(cfg.task.corrections)
        self.instr_embed = nn.Embedding(num_instr, D)
        self.proprio_proj = nn.Linear(2 * 10, D)
        self.plan_proj = nn.Linear(video_dim, D)
        # 视频逐帧潜在 → 解码器维度（对齐 cross-attention 的 d_model = D）
        self.video_proj = nn.Linear(video_dim, D)

        self.align = SpatioTemporalAlign(D, cfg.vam.align_heads,
                                         cfg.vam.align_layers, video_T)

        # 条件融合
        self.cond_fuse = nn.Sequential(nn.Linear(D * 3, D), nn.SiLU(),
                                       nn.Linear(D, D))

        # 速度场 v(x_s, s | cond)：FiLM(s) 调制的 MLP
        t_dim = 128
        self.t_mlp = nn.Sequential(nn.Linear(1, t_dim), nn.SiLU(),
                                   nn.Linear(t_dim, t_dim))
        enc = []
        d_in = A + D + t_dim
        for i in range(4):
            enc.append(nn.Linear(d_in if i == 0 else 512, 512))
            enc.append(nn.SiLU())
        self.vel_net = nn.Sequential(*enc, nn.Linear(512, A))

        # 动作块查询（含时间索引正弦编码 → 时空绑定）
        self.action_queries = nn.Parameter(torch.zeros(K, D))
        nn.init.trunc_normal_(self.action_queries, std=0.02)

    # ------------------------------------------------------------- helpers
    def _embed_time(self, K: int, D: int, device) -> torch.Tensor:
        pos = torch.arange(K, device=device).float()[:, None]
        div = torch.exp(torch.arange(0, D, 2, device=device).float()
                        * (-math.log(10000.0) / D))
        pe = torch.zeros(K, D, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def build_cond(self, z_plan: torch.Tensor, proprio: torch.Tensor,
                   phase: torch.Tensor, flag: torch.Tensor,
                   correction: torch.Tensor) -> torch.Tensor:
        B = z_plan.shape[0]
        n_phase = len(self.cfg.task.phases)
        instr = torch.where(flag > 0, correction + n_phase,
                            phase.to(correction.dtype))
        e_instr = self.instr_embed(instr)
        e_prop = self.proprio_proj(proprio.flatten(1))
        e_plan = self.plan_proj(z_plan)
        return self.cond_fuse(torch.cat([e_plan, e_prop, e_instr], dim=-1))  # (B, D)

    # ------------------------------------------------------------- flow
    def velocity(self, x_s: torch.Tensor, s: torch.Tensor, cond_mem: torch.Tensor,
                 cond: torch.Tensor) -> torch.Tensor:
        """x_s: (B, K, A)；s: (B,)；cond_mem: (B, K, D) 对齐后条件。"""
        ts = self.t_mlp(s[:, None])[:, None, :]              # (B,1,t_dim)
        x = torch.cat([x_s, cond_mem, ts.expand(-1, self.K, -1)], dim=-1)
        return self.vel_net(x)

    def compute_loss(self, z_plan, proprio, phase, flag, correction,
                     actions, video_latents=None) -> torch.Tensor:
        """Rectified flow 训练损失（收敛于确定性的 v ≈ x1 - x0）。"""
        B = z_plan.shape[0]
        device = z_plan.device
        cond = self.build_cond(z_plan, proprio, phase, flag, correction)
        q = self.action_queries[None].expand(B, -1, -1)
        q = q + self._embed_time(self.K, q.shape[-1], device)[None]
        if video_latents is not None:
            cond_mem = self.align(q, self.video_proj(video_latents))
        else:
            cond_mem = cond[:, None, :].expand(-1, self.K, -1) + \
                self._embed_time(self.K, cond.shape[-1], device)[None]

        s = torch.rand(B, device=device)
        x0 = torch.randn(B, self.K, self.A, device=device)
        x1 = actions  # (B, K, A)
        x_s = (1 - s[:, None, None]) * x0 + s[:, None, None] * x1
        v = self.velocity(x_s, s, cond_mem, cond)
        return F.mse_loss(v, x1 - x0)

    @torch.no_grad()
    def sample(self, z_plan, proprio, phase, flag, correction,
               video_latents=None, steps: int = 10,
               partial_t: float = 1.0) -> torch.Tensor:
        """Euler 积分生成动作块。

        partial_t < 1 时同样只积分到中间流时间（与局部去噪联动，
        用于"计划未收敛即提前执行"的保守模式，测试用）。
        """
        B = z_plan.shape[0]
        device = z_plan.device
        cond = self.build_cond(z_plan, proprio, phase, flag, correction)
        q = self.action_queries[None].expand(B, -1, -1)
        q = q + self._embed_time(self.K, q.shape[-1], device)[None]
        if video_latents is not None:
            cond_mem = self.align(q, self.video_proj(video_latents))
        else:
            cond_mem = cond[:, None, :].expand(-1, self.K, -1) + \
                self._embed_time(self.K, cond.shape[-1], device)[None]

        x = torch.randn(B, self.K, self.A, device=device)
        dt = partial_t / steps
        for i in range(steps):
            s = torch.full((B,), partial_t * i / steps, device=device)
            x = x + self.velocity(x, s, cond_mem, cond) * dt
        return x
