"""高层子任务规划器 π_HL（SRT-H 式）。

对应 SRT-H 论文：
  - 输入：内窥镜/腕部相机当前帧 + 间隔 1s 的 K 帧历史 + 中心裁剪（AnyRes 思想）
  - 视觉编码：Swin-T（ImageNet 预训练）
  - Transformer Decoder（6 层 8 头）+ 任务查询嵌入
  - 三个输出头：
      p_t  任务/子任务指令（本项目：7 类缝合阶段）
      c_t  纠错标志（布尔）
      m_t  纠正指令（18 类方向动作）
      g_t  进针几何子目标（进针点热图 + 进针角度）——技术方案的
           "一等条件控制信号"，SRT-H 中没有，由本项目的解剖约束引入
  - 损失：CE × (1 + L1 类距)（Eq.1），三项加权 0.4 / 0.3 / 0.3
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .config import Config
from .utils import DistanceWeightedCE


class HighLevelPolicy(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.num_phases = len(cfg.task.phases)
        self.num_corr = len(cfg.task.corrections)
        dim = cfg.hl.dim

        # Swin-T（ImageNet 预训练，SRT-H 同款骨干；紧凑 token 输出适合时序建模）
        swin = torchvision.models.swin_t(
            weights=torchvision.models.Swin_T_Weights.IMAGENET1K_V1)
        self.backbone = swin.features  # 输出 (B, 768, 7, 7)
        self.proj = nn.Linear(768, dim)

        self.grid = 7  # Swin-T 特征图 7x7
        self.query_embed = nn.Embedding(3 + 2, dim)  # p/c/m 三任务查询 + 进针点/角度
        dec_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=cfg.hl.heads, dim_feedforward=dim * 4,
            dropout=cfg.hl.dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.hl.depth)

        self.phase_head = nn.Linear(dim, self.num_phases)
        self.flag_head = nn.Linear(dim, 2)
        self.corr_head = nn.Linear(dim, self.num_corr)
        # 进针几何子目标：7x7 进针点热图 + 进针角（单位向量 3D）
        self.heat_head = nn.Linear(dim, self.grid * self.grid)
        self.angle_head = nn.Linear(dim, 3)

        self.phase_loss_fn = DistanceWeightedCE(self.num_phases)
        w = cfg.hl
        self.wp, self.wc, self.wm = (
            w.l1_loss_weight, w.flag_loss_weight, w.corr_loss_weight)

    def _encode(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, 1+K, 3, H, W) → tokens (B, (1+K)*49, dim)。"""
        B, T = frames.shape[:2]
        x = frames.reshape(B * T, *frames.shape[2:])
        f = self.backbone(x)                      # (B*T, 768, 7, 7)
        f = f.flatten(2).transpose(1, 2)          # (B*T, 49, 768)
        f = self.proj(f).reshape(B, T * self.grid * self.grid, -1)
        # 正弦位置编码（时序 + 空间展平，SRT-H 用 sinusoidal PE）
        pos = self._sin_pos(f.shape[1], f.shape[2], device=f.device)
        return f + pos[None]

    @staticmethod
    def _sin_pos(n: int, d: int, device) -> torch.Tensor:
        pe = torch.zeros(n, d, device=device)
        pos = torch.arange(n, device=device).float()[:, None]
        div = torch.exp(torch.arange(0, d, 2, device=device).float()
                        * (-torch.log(torch.tensor(10000.0)) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, hl_frames: torch.Tensor) -> dict:
        """hl_frames: (B, 1+K, 3, H, W)。返回各头 logits 与几何子目标。"""
        tokens = self._encode(hl_frames)
        B = tokens.shape[0]
        q = self.query_embed.weight[None].expand(B, -1, -1)  # (B, 5, D)
        out = self.decoder(q, tokens)                        # (B, 5, D)
        heat = self.heat_head(out[:, 3]).reshape(B, self.grid, self.grid)
        angle = F.normalize(self.angle_head(out[:, 4]), dim=-1)
        return dict(
            phase_logits=self.phase_head(out[:, 0]),
            flag_logits=self.flag_head(out[:, 1]),
            corr_logits=self.corr_head(out[:, 2]),
            entry_heat=heat,
            entry_angle=angle,
        )

    def compute_loss(self, hl_frames, phase, flag, correction,
                     entry_heat=None, entry_angle=None) -> dict:
        out = self.forward(hl_frames)
        l_phase = self.phase_loss_fn(out["phase_logits"], phase)
        l_flag = F.cross_entropy(out["flag_logits"], flag)
        l_corr = F.cross_entropy(out["corr_logits"], correction)
        loss = self.wp * l_phase + self.wc * l_flag + self.wm * l_corr
        metrics = dict(loss=loss, phase=l_phase.detach(), flag=l_flag.detach(),
                       corr=l_corr.detach())

        # 几何子目标损失（有标注时）：热图软标签 + 角度余弦
        if entry_heat is not None:
            logp = F.log_softmax(out["entry_heat"].flatten(1), dim=-1)
            l_heat = -(entry_heat.flatten(1) * logp).sum(-1).mean()
            loss = loss + 0.2 * l_heat
            metrics["heat"] = l_heat.detach()
        if entry_angle is not None:
            l_ang = 1 - (out["entry_angle"] * entry_angle).sum(-1).mean()
            loss = loss + 0.2 * l_ang
            metrics["angle"] = l_ang.detach()
        metrics["loss"] = loss
        return metrics

    @torch.no_grad()
    def predict(self, hl_frames: torch.Tensor) -> dict:
        """推理：每 3s 调一次（SRT-H）；返回子任务 id、纠错标志与纠正指令。"""
        out = self.forward(hl_frames)
        return dict(
            phase=out["phase_logits"].argmax(-1),
            flag=out["flag_logits"].argmax(-1),
            correction=out["corr_logits"].argmax(-1),
            entry_heat=out["entry_heat"],
            entry_angle=out["entry_angle"],
        )

    def instruction_text(self, phase_id: int, flag: int, corr_id: int) -> str:
        """将 HL 输出转成 LL 的语言条件（SRT-H Eq.2 的 l_t 选择逻辑）。"""
        from . import PHASES, PHASE_ZH
        if int(flag) == 1:
            return self.cfg.task.corrections[int(corr_id)]
        return PHASE_ZH.get(PHASES[int(phase_id)], PHASES[int(phase_id)])
