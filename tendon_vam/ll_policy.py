"""低层策略 π_LL：SRT-H 式语言条件 ACT（基线版）。

对应 SRT-H 论文：
  - 三路相机（内窥镜左目 + 左右腕部）EfficientNet-B3 编码
  - 语言指令经 DistilBERT → FiLM 调制视觉特征（本项目用指令嵌入表替代
    DistilBERT：子任务/纠正指令是有限闭集，嵌入表更小更稳；有需要可换）
  - Transformer Decoder 输出 k×20 动作块（hybrid-relative）
  - 行为克隆 L1 损失；训练时 7% 概率随机丢弃一路相机

注：这是"无视频模型先验"的基线策略，与 flow_decoder.py（VAM 版动作解码器）
构成消融对比的两个分支。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

from .config import Config
from .utils import FiLM


class LowLevelPolicy(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        dim = cfg.ll.dim
        A = cfg.action.total_dim
        k = cfg.action.chunk
        num_instr = len(cfg.task.phases) + len(cfg.task.corrections)

        weights = torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1
        backbones = []
        for _ in range(3):  # 内窥镜 + 左右腕部
            net = torchvision.models.efficientnet_b3(weights=weights)
            backbones.append(nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1)))
        self.cnn = nn.ModuleList(backbones)
        feat_dim = 1536
        self.feat_proj = nn.Linear(feat_dim, dim)

        # 语言条件：指令嵌入（可替换为 DistilBERT 编码）
        self.instr_embed = nn.Embedding(num_instr, dim)
        self.film = FiLM(dim, dim) if cfg.ll.fiLM else None

        dec_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=cfg.ll.heads, dim_feedforward=dim * 4,
            dropout=0.1, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.ll.depth)
        self.action_queries = nn.Parameter(torch.zeros(k, dim))
        nn.init.trunc_normal_(self.action_queries, std=0.02)
        self.head = nn.Linear(dim, A)
        self.k, self.A = k, A

    def encode_instr(self, phase: torch.Tensor, flag: torch.Tensor,
                     correction: torch.Tensor) -> torch.Tensor:
        """按 SRT-H Eq.2 选择语言指令：c_t=0 用任务指令，c_t=1 用纠正指令。"""
        n_phase = len(self.cfg.task.phases)
        instr = torch.where(
            flag > 0, correction + n_phase, phase.to(correction.dtype))
        return self.instr_embed(instr)  # (B, D)

    def forward(self, left: torch.Tensor, right: torch.Tensor,
                wrist_left: torch.Tensor, wrist_right: torch.Tensor,
                instr_emb: torch.Tensor,
                cam_drop: bool = True) -> torch.Tensor:
        """wrist_left/right: (B,3,H,W) 双腕部相机；left/right: 内窥镜双目。

        用户当前数据是双腕部相机：调用时把 left_endo 传给 left，
        right_endo 传给 right，内窥镜通道可传同帧或置零（见 dataset 说明）。
        """
        B = left.shape[0]
        feats = []
        for i, img in enumerate((left, right, wrist_left, wrist_right)):
            if i == 3 and wrist_right is None:
                continue
            f = self.cnn[min(i, 2)](img).flatten(1)
            # SRT-H：7% 概率丢一路相机，防过度依赖单路
            if cam_drop and self.training and torch.rand(1).item() < self.cfg.ll.cam_dropout:
                f = torch.zeros_like(f)
            feats.append(self.feat_proj(f))
        vis = torch.stack(feats, dim=1)  # (B, n_cam, D)

        if self.film is not None:
            vis = self.film(vis.flatten(0, 1), instr_emb[:, None].expand(
                -1, vis.shape[1], -1).flatten(0, 1)).reshape(vis.shape)

        q = self.action_queries[None].expand(B, -1, -1)
        out = self.decoder(q, vis)  # (B, k, D)
        return self.head(out)       # (B, k, A) hybrid-relative 动作块

    def compute_loss(self, batch, device) -> torch.Tensor:
        left = batch["left"].to(device)
        right = batch["right"].to(device)
        wl = batch.get("wrist_left", batch["left"]).to(device)
        wr = batch.get("wrist_right", batch["right"]).to(device)
        instr = self.encode_instr(batch["phase"].to(device),
                                  batch["flag"].to(device),
                                  batch["correction"].to(device))
        pred = self.forward(left, right, wl, wr, instr)
        target = batch["actions"].to(device)
        # 只在有运动学数据的样本上计算动作损失
        mask = (batch["has_kin"].to(device) > 0).float()[:, None, None]
        l1 = (pred - target).abs() * mask
        return l1.sum() / mask.sum().clamp(min=1) / self.A
