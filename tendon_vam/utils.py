"""通用工具：6D 旋转表示、hybrid-relative 动作、FiLM、损失等。"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- 6D rotation
def rotation_matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """R: (..., 3, 3) -> (..., 6)，取旋转矩阵前两列（Zhou et al. CVPR 2019）。"""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def quat_to_6d(q: np.ndarray) -> np.ndarray:
    """四元数 (w,x,y,z) -> 6D 表示。"""
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    w, x, y, z = [q[..., i] for i in range(4)]
    R = np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))
    return rotation_matrix_to_6d(R)


def sixd_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """(N, 6) -> (N, 3, 3)，用于部署时恢复完整旋转。"""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


# ------------------------------------------------- hybrid-relative 动作（SRT）
def make_hybrid_relative_actions(
    ee_pos: np.ndarray, ee_rot6d: np.ndarray, endo_pos: np.ndarray, gripper: np.ndarray
) -> np.ndarray:
    """将原始运动学转成 hybrid-relative 动作序列（SRT 论文的动作表示）。

    ee_pos: (T, 3)      末端执行器位置（mm）
    ee_rot6d: (T, 6)    末端执行器旋转 6D 表示
    endo_pos: (T, 3)    内窥镜尖端位置（作为平移参考系）
    gripper: (T,)       夹持开合 [0,1]

    返回 (T, 10)：平移增量（相对内窥镜尖端）3 + 旋转增量 6 + 夹爪 1。
    """
    rel_pos = ee_pos - endo_pos  # 两者均为 (T, 3)：末端位置相对内窥镜尖端
    d_pos = np.diff(rel_pos, axis=0, prepend=rel_pos[:1])
    d_rot = np.diff(ee_rot6d, axis=0, prepend=ee_rot6d[:1])
    return np.concatenate([d_pos, d_rot, gripper[:, None]], axis=-1)


# ---------------------------------------------------------------- FiLM 调制
class FiLM(nn.Module):
    """feature-wise linear modulation：用语言嵌入调制视觉特征（SRT-H LL）。"""

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feat_dim)
        self.beta = nn.Linear(cond_dim, feat_dim)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # feat: (B, F), cond: (B, C)
        return self.gamma(cond) * feat + self.beta(cond)


# ---------------------------------------------------------------- 损失
class DistanceWeightedCE(nn.Module):
    """SRT-H Eq.1：CE 损失 × 预测类与真值类的 L1 距离。

    用于区分"视觉相似但时间上相距很远"的子任务（如两次进针）。
    """

    def __init__(self, num_classes: int, reduction: str = "mean"):
        super().__init__()
        idx = torch.arange(num_classes).float()
        self.register_buffer("dist", (idx[:, None] - idx[None, :]).abs())
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        pred = logits.argmax(-1)
        w = self.dist[pred, target]
        out = ce * (1.0 + w)
        if self.reduction == "mean":
            return out.mean()
        return out


# ---------------------------------------------------------------- 其它
def cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1 + np.cos(np.pi * t))


class EMA:
    """权重指数滑动平均（VAM 训练稳定器）。"""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model: nn.Module):
        model.load_state_dict({k: v for k, v in self.shadow.items()}, strict=True)
