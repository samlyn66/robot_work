"""双腕部相机配对数据集。

数据约定（每个 recording 一个文件夹）::

    data/recording_YYYYMMDD_HHMMSS/
        left_endo_frames/  img_000000.jpg   # extract_frames.py 产出
        right_endo_frames/ img_000000.jpg
        kinematics.csv     (可选) t, l_px,l_py,l_pz,l_qw,l_qx,l_qy,l_qz,l_grip,
                                  r_px,r_py,r_pz,r_qw,r_qx,r_qy,r_qz,r_grip,
                                  endo_px,endo_py,endo_pz
        subtasks.json      (可选) 人工子任务标注

- HL 训练：只需视频帧 + subtasks.json（纯视频模仿）。
- LL/VAM 训练：还需要 kinematics.csv（生成 hybrid-relative 动作块）。
"""
from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Config
from .utils import make_hybrid_relative_actions, quat_to_6d


@dataclass
class Recording:
    name: str
    root: str
    left_frames: List[str]
    right_frames: List[str]
    fps: float                       # 抽帧后的帧率
    subtasks: Optional[List[dict]]   # [{"t_start","t_end","phase"}]
    kinematics: Optional[Dict[str, np.ndarray]]

    @property
    def num_frames(self) -> int:
        return min(len(self.left_frames), len(self.right_frames))

    def phase_at(self, t: float) -> str:
        if not self.subtasks:
            return "idle"
        for seg in self.subtasks:
            if seg["t_start"] <= t < seg["t_end"]:
                return seg["phase"]
        return "idle"


def _list_frames(d: str) -> List[str]:
    if not os.path.isdir(d):
        return []
    files = sorted(
        f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png", ".jpeg"))
    )
    return [os.path.join(d, f) for f in files]


def _load_kinematics(path: str) -> Optional[Dict[str, np.ndarray]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    out: Dict[str, List[float]] = {k: [] for k in rows[0]}
    for r in rows:
        for k, v in r.items():
            try:
                out[k].append(float(v))
            except (TypeError, ValueError):
                out[k].append(0.0)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def load_recordings(root: str, fps: float) -> List[Recording]:
    """扫描 root 下全部 recording，加载抽帧、标注与运动学。"""
    recs = []
    if not os.path.isdir(root):
        return recs
    for name in sorted(os.listdir(root)):
        rroot = os.path.join(root, name)
        if not os.path.isdir(rroot):
            continue
        left = _list_frames(os.path.join(rroot, "left_endo_frames"))
        right = _list_frames(os.path.join(rroot, "right_endo_frames"))
        if not left or not right:
            continue
        subtasks = None
        sj = os.path.join(rroot, "subtasks.json")
        if os.path.exists(sj):
            with open(sj, "r", encoding="utf-8") as f:
                subtasks = json.load(f).get("events")
        kin = _load_kinematics(os.path.join(rroot, "kinematics.csv"))
        recs.append(Recording(name, rroot, left, right, fps, subtasks, kin))
    return recs


def split_recordings(recs: List[Recording], cfg: Config):
    """按配置划分训练/验证（以录像为单位，避免同录像泄漏）。"""
    names = {r.name for r in recs}
    val = set(n for n in cfg.data.val_recordings if n in names)
    if not val:
        rng = random.Random(cfg.data.seed)
        pool = sorted(names)
        rng.shuffle(pool)
        val = set(pool[: max(1, int(0.2 * len(pool)))]) if len(pool) > 1 else set()
    train = [r for r in recs if r.name not in val]
    valr = [r for r in recs if r.name in val]
    return train, valr


class PairedVideoDataset(Dataset):
    """双相机配对帧数据集，同时服务 HL / LL / VAM 训练。

    返回 dict：
      left, right            (3, H, W) float [0,1]
      hl_frames              (1+K, 3, H, W) 当前帧 + 历史帧（HL）
      phase                  int 子任务类别（HL 目标）
      correction             int 纠正指令类别（recovery 片段用，默认 0）
      flag                   int 纠错标志
      actions                (chunk, A) hybrid-relative 动作块（有运动学时）
      proprio                (2, 10) 当前双臂本体感觉（位置3+rot6+grip）
      has_kin                bool
    """

    def __init__(self, recs: List[Recording], cfg: Config, mode: str = "train",
                 need_actions: bool = False):
        self.cfg = cfg
        self.mode = mode
        self.need_actions = need_actions
        self.phase_to_idx = {p: i for i, p in enumerate(cfg.task.phases)}
        self.samples: List[tuple] = []
        for rec in recs:
            n = rec.num_frames
            if n < 2:
                continue
            max_t = n / rec.fps
            # 起始帧要给 HL 历史留空间；结尾给动作块留空间
            t0 = cfg.data.hl_history * cfg.data.hl_history_stride_sec
            ts = np.arange(t0, max_t, 1.0 / rec.fps)
            for i, t in enumerate(ts):
                self.samples.append((rec, float(t)))
        # recovery 片段：phase 标注含 "recover_" 前缀时 flag=1（SRT-H 机制）

    def __len__(self):
        return len(self.samples)

    # -------------------------------------------------------------- helpers
    def _load_img(self, path: str, size: int) -> torch.Tensor:
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((size, size, 3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if self.mode == "train":
            # 轻量增强：亮度/对比度抖动（Albumentations 的简易版）
            x = x * (0.9 + 0.2 * torch.rand(3, 1, 1)) + 0.05 * torch.randn(1)
            x = x.clamp(0, 1)
        return x

    def _center_crop(self, x: torch.Tensor, w_frac: float, h_frac: float) -> torch.Tensor:
        """SRT-H HL 中心裁剪：内 50% 宽、下 80% 高。"""
        _, H, W = x.shape
        w0, w1 = int(W * (1 - w_frac) / 2), int(W * (1 + w_frac) / 2)
        h0, h1 = int(H * (1 - h_frac)), H
        return x[:, h0:h1, w0:w1]

    def __getitem__(self, idx: int):
        rec, t = self.samples[idx]
        c = self.cfg
        fi = min(int(t * rec.fps), rec.num_frames - 1)
        left = self._load_img(rec.left_frames[fi], c.data.hl_frame_size)
        right = self._load_img(rec.right_frames[fi], c.data.hl_frame_size)

        # ---- HL 历史：当前帧 + 间隔 1s 的 K 帧（SRT-H）
        hist = [left]
        stride = int(round(c.data.hl_history_stride_sec * rec.fps))
        for k in range(1, c.data.hl_history + 1):
            j = max(fi - k * stride, 0)
            hist.append(self._load_img(rec.left_frames[j], c.data.hl_frame_size))
        hl_frames = torch.stack(hist)  # (1+K, 3, H, W)

        # ---- 子任务标签（predict offset：0.5s 未来的阶段，SRT-H 技巧）
        phase = rec.phase_at(t + 0.5)
        if phase.startswith("recover_"):
            flag, base_phase = 1, phase[len("recover_"):]
        else:
            flag, base_phase = 0, phase
        phase_idx = self.phase_to_idx.get(base_phase, 0)
        correction = 0  # 有 recovery 标注时由标注文件给出方向类别

        out = dict(
            left=left,
            right=right,
            hl_frames=hl_frames,
            phase=torch.tensor(phase_idx, dtype=torch.long),
            correction=torch.tensor(correction, dtype=torch.long),
            flag=torch.tensor(flag, dtype=torch.long),
            t=torch.tensor(t, dtype=torch.float32),
            has_kin=torch.tensor(0 if rec.kinematics is None else 1),
        )

        # ---- 运动学 → hybrid-relative 动作块 + 本体感觉
        if rec.kinematics is not None:
            kin = rec.kinematics
            kt = kin.get("t")
            if kt is not None and len(kt) > 1:
                # 最近邻对齐到抽帧时间轴
                def at(key: str, time: float) -> float:
                    j = int(np.searchsorted(kt, time).clip(0, len(kt) - 1))
                    return float(kin[key][j])

                T = c.action.chunk
                times = t + np.arange(T + 1) / rec.fps
                arms = {}
                for arm in ("l", "r"):
                    pos = np.stack(
                        [[at(f"{arm}_px", tt), at(f"{arm}_py", tt), at(f"{arm}_pz", tt)]
                         for tt in times])
                    quat = np.stack(
                        [[at(f"{arm}_qw", tt), at(f"{arm}_qx", tt), at(f"{arm}_qy", tt),
                          at(f"{arm}_qz", tt)] for tt in times])
                    grip = np.array([at(f"{arm}_grip", tt) for tt in times])
                    arms[arm] = (
                        make_hybrid_relative_actions(
                            pos, quat_to_6d(quat),
                            np.stack([[at("endo_px", tt), at("endo_py", tt),
                                       at("endo_pz", tt)] for tt in times]), grip),
                        np.concatenate([pos[0], quat_to_6d(quat)[0], grip[:1]]),
                    )
                # (chunk, 2*10)：两臂动作块拼接
                act = np.concatenate(
                    [arms["l"][0][:T], arms["r"][0][:T]], axis=-1)
                proprio = np.stack([arms["l"][1], arms["r"][1]])  # (2, 10)
                out["actions"] = torch.from_numpy(act.astype(np.float32))
                out["proprio"] = torch.from_numpy(proprio.astype(np.float32))
                out["has_kin"] = torch.tensor(1)
        if "actions" not in out:
            A = c.action.total_dim
            out["actions"] = torch.zeros(c.action.chunk, A)
            out["proprio"] = torch.zeros(c.action.num_arms, 10)
        return out
