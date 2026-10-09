"""配置加载与全局常量。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import yaml


@dataclass
class DataCfg:
    root: str = "data"
    frame_rate: float = 10.0
    hl_history: int = 4
    hl_history_stride_sec: float = 1.0
    hl_frame_size: int = 224
    ll_frame_size: int = 224
    val_recordings: List[str] = field(default_factory=list)
    seed: int = 42


@dataclass
class TaskCfg:
    phases: List[str] = field(default_factory=lambda: list(PHASES_DEFAULT))
    corrections: List[str] = field(default_factory=list)


PHASES_DEFAULT = ["idle", "grasp_needle", "insert", "exit", "pull", "knot", "cut"]


@dataclass
class ActionCfg:
    dim_per_arm: int = 10
    num_arms: int = 2
    chunk: int = 20
    use_6d_rotation: bool = True

    @property
    def total_dim(self) -> int:
        return self.dim_per_arm * self.num_arms


@dataclass
class HLCfg:
    dim: int = 384
    depth: int = 6
    heads: int = 8
    center_crop: List[float] = field(default_factory=lambda: [0.5, 0.8])
    l1_loss_weight: float = 0.4
    flag_loss_weight: float = 0.3
    corr_loss_weight: float = 0.3
    predict_offset_sec: float = 0.5
    dropout: float = 0.1


@dataclass
class LLCfg:
    backbone: str = "efficientnet_b3"
    dim: int = 512
    depth: int = 6
    heads: int = 8
    cam_dropout: float = 0.07
    fiLM: bool = True


@dataclass
class VAMCfg:
    video_backbone: str = "efficientnet_b0"
    latent_dim: int = 256
    partial_denoise_t: float = 0.6
    flow_steps: int = 10
    align_heads: int = 4
    align_layers: int = 2
    ema: float = 0.999


@dataclass
class TrainCfg:
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.05
    epochs_hl: int = 60
    epochs_ll: int = 300
    epochs_vam: int = 300
    warmup_epochs: int = 3
    amp: bool = True
    num_workers: int = 4
    out_dir: str = "runs"


@dataclass
class Config:
    data: DataCfg = field(default_factory=DataCfg)
    task: TaskCfg = field(default_factory=TaskCfg)
    action: ActionCfg = field(default_factory=ActionCfg)
    hl: HLCfg = field(default_factory=HLCfg)
    ll: LLCfg = field(default_factory=LLCfg)
    vam: VAMCfg = field(default_factory=VAMCfg)
    train: TrainCfg = field(default_factory=TrainCfg)


def load_config(path: str) -> Config:
    cfg = Config()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        for section, cls in [
            ("data", DataCfg),
            ("task", TaskCfg),
            ("action", ActionCfg),
            ("hl", HLCfg),
            ("ll", LLCfg),
            ("vam", VAMCfg),
            ("train", TrainCfg),
        ]:
            if section in raw:
                cur = getattr(cfg, section)
                for k, v in raw[section].items():
                    if hasattr(cur, k):
                        setattr(cur, k, v)
    if not cfg.task.corrections:
        from . import PHASES
        cfg.task.phases = cfg.task.phases or list(PHASES)
        cfg.task.corrections = [
            "close_left_gripper", "close_right_gripper",
            "open_left_gripper", "open_right_gripper",
            "move_left_left", "move_left_right", "move_left_towards",
            "move_left_away", "move_left_up", "move_left_down",
            "move_right_left", "move_right_right", "move_right_towards",
            "move_right_away", "move_right_up", "move_right_down",
            "close_both_grippers", "open_both_grippers",
        ]
    return cfg
