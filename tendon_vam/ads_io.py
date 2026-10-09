"""ADS 原始运动学数据读取与对齐。

数据来源（无官方字段说明，列映射由 scripts/inspect_ads.py 数值分析确定）::

    recording_xxx/
        ads_data.csv                      # 无表头 CSV，367 列，约 335 Hz
        left_endo_YYYYMMDD_HHMMSS_H.264.mp4   # 1920x1080 @ 60 fps
        right_endo_YYYYMMDD_HHMMSS_H.264.mp4

关键结论（recording_20260826_172240 实测验证）：
  - CSV 行数 / 视频帧数 ≈ 5.59，且 CSV 时长 ≈ 视频时长（340 s）→ 两者同时起录，
    用「行/帧比例」做时间对齐（无时间戳列时的最稳做法）。
  - 主臂末端位姿：cols 12-14 (xyz) + 15-18 (quat wxyz)，运动范围最大、四元数连续。
  - 副臂末端位姿：cols 167-169 (xyz) + 170-173 (quat wxyz)。
  - 夹爪候选：col 43（主臂）、col 133（副臂），值域随臂而异，转 [0,1] 时用全数据集统计。
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------- 列映射
@dataclass
class ArmColumns:
    pos_cols: List[int]
    quat_cols: List[int]
    gripper_col: int


@dataclass
class AdsColumnMap:
    arm0: ArmColumns
    arm1: ArmColumns
    quat_order: str = "wxyz"          # wxyz 或 xyzw

    @classmethod
    def from_config(cls, cfg) -> "AdsColumnMap":
        a = cfg.ads
        return cls(
            arm0=ArmColumns(list(a.arm0.pos_cols), list(a.arm0.quat_cols),
                            int(a.arm0.gripper_col)),
            arm1=ArmColumns(list(a.arm1.pos_cols), list(a.arm1.quat_cols),
                            int(a.arm1.gripper_col)),
            quat_order=getattr(a, "quat_order", "wxyz"),
        )


# ---------------------------------------------------------------- 读取
def load_ads_csv(path: str, max_rows: Optional[int] = None) -> np.ndarray:
    """读取无表头 ads_data.csv → (N, C) float64。"""
    rows: List[List[float]] = []
    with open(path, "r", newline="", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split(",")])
            if max_rows is not None and i + 1 >= max_rows:
                break
    return np.asarray(rows, dtype=np.float64)


def fix_quaternion_sign(q: np.ndarray) -> np.ndarray:
    """消除四元数双覆盖（q 与 -q 表示同一旋转）造成的符号跳变。"""
    q = q.copy()
    for i in range(1, len(q)):
        if float(np.dot(q[i], q[i - 1])) < 0:
            q[i] = -q[i]
    return q


def reorder_quat(q: np.ndarray, order: str) -> np.ndarray:
    """统一为 (w, x, y, z)。"""
    if order == "wxyz":
        return q
    if order == "xyzw":
        return q[:, [3, 0, 1, 2]]
    raise ValueError(f"未知四元数顺序: {order}")


def normalize_gripper(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """夹爪原始值 → [0,1]（闭合=1 约定，若实测相反在 config 里翻转）。"""
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


# ---------------------------------------------------------------- 对齐
def csv_row_for_time(t: np.ndarray, n_csv: int, video_duration: float) -> np.ndarray:
    """按行/帧比例：视频时刻 t(s) → CSV 行号（假设 CSV 与视频同时起录、同时结束）。"""
    return np.clip((t / max(video_duration, 1e-6)) * (n_csv - 1), 0, n_csv - 1)


@dataclass
class AdsKinematics:
    """转换后的双臂运动学（统一 10/30 Hz 采样，单位与 CSV 原始一致）。"""
    t: np.ndarray                    # (T,) 秒（视频时间轴）
    pos: Dict[str, np.ndarray]       # arm0/arm1: (T, 3)
    quat: Dict[str, np.ndarray]      # arm0/arm1: (T, 4) wxyz，符号连续
    grip: Dict[str, np.ndarray]      # arm0/arm1: (T,) [0,1]


def extract_ads_kinematics(
    csv_path: str,
    cmap: AdsColumnMap,
    video_duration: float,
    out_rate: float = 30.0,
    fix_sign: bool = True,
    gripper_stats: Optional[Dict[str, tuple]] = None,
) -> AdsKinematics:
    """ads_data.csv → 按视频时间轴重采样的双臂运动学。

    gripper_stats: {"arm0": (lo, hi), "arm1": (lo, hi)}，缺省用本段数据 min/max。
    """
    arr = load_ads_csv(csv_path)
    n_csv = len(arr)
    T = max(int(video_duration * out_rate), 2)
    t = np.arange(T) / out_rate
    rows = csv_row_for_time(t, n_csv, video_duration).astype(np.int64)

    pos, quat, grip = {}, {}, {}
    for arm, cols in (("arm0", cmap.arm0), ("arm1", cmap.arm1)):
        p = arr[:, cols.pos_cols][rows]                       # (T, 3)
        q = reorder_quat(arr[:, cols.quat_cols][rows], cmap.quat_order)
        if fix_sign:
            q = fix_quaternion_sign(q)
        g_raw = arr[:, cols.gripper_col][rows]
        if gripper_stats and arm in gripper_stats:
            lo, hi = gripper_stats[arm]
        else:
            lo, hi = float(arr[:, cols.gripper_col].min()), float(arr[:, cols.gripper_col].max())
        pos[arm], quat[arm], grip[arm] = p, q, normalize_gripper(g_raw, lo, hi)
    return AdsKinematics(t=t, pos=pos, quat=quat, grip=grip)


# ---------------------------------------------------------------- 写出 kinematics.csv
KIN_HEADER = [
    "t",
    "l_px", "l_py", "l_pz", "l_qw", "l_qx", "l_qy", "l_qz", "l_grip",
    "r_px", "r_py", "r_pz", "r_qw", "r_qx", "r_qy", "r_qz", "r_grip",
]


def write_kinematics_csv(kin: AdsKinematics, out_path: str) -> None:
    """写成 dataset.py 消费的 kinematics.csv（arm0→l_*，arm1→r_*）。

    注：本数据无内窥镜位姿流（腕部相机方案），不写 endo_* 列；
    dataset.py 检测到缺列时自动退化为「相对自身起点」的动作表示（等价于增量动作）。
    """
    n = len(kin.t)
    mat = np.zeros((n, len(KIN_HEADER)))
    mat[:, 0] = kin.t
    for i, (arm, pre) in enumerate((("arm0", "l"), ("arm1", "r"))):
        c = 1 + i * 8
        mat[:, c:c + 3] = kin.pos[arm]
        mat[:, c + 3:c + 7] = kin.quat[arm]
        mat[:, c + 7] = kin.grip[arm]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(KIN_HEADER)
        for row in mat:
            w.writerow([f"{v:.6f}" for v in row])


def find_video(rec_root: str, pattern: str) -> Optional[str]:
    """按 glob 模式在 recording 目录里找视频（文件名含时间戳，须模糊匹配）。"""
    import glob as _glob
    hits = sorted(_glob.glob(os.path.join(rec_root, pattern)))
    return hits[0] if hits else None


def video_info(path: str) -> Dict[str, float]:
    """视频帧数 / fps / 时长（cv2）。"""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"frames": n, "fps": fps, "duration": n / fps}


def scan_gripper_stats(data_root: str, cmap: AdsColumnMap,
                       csv_name: str = "ads_data.csv") -> Dict[str, tuple]:
    """全数据集扫描夹爪值域（用于统一的 [0,1] 归一化）。"""
    stats = {"arm0": [np.inf, -np.inf], "arm1": [np.inf, -np.inf]}
    for name in sorted(os.listdir(data_root)):
        p = os.path.join(data_root, name, csv_name)
        if not os.path.isfile(p):
            continue
        arr = load_ads_csv(p, max_rows=20000)
        # 均匀抽样后半段，避免只扫到开头静止段
        for arm, cols in (("arm0", cmap.arm0), ("arm1", cmap.arm1)):
            g = arr[:, cols.gripper_col]
            stats[arm][0] = min(stats[arm][0], float(g.min()))
            stats[arm][1] = max(stats[arm][1], float(g.max()))
    return {k: tuple(v) for k, v in stats.items()}
