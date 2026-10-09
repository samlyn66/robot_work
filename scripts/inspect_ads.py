"""ADS 数据列结构自动分析工具（无表头 CSV 的"逆向工程"）。

用法::

    python scripts/inspect_ads.py --rec <recording目录> [--sample 5000]

输出：
  1. 四元数块检测（连续 4 列范数 ≡ 1）；
  2. 各 pose 块运动幅度 / 平滑度 / 四元数符号跳变数（挑出"真实末端位姿流"）；
  3. 离散标志列（疑似夹爪/离合/踏板）；
  4. 与视频对齐的采样率估计（CSV 行数 / 视频帧数 × 视频 fps）；
  5. 建议的列映射 YAML 片段。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tendon_vam.ads_io import find_video, load_ads_csv, video_info


def analyze(rec: str, sample: int) -> dict:
    csvs = [f for f in os.listdir(rec) if f.endswith(".csv")]
    if not csvs:
        return {"error": f"{rec} 下没有 CSV"}
    csv_path = os.path.join(rec, csvs[0])
    arr = load_ads_csv(csv_path)
    n_full = len(arr)
    # 均匀抽样做统计（全量 11 万行也快，但抽样足够）
    if n_full > sample:
        idx = np.linspace(0, n_full - 1, sample).astype(int)
        s = arr[idx]
    else:
        s = arr
    n, c = s.shape
    out = {"csv": csv_path, "rows": n_full, "cols": c, "sampled": len(s)}

    # ---- 1. 四元数块
    quat_blocks = []
    j = 0
    while j < c - 3:
        norms = np.linalg.norm(s[:, j:j + 4], axis=1)
        if np.all((norms > 0.98) & (norms < 1.02)):
            quat_blocks.append((j, j + 3))
            j += 4
        else:
            j += 1
    out["quat_blocks"] = quat_blocks

    # ---- 2. pose 块指标
    blocks = []
    for (qs, qe) in quat_blocks:
        ps = qs - 3
        if ps < 0:
            continue
        pos = s[:, ps:ps + 3]
        q = s[:, qs:qe + 1]
        span = pos.max(0) - pos.min(0)
        dots = np.sum(q[1:] * q[:-1], axis=1)
        blocks.append(dict(
            pos_cols=[ps, ps + 1, ps + 2], quat_cols=[qs, qs + 1, qs + 2, qs + 3],
            pos_span=[round(float(x), 4) for x in span],
            pos_jerk=float(np.abs(np.diff(pos, 2, axis=0)).mean()),
            quat_flips=int((dots < 0).sum()),
            motion_score=round(float(span.sum()), 4),
        ))
    blocks.sort(key=lambda b: -b["motion_score"])
    out["pose_blocks_by_motion"] = blocks

    # ---- 3. 离散列（2-12 个唯一值）与夹爪候选
    disc, grip_cand = [], []
    for j in range(c):
        if s[:, j].std() < 1e-4:
            continue
        u = np.unique(s[:, j])
        if 2 <= len(u) <= 12:
            disc.append(dict(col=j, vals=[round(float(x), 4) for x in u],
                             counts=[int((s[:, j] == x).sum()) for x in u]))
    # 夹爪候选：与 pose 块相邻、连续值、缝合中呈台阶状开合
    for j in range(c):
        x = s[:, j]
        if x.std() < 1e-3 or np.unique(x).size < 100:
            continue
        ch = np.abs(np.diff(x))
        n_big = int((ch > 0.02).sum())
        if 5 <= n_big <= 0.01 * len(s):
            grip_cand.append(dict(col=j, uniq=int(np.unique(x).size),
                                  range=[round(float(x.min()), 3), round(float(x.max()), 3)],
                                  big_changes=n_big))
    out["discrete_flag_cols"] = disc
    out["gripper_candidates"] = grip_cand

    # ---- 4. 视频对齐
    for pat, side in (("*left_endo*.mp4", "left"), ("*right_endo*.mp4", "right")):
        v = find_video(rec, pat)
        if v:
            vi = video_info(v)
            out[f"video_{side}"] = dict(file=os.path.basename(v), **vi,
                                        rows_per_frame=round(n_full / max(vi["frames"], 1), 4),
                                        est_csv_rate_hz=round(
                                            n_full / max(vi["duration"], 1e-6), 1))
            break

    # ---- 5. 建议映射（运动幅度最大的两个块）
    if len(blocks) >= 2:
        b0, b1 = blocks[0], blocks[1]
        out["suggested_yaml"] = (
            f"arm0:\n  pos_cols: {b0['pos_cols']}\n  quat_cols: {b0['quat_cols']}\n"
            f"  gripper_col: <?>\n"
            f"arm1:\n  pos_cols: {b1['pos_cols']}\n  quat_cols: {b1['quat_cols']}\n"
            f"  gripper_col: <?>"
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True, help="recording 目录")
    ap.add_argument("--sample", type=int, default=8000)
    ap.add_argument("--json", default=None, help="结果另存 JSON")
    args = ap.parse_args()

    r = analyze(args.rec, args.sample)
    txt = json.dumps(r, ensure_ascii=False, indent=2, default=str)
    print(txt)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(txt)


if __name__ == "__main__":
    main()
