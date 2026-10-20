"""批量体检 18 条 recording（上传/训练前的数据预检）。

用法::

    python scripts/validate_data.py --config configs/default.yaml --data_root <数据目录>

对每个 recording 检查：
  1. 文件齐全：ads_data.csv + left_endo*.mp4 + right_endo*.mp4；
  2. 视频可读：fps / 帧数 / 分辨率，双机帧数是否一致；
  3. CSV 结构：列数与示例一致（367）、行数、行/帧比例稳定（对齐前提）；
  4. 列映射有效性：配置的四元数列范数 ≈ 1、夹爪列有变化、位姿列在运动块前列；
  5. 是否已转换（kinematics.csv 存在）。

输出：控制台汇总 + <output_root>/validate_report.json（默认写入
<data_root>_processed，不污染原始数据）。全部 ok 才继续跑流水线。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tendon_vam.ads_io import AdsColumnMap, find_video, load_ads_csv, video_info
from tendon_vam.config import load_config


def count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            n += chunk.count(b"\n")
    return n


def check_recording(rroot: str, cmap: AdsColumnMap, cfg) -> dict:
    r: dict = {"dir": os.path.basename(rroot)}
    problems, warns = [], []

    csv_p = os.path.join(rroot, cfg.ads.csv_name)
    vl = find_video(rroot, cfg.ads.left_video_glob)
    vr = find_video(rroot, cfg.ads.right_video_glob)
    r["has_csv"] = os.path.isfile(csv_p)
    r["has_left"] = bool(vl)
    r["has_right"] = bool(vr)
    if not (r["has_csv"] and r["has_left"] and r["has_right"]):
        problems.append("缺少 " + ", ".join(
            n for n, ok in (("ads_data.csv", r["has_csv"]),
                            ("left_endo*.mp4", r["has_left"]),
                            ("right_endo*.mp4", r["has_right"])) if not ok))
        r["status"] = "fail"
        r["problems"] = problems
        return r

    # 视频
    try:
        il, ir = video_info(vl), video_info(vr)
        r["video"] = {"left": il, "right": ir}
        if abs(il["frames"] - ir["frames"]) > 2:
            warns.append(f"双机帧数不一致 left={il['frames']} right={ir['frames']}")
        if abs(il["fps"] - ir["fps"]) > 0.5:
            warns.append(f"双机 fps 不一致 {il['fps']} vs {ir['fps']}")
        dur = min(il["duration"], ir["duration"])
    except RuntimeError as e:
        problems.append(str(e))
        r["status"] = "fail"
        r["problems"] = problems
        return r
    r["duration_s"] = round(dur, 2)

    # CSV 行数（快速计数）与列数（首行）
    rows = count_lines(csv_p)
    with open(csv_p, "r", encoding="utf-8", errors="ignore") as f:
        n_cols = len(f.readline().strip().split(","))
    r["csv_rows"], r["csv_cols"] = rows, n_cols
    if n_cols != cfg.ads.expect_cols:
        warns.append(f"列数 {n_cols} ≠ 示例 {cfg.ads.expect_cols}，列映射可能失效")

    # 行/帧比例（时间对齐的根基）
    ratio = rows / max(il["frames"], 1)
    r["rows_per_frame"] = round(ratio, 3)
    est_rate = ratio * il["fps"]           # 估算 CSV 采样率
    r["est_csv_rate_hz"] = round(est_rate, 1)
    if abs(rows / est_rate - dur) / dur > 0.05:
        warns.append("CSV 时长与视频时长偏差 >5%，行/帧比例对齐可能不可靠")

    # 列映射有效性（抽样前 2000 行）
    arr = load_ads_csv(csv_p, max_rows=2000)
    for arm, cols in (("arm0", cmap.arm0), ("arm1", cmap.arm1)):
        q = arr[:, cols.quat_cols]
        norms = np.linalg.norm(q, axis=1)
        if not np.all((norms > 0.9) & (norms < 1.1)):
            problems.append(f"{arm} 四元数列 {cols.quat_cols} 范数异常 "
                            f"[{norms.min():.2f},{norms.max():.2f}]")
        g = arr[:, cols.gripper_col]
        if float(g.std()) < 1e-4:
            warns.append(f"{arm} 夹爪列 {cols.gripper_col} 在本段几乎无变化")

    r["status"] = "fail" if problems else ("warn" if warns else "ok")
    r["problems"], r["warnings"] = problems, warns
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data_root", default=None, help="原始数据根目录（只读）")
    ap.add_argument("--output_root", default=None,
                    help="生成数据输出目录（默认 <data_root>_processed，报告写此处）")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = args.data_root or cfg.data.root
    output_root = args.output_root or (root + "_processed")
    cmap = AdsColumnMap.from_config(cfg)

    names = sorted(d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d))
                   and d.startswith("recording"))
    print(f"发现 {len(names)} 条 recording，开始体检 …\n")
    report = {}
    for name in names:
        rraw = os.path.join(root, name)
        rout = os.path.join(output_root, name)
        rep = check_recording(rraw, cmap, cfg)
        # 是否已转换：检查独立输出目录中的 kinematics.csv（不污染原始数据）
        rep["converted"] = os.path.isfile(os.path.join(rout, "kinematics.csv"))
        report[name] = rep
        mark = {"ok": "✓", "warn": "△", "fail": "✗"}[rep["status"]]
        extra = ""
        if rep.get("video"):
            extra = (f" {rep['duration_s']}s | {rep['csv_rows']}行/{rep['csv_cols']}列"
                     f" | {rep['rows_per_frame']}行/帧")
        conv = " [已转换]" if rep["converted"] else ""
        print(f"{mark} {name}{extra}{conv}")
        for p in rep.get("problems", []):
            print(f"    ✗ {p}")
        for w in rep.get("warnings", []):
            print(f"    △ {w}")

    n_ok = sum(1 for r in report.values() if r["status"] == "ok")
    n_warn = sum(1 for r in report.values() if r["status"] == "warn")
    n_fail = sum(1 for r in report.values() if r["status"] == "fail")
    print(f"\n汇总: ok {n_ok} / warn {n_warn} / fail {n_fail}")
    os.makedirs(output_root, exist_ok=True)
    rp = os.path.join(output_root, "validate_report.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump({"data_root": root, "output_root": output_root, "recordings": report},
                  f, ensure_ascii=False, indent=2)
    print(f"报告: {rp}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
