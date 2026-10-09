"""把 ADS 原始数据转换成训练用 kinematics.csv（含视频时间对齐）。

用法::

    python scripts/convert_ads.py --config configs/default.yaml --data_root <dataset目录>

对每个 recording_xxx：
  1. 找到 left_endo*.mp4 / right_endo*.mp4，读取帧数与 fps；
  2. 读取 ads_data.csv（无表头 367 列），按列映射提取双臂 pos/quat/gripper；
  3. 「CSV 行数 / 视频帧数」比例对齐到视频时间轴（数据无时间戳列的最稳做法，
     实测 CSV 时长 ≈ 视频时长，误差 <1%）；
  4. 夹爪按全数据集统计归一化到 [0,1]；四元数消除符号跳变；
  5. 写出 kinematics.csv（t + l_*/r_* 共 17 列，30 Hz）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tendon_vam.ads_io import (AdsColumnMap, extract_ads_kinematics,
                               find_video, scan_gripper_stats, video_info,
                               write_kinematics_csv)
from tendon_vam.config import load_config


def convert_recording(rroot: str, cmap: AdsColumnMap, cfg, gstats) -> dict:
    info = {}
    vid_l = find_video(rroot, cfg.ads.left_video_glob)
    vid_r = find_video(rroot, cfg.ads.right_video_glob)
    csv_p = os.path.join(rroot, cfg.ads.csv_name)
    if not (vid_l and vid_r and os.path.isfile(csv_p)):
        info["status"] = "skip（缺视频或 ads_data.csv）"
        return info
    vl, vr = video_info(vid_l), video_info(vid_r)
    # 以双机中较短时长为基准（双机同起录、帧数一致时无差异）
    dur = min(vl["duration"], vr["duration"])
    kin = extract_ads_kinematics(
        csv_p, cmap, dur,
        out_rate=float(cfg.ads.out_rate),
        fix_sign=bool(cfg.ads.fix_quat_sign),
        gripper_stats=gstats,
    )
    out = os.path.join(rroot, "kinematics.csv")
    write_kinematics_csv(kin, out)
    info.update(status="ok", duration_s=round(dur, 2),
                csv_rows=None, out_rows=len(kin.t),
                out_file=out)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data_root", default=None, help="默认取 config data.root")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = args.data_root or cfg.data.root
    cmap = AdsColumnMap.from_config(cfg)

    print(f"[1/2] 扫描夹爪值域（用于统一归一化）…")
    gstats = scan_gripper_stats(root, cmap, cfg.ads.csv_name)
    print(f"      arm0 grip range={gstats['arm0']}, arm1 grip range={gstats['arm1']}")

    print(f"[2/2] 转换各 recording …")
    report = {}
    for name in sorted(os.listdir(root)):
        rroot = os.path.join(root, name)
        if not os.path.isdir(rroot):
            continue
        r = convert_recording(rroot, cmap, cfg, gstats)
        report[name] = r
        print(f"      {name}: {r}")

    rp = os.path.join(root, "ads_convert_report.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump({"gripper_stats": {k: list(v) for k, v in gstats.items()},
                   "recordings": report}, f, ensure_ascii=False, indent=2)
    print(f"完成。报告: {rp}")


if __name__ == "__main__":
    main()
