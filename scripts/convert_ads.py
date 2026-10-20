"""把 ADS 原始数据转换成训练用 kinematics.csv（含视频时间对齐）。

用法::

    python scripts/convert_ads.py --config configs/default.yaml --data_root <dataset目录>

对每个 recording_xxx：
  1. 找到 left_endo*.mp4 / right_endo*.mp4，读取帧数与 fps（**原始目录只读**）；
  2. 读取 ads_data.csv（无表头 367 列），按列映射提取双臂 pos/quat/gripper；
  3. 「CSV 行数 / 视频帧数」比例对齐到视频时间轴（数据无时间戳列的最稳做法，
     实测 CSV 时长 ≈ 视频时长，误差 <1%）；
  4. 夹爪按全数据集统计归一化到 [0,1]；四元数消除符号跳变；
  5. 写出 kinematics.csv（t + l_*/r_* 共 17 列，30 Hz）到**独立输出目录**
     （默认 <data_root>_processed，原始数据不被污染）。
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


def convert_recording(rraw: str, rout: str, cmap: AdsColumnMap, cfg, gstats) -> dict:
    info = {}
    vid_l = find_video(rraw, cfg.ads.left_video_glob)
    vid_r = find_video(rraw, cfg.ads.right_video_glob)
    csv_p = os.path.join(rraw, cfg.ads.csv_name)
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
    os.makedirs(rout, exist_ok=True)
    out = os.path.join(rout, "kinematics.csv")
    write_kinematics_csv(kin, out)
    info.update(status="ok", duration_s=round(dur, 2),
                csv_rows=None, out_rows=len(kin.t),
                out_file=out)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data_root", default=None, help="原始数据根目录（只读）")
    ap.add_argument("--output_root", default=None,
                    help="生成数据输出目录（默认 <data_root>_processed）")
    ap.add_argument("--skip_existing", action="store_true",
                    help="kinematics.csv 已存在的 recording 跳过")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = args.data_root or cfg.data.root
    output_root = args.output_root or (root + "_processed")
    cmap = AdsColumnMap.from_config(cfg)

    print(f"[1/2] 扫描夹爪值域（用于统一归一化）…（源: {root}）")
    gstats = scan_gripper_stats(root, cmap, cfg.ads.csv_name)
    print(f"      arm0 grip range={gstats['arm0']}, arm1 grip range={gstats['arm1']}")

    print(f"[2/2] 转换各 recording …（输出: {output_root}）")
    report = {}
    for name in sorted(os.listdir(root)):
        rraw = os.path.join(root, name)
        if not os.path.isdir(rraw):
            continue
        rout = os.path.join(output_root, name)
        if args.skip_existing and os.path.isfile(os.path.join(rout, "kinematics.csv")):
            report[name] = {"status": "skip（kinematics.csv 已存在）"}
            print(f"      {name}: 跳过（已转换）")
            continue
        r = convert_recording(rraw, rout, cmap, cfg, gstats)
        report[name] = r
        print(f"      {name}: {r}")

    rp = os.path.join(output_root, "ads_convert_report.json")
    os.makedirs(output_root, exist_ok=True)
    with open(rp, "w", encoding="utf-8") as f:
        json.dump({"data_root": root, "output_root": output_root,
                   "gripper_stats": {k: list(v) for k, v in gstats.items()},
                   "recordings": report}, f, ensure_ascii=False, indent=2)
    print(f"完成。报告: {rp}")


if __name__ == "__main__":
    main()
