"""一键数据流水线：体检 → 转换 kinematics.csv → 抽帧（全部 recording）。

用法::

    python scripts/run_pipeline.py --data_root C:\\data\\gczjj --fps 10

等价于依次执行 validate_data.py / convert_ads.py / extract_frames.py，
并自动跳过已完成的部分：
  - convert：kinematics.csv 已存在的 recording 跳过（--force 重做）；
  - extract：断点续抽（已有帧不重抽）。

完成后打印训练入口提示。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script: str, args: list) -> int:
    cmd = [sys.executable, os.path.join(HERE, script)] + args
    print(f"\n=== python {os.path.basename(script)} {' '.join(args)} ===\n")
    return subprocess.run(cmd).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data_root", required=True,
                    help="18 条 recording 所在目录，如 C:\\data\\gczjj")
    ap.add_argument("--fps", type=float, default=10.0, help="抽帧目标帧率")
    ap.add_argument("--force_convert", action="store_true",
                    help="已有 kinematics.csv 也重做转换")
    ap.add_argument("--skip_validate", action="store_true")
    ap.add_argument("--skip_convert", action="store_true")
    ap.add_argument("--skip_extract", action="store_true")
    args = ap.parse_args()

    rc = 0
    if not args.skip_validate:
        rc = run("validate_data.py",
                 ["--config", args.config, "--data_root", args.data_root])
        if rc != 0:
            print("\n体检存在 fail 项：请先解决缺失/损坏的 recording，"
                  "或用 --skip_validate 跳过体检继续。")
            if not args.skip_validate:
                return rc

    if not args.skip_convert:
        extra = [] if args.force_convert else ["--skip_existing"]
        run("convert_ads.py",
            ["--config", args.config, "--data_root", args.data_root] + extra)

    if not args.skip_extract:
        run("extract_frames.py",
            ["--data_root", args.data_root, "--fps", str(args.fps)])

    print("\n流水线结束。下一步：")
    print("  1) 标注子任务（每条 1 分钟，标 3~5 条即可先训 HL）：")
    print(f"     python scripts\\make_subtask_labels.py --recording {args.data_root}\\recording_xxx")
    print("  2) 训练：")
    print(f"     python scripts\\train_hl.py  --config {args.config} --data_root {args.data_root}")
    print(f"     python scripts\\train_ll.py  --config {args.config} --data_root {args.data_root}")
    print(f"     python scripts\\train_vam.py --config {args.config} --data_root {args.data_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
