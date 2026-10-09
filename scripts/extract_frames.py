"""对录像抽帧：left_endo* / right_endo* → 统一帧率 jpg 序列。

用法::

    python scripts/extract_frames.py --data_root data --fps 10

适配实际数据布局（视频直接位于 recording 根目录，文件名带时间戳后缀）::

    recording_xxx/
        left_endo_20260826_172246_H.264.mp4    (1920x1080 @ ~60 fps)
        right_endo_20260826_172246_H.264.mp4
        ads_data.csv

- 按各自原始 FPS 计算目标时刻，最近帧对齐；
- 输出到 recording/left_endo_frames、right_endo_frames；
- 两路帧数不一致时以较短一路为准（数据集侧处理）并打印告警；
- 已有抽帧结果默认跳过（--overwrite 强制重抽）。
"""
from __future__ import annotations

import argparse
import glob as globmod
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".mts")


def find_videos(rroot: str, pattern: str):
    hits = []
    for ext in VIDEO_EXT:
        hits += globmod.glob(os.path.join(rroot, pattern.replace(".mp4", ext)))
    return sorted(set(hits))


def extract(video_path: str, out_dir: str, target_fps: float,
            max_frames: int = 0) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[跳过] 无法打开 {video_path}")
        return 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    # 顺序读取 + 按时间戳采样的抽帧（比逐帧 seek 快一个量级）
    step = src_fps / target_fps
    next_pick = 0.0
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi >= next_pick - 1e-6:
            cv2.imwrite(os.path.join(out_dir, f"img_{count:06d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            count += 1
            next_pick += step
            if max_frames and count >= max_frames:
                break
        fi += 1
    cap.release()
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--max_frames", type=int, default=0, help="测试用：最多抽取帧数")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    for name in sorted(os.listdir(args.data_root)):
        rroot = os.path.join(args.data_root, name)
        if not os.path.isdir(rroot):
            continue
        counts = {}
        for cam in ("left_endo", "right_endo"):
            out_dir = os.path.join(rroot, f"{cam}_frames")
            vids = find_videos(rroot, f"{cam}*")
            if not vids:
                continue
            if os.path.isdir(out_dir) and os.listdir(out_dir) and not args.overwrite:
                counts[cam] = len([f for f in os.listdir(out_dir)
                                   if f.endswith((".jpg", ".png"))])
                continue
            total = 0
            for v in vids:
                total += extract(v, out_dir, args.fps, args.max_frames)
            counts[cam] = total
        if not counts:
            continue
        if len(counts) == 2 and abs(counts["left_endo"] - counts["right_endo"]) > 2:
            print(f"[告警] {name}: 双相机帧数不一致 "
                  f"L={counts['left_endo']} R={counts['right_endo']}，"
                  f"数据集将以较短一路为准，请检查双机时间同步。")
        print(f"[完成] {name}: {counts}")


if __name__ == "__main__":
    main()
