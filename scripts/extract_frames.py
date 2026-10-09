"""对 18 组录像抽帧：left_endo / right_endo → 统一帧率 jpg 序列。

用法::

    python scripts/extract_frames.py --data_root data --fps 10

- 两个相机按各自原始 FPS 计算时间索引，以最近帧对齐（误差 <= 0.5 原始帧）。
- 输出到 recording/left_endo_frames、right_endo_frames。
- 若两路帧数不一致（录制起始时刻差），以较短一路为准并打印告警。
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".mts")


def extract(video_path: str, out_dir: str, target_fps: float) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[跳过] 无法打开 {video_path}")
        return 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(out_dir, exist_ok=True)
    times = []
    t, step = 0.0, 1.0 / target_fps
    while t * src_fps < n_src:
        times.append(t)
        t += step
    count = 0
    for tt in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, tt * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(out_dir, f"img_{count:06d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        count += 1
    cap.release()
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    for name in sorted(os.listdir(args.data_root)):
        rroot = os.path.join(args.data_root, name)
        if not os.path.isdir(rroot):
            continue
        counts = {}
        for cam in ("left_endo", "right_endo"):
            cam_dir = os.path.join(rroot, cam)
            if not os.path.isdir(cam_dir):
                continue
            vids = [f for f in sorted(os.listdir(cam_dir))
                    if f.lower().endswith(VIDEO_EXT)]
            if not vids:
                continue
            out_dir = os.path.join(rroot, f"{cam}_frames")
            total = 0
            for v in vids:
                total += extract(os.path.join(cam_dir, v), out_dir, args.fps)
            counts[cam] = total
        if len(counts) == 2 and abs(counts["left_endo"] - counts["right_endo"]) > 2:
            print(f"[告警] {name}: 双相机帧数不一致 "
                  f"L={counts['left_endo']} R={counts['right_endo']}，"
                  f"数据集将以较短一路为准，请检查双机时间同步。")
        print(f"[完成] {name}: {counts}")


if __name__ == "__main__":
    main()
