"""对录像抽帧：left_endo* / right_endo* → 统一帧率 jpg 序列。

用法::

    python scripts/extract_frames.py --data_root data --fps 10

适配实际数据布局（视频直接位于 recording 根目录，文件名带时间戳后缀）::

    recording_xxx/
        left_endo_20260826_172246_H.264.mp4    (1920x1080 @ ~60 fps)
        right_endo_20260826_172246_H.264.mp4
        ads_data.csv

- 原始视频**只读**（位于 data_root/recording_xxx/），生成帧写入独立
  output_root/recording_xxx/（默认 data_root + "_processed"），原始数据不被污染；
- 两路帧数不一致时以较短一路为准（数据集侧处理）并打印告警；
- 已有抽帧结果自动**续抽**（从已有帧之后继续，时间轴无缝衔接）；
  --overwrite 清空后重抽。
"""
from __future__ import annotations

import argparse
import sys
import os

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".mts")


def find_videos(rroot: str, pattern: str):
    """递归搜索 recording 目录（含 left_endo\\、right_endo\\ 子文件夹）里的视频。

    模式对相对路径匹配，如 left_endo* 能匹配 left_endo\\xxx.mp4。
    只接受视频扩展名，排除同名输出文件夹等干扰。
    """
    import fnmatch
    hits = []
    for dirpath, _dirs, files in os.walk(rroot):
        for fn in files:
            if not fn.lower().endswith(VIDEO_EXT):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), rroot)
            if any(fnmatch.fnmatch(rel, pattern.replace(".mp4", ext))
                   or fnmatch.fnmatch(fn, pattern.replace(".mp4", ext))
                   for ext in VIDEO_EXT):
                hits.append(os.path.join(dirpath, fn))
    return sorted(set(hits))


def extract(video_path: str, out_dir: str, target_fps: float,
            max_frames: int = 0, start_index: int = 0) -> int:
    """从第 start_index 张已抽取帧之后续抽（命名无缝衔接）。

    抽帧规则：第 k 张目标帧 = 源视频第 round(k*step) 帧，因此续抽时
    直接从 next_pick = start_index*step 开始顺序解码即可，时间轴不变。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[跳过] 无法打开 {video_path}")
        return 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    os.makedirs(out_dir, exist_ok=True)
    count = start_index
    # 顺序读取 + 按时间戳采样的抽帧（比逐帧 seek 快一个量级）
    step = src_fps / target_fps
    next_pick = start_index * step
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
    return count - start_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data",
                    help="原始数据根目录（只读，含 recording_xxx 下的视频）")
    ap.add_argument("--output_root", default=None,
                    help="生成帧输出目录（默认 <data_root>_processed，与原始数据隔离）")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--max_frames", type=int, default=0, help="测试用：最多抽取帧数")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    output_root = args.output_root or (args.data_root + "_processed")
    print(f"[抽帧] 源(只读): {args.data_root}\n       输出(生成): {output_root}")

    for name in sorted(os.listdir(args.data_root)):
        rraw = os.path.join(args.data_root, name)
        if not os.path.isdir(rraw):
            continue
        rout = os.path.join(output_root, name)
        counts = {}
        for cam in ("left_endo", "right_endo"):
            os.makedirs(rout, exist_ok=True)
            out_dir = os.path.join(rout, f"{cam}_frames")
            vids = find_videos(rraw, f"{cam}*")
            if not vids:
                continue
            existing = 0
            if os.path.isdir(out_dir):
                existing = len([f for f in os.listdir(out_dir)
                                if f.endswith((".jpg", ".png"))])
                if existing and args.overwrite:
                    for f in os.listdir(out_dir):
                        if f.endswith((".jpg", ".png")):
                            os.remove(os.path.join(out_dir, f))
                    existing = 0
            start_index = existing  # 续抽：已有帧数 = 命名起点
            total_new = 0
            for v in vids:
                n = extract(v, out_dir, args.fps, args.max_frames,
                            start_index=start_index)
                total_new += n
                start_index += n
            counts[cam] = start_index
            if total_new == 0 and existing:
                print(f"[续抽] {name}/{cam}: 已有 {existing} 帧，无新增")
            elif existing:
                print(f"[续抽] {name}/{cam}: 已有 {existing} 帧，新增 {total_new} 帧")
        if not counts:
            continue
        if len(counts) == 2 and abs(counts["left_endo"] - counts["right_endo"]) > 2:
            print(f"[告警] {name}: 双相机帧数不一致 "
                  f"L={counts['left_endo']} R={counts['right_endo']}，"
                  f"数据集将以较短一路为准，请检查双机时间同步。")
        print(f"[完成] {name}: {counts}")


if __name__ == "__main__":
    main()
