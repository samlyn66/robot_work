"""子任务半自动标注工具：播放抽帧序列，键盘打点生成 subtasks.json。

用法::

    python scripts/make_subtask_labels.py --recording data/recording_20260826_162504 --fps 10

按键（播放窗口内）：
    SPACE  播放/暂停
    , / .  上一帧 / 下一帧
    1-6    在当前时间打一个"阶段开始"标记（依次：
           1=持针 grasp_needle 2=进针 insert 3=出针 exit
           4=拉线 pull 5=打结 knot 6=剪线 cut）
    9      打一个"恢复片段开始"标记（会生成 recover_ 前缀阶段，
           对应 SRT-H 的 recovery demonstrations 机制）
    u      撤销上一个标记
    s      保存 subtasks.json（阶段区间 = 相邻标记之间）
    q      退出
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tendon_vam import PHASES  # noqa: E402

KEY_PHASES = {
    ord("1"): "grasp_needle",
    ord("2"): "insert",
    ord("3"): "exit",
    ord("4"): "pull",
    ord("5"): "knot",
    ord("6"): "cut",
    ord("9"): "recover_idle",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", required=True)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--cam", default="left_endo_frames")
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(args.recording, args.cam, "*.jpg")))
    if not frames:
        print("未找到抽帧，请先运行 extract_frames.py")
        return
    marks = []  # [(time, phase)]
    i = 0
    playing = False
    win = "label tool"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        img = cv2.imread(frames[i])
        t = i / args.fps
        overlay = img.copy()
        cv2.putText(overlay, f"{os.path.basename(args.recording)}  "
                             f"frame {i}/{len(frames)}  t={t:.2f}s",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for k, (mt, ph) in enumerate(marks):
            cv2.putText(overlay, f"{k}: {ph} @ {mt:.2f}s", (10, 56 + 24 * k),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        cv2.imshow(win, overlay)
        key = cv2.waitKey(1 if playing else 30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            playing = not playing
        elif key == ord(","):
            i = max(0, i - 1)
        elif key == ord("."):
            i = min(len(frames) - 1, i + 1)
        elif key in KEY_PHASES:
            marks.append((round(t, 3), KEY_PHASES[key]))
        elif key == ord("u") and marks:
            marks.pop()
        if playing:
            i = min(len(frames) - 1, i + 1)
            if i == len(frames) - 1:
                playing = False

    events = []
    for k, (mt, ph) in enumerate(marks):
        t_end = marks[k + 1][0] if k + 1 < len(marks) else len(frames) / args.fps
        events.append({"t_start": mt, "t_end": t_end, "phase": ph})
    out = {"fps_original": args.fps, "events": events}
    with open(os.path.join(args.recording, "subtasks.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(events)} 个阶段区间 -> subtasks.json")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
