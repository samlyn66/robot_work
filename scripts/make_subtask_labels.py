"""子任务半自动标注工具：播放抽帧序列，键盘打点生成 subtasks.json。

用法::

    # 单条标注
    python scripts/make_subtask_labels.py --recording data/recording_20260826_162504 --fps 10

    # 批量标注：遍历 root 下所有 recording_*，逐条弹出窗口（已标过的跳过）
    python scripts/make_subtask_labels.py --root data_processed --all --fps 10
    #   加 --redo 可对已有 subtasks.json 的 recording 重新标注

⚠️ Windows 注意事项：
    1. 请在服务器**桌面本机的 PowerShell/Anaconda Prompt** 里运行，
       不要从 VS Code Remote-SSH 终端启动 —— SSH 会话里启动的 OpenCV
       窗口收不到键盘消息（窗口显示"未响应"）。
    2. 键盘按键必须作用在**视频窗口**上（先点一下视频窗口获得焦点），
       而不是终端里。
    3. 直接点窗口右上角 ✕ 也可以安全退出（会自动保存已打的标记）。

按键（在视频窗口内按）：
    SPACE  播放/暂停
    , / .  上一帧 / 下一帧
    1-6    在当前时间打一个"阶段开始"标记（依次：
           1=持针 grasp_needle 2=进针 insert 3=出针 exit
           4=拉线 pull 5=打结 knot 6=剪线 cut）
    9      打一个"恢复片段开始"标记（对应 SRT-H 的 recovery 机制）
    u      撤销上一个标记
    s      保存 subtasks.json（阶段区间 = 相邻标记之间）
    q / ESC  退出（退出时自动保存）
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


def save(recording: str, marks, n_frames: int, fps: float) -> str:
    events = []
    for k, (mt, ph) in enumerate(marks):
        t_end = marks[k + 1][0] if k + 1 < len(marks) else n_frames / fps
        events.append({"t_start": mt, "t_end": t_end, "phase": ph})
    out = {"fps_original": fps, "events": events}
    path = os.path.join(recording, "subtasks.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return path


def label_one(recording: str, fps: float, cam: str) -> None:
    """对单个 recording 弹窗标注，退出时自动保存 subtasks.json。"""
    frames = sorted(glob.glob(os.path.join(recording, cam, "*.jpg")))
    if not frames:
        print(f"[跳过] {recording}: 未找到抽帧，请先运行 extract_frames.py")
        return
    print(f"\n>>> 正在标注: {os.path.basename(recording)}  ({len(frames)} 帧)")
    print("    （按键说明见上方；先点视频窗口获得焦点）")
    print(__doc__)  # 在终端打印按键说明
    marks = []  # [(time, phase)]
    i = 0
    playing = False
    win = f"label - {os.path.basename(recording)}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    closed = False
    while True:
        img = cv2.imread(frames[i])
        if img is None:  # 坏帧容错
            i = min(len(frames) - 1, i + 1)
            if img is None and cv2.waitKey(1) == 27:
                break
            continue
        t = i / fps
        overlay = img.copy()
        cv2.putText(overlay, f"{os.path.basename(recording)}  "
                             f"frame {i}/{len(frames)}  t={t:.2f}s"
                             f"{'  [PLAY]' if playing else ''}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for k, (mt, ph) in enumerate(marks):
            cv2.putText(overlay, f"{k}: {ph} @ {mt:.2f}s", (10, 56 + 24 * k),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        cv2.putText(overlay, "SPACE play  ,/. frame  1-6 mark  u undo  s save  q quit",
                    (10, overlay.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow(win, overlay)
        key = cv2.waitKey(1 if playing else 30) & 0xFF
        # 检测用户点了窗口 ✕（Windows 下 getWindowProperty 变 -1）
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            closed = True
        if key in (ord("q"), 27) or closed:  # q / ESC / 点✕ 都退出
            break
        elif key == ord(" "):
            playing = not playing
        elif key == ord(","):
            i = max(0, i - 1)
        elif key == ord("."):
            i = min(len(frames) - 1, i + 1)
        elif key in KEY_PHASES:
            marks.append((round(t, 3), KEY_PHASES[key]))
            marks.sort()
        elif key == ord("u") and marks:
            marks.pop()
        elif key == ord("s"):
            p = save(recording, marks, len(frames), fps)
            print(f"已保存 -> {p}")
        if playing:
            i = min(len(frames) - 1, i + 1)
            if i == len(frames) - 1:
                playing = False

    p = save(recording, marks, len(frames), fps)
    print(f"已保存 {len(marks)} 个阶段区间 -> {p}")
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default=None,
                    help="单条标注：recording 目录（含 *_frames）")
    ap.add_argument("--root", default=None,
                    help="批量标注(--all)：所有 recording_* 所在目录，"
                         "即流水线的输出目录（如 C:\\data\\gczjj_processed）")
    ap.add_argument("--all", action="store_true",
                    help="批量模式：遍历 --root 下全部 recording，逐条标注")
    ap.add_argument("--redo", action="store_true",
                    help="批量模式下，已有 subtasks.json 的 recording 也重新标注")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--cam", default="left_endo_frames")
    args = ap.parse_args()

    if args.all:
        if not args.root:
            ap.error("--all 需要同时指定 --root <流水线输出目录>")
        names = sorted(d for d in os.listdir(args.root)
                       if os.path.isdir(os.path.join(args.root, d))
                       and d.startswith("recording"))
        todo = []
        for n in names:
            sp = os.path.join(args.root, n, "subtasks.json")
            if args.redo or not os.path.isfile(sp):
                todo.append(os.path.join(args.root, n))
        print(f"批量标注：共 {len(names)} 条 recording，"
              f"待标注 {len(todo)} 条（其余已有 subtasks.json 跳过）")
        for k, rec in enumerate(todo, 1):
            print(f"\n===== [{k}/{len(todo)}] "
                  f"{os.path.basename(rec)} =====")
            label_one(rec, args.fps, args.cam)
        print("\n批量标注结束。已标注：",
              sum(1 for n in names
                  if os.path.isfile(os.path.join(args.root, n, "subtasks.json"))),
              "/", len(names))
        return

    # 单条模式
    if not args.recording:
        ap.error("请指定 --recording <recording目录>，或用 --root ... --all 批量标注")
    label_one(args.recording, args.fps, args.cam)


if __name__ == "__main__":
    main()
