"""评估与可视化。

- HL：留出视频上滚动预测子任务序列，输出逐帧阶段曲线图（PNG）+ 准确率
- LL/VAM：有运动学与权重时，采样动作块并绘制 XYZ 轨迹对比图

用法::

    python scripts/evaluate.py --config configs/default.yaml \
        --ckpt runs/hl/best.pt --recording data/recording_xxx
"""
from __future__ import annotations

import argparse
import os
import sys

# 兼容性补丁：部分装有 SSL VPN 安全客户端的 Windows 机器会设置
# SSLKEYLOGFILE 指向系统盘根目录（如 C:\nss_ssl_sfagent.log），
# 导入 torch/ssl 时会尝试写入导致 PermissionError；不可写则重定向。
_klf = os.environ.get("SSLKEYLOGFILE")
if _klf:
    try:  # 真实试写（Windows 的 os.access 不可靠：不查 ACL/文件锁）
        with open(_klf, "a"):
            pass
    except OSError:
        os.environ["SSLKEYLOGFILE"] = os.path.join(
            os.environ.get("TEMP", os.getcwd()), "ssl_keylog.log")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tendon_vam import PHASES  # noqa: E402
from tendon_vam.config import load_config  # noqa: E402
from tendon_vam.dataset import PairedVideoDataset, load_recordings  # noqa: E402
from tendon_vam.hl_policy import HighLevelPolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="runs/hl/best.pt")
    ap.add_argument("--recording", default=None, help="指定留出录像名；默认全部无标注录像")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    recs = load_recordings(cfg.data.root, cfg.data.frame_rate)
    if args.recording:
        recs = [r for r in recs if r.name == args.recording]
    if not recs:
        print("未找到目标录像。")
        return

    model = HighLevelPolicy(cfg).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    ds = PairedVideoDataset(recs, cfg, "val")
    dl = DataLoader(ds, cfg.train.batch_size, num_workers=2)
    ts, pred_phase, true_phase, flags = [], [], [], []
    correct = 0
    with torch.no_grad():
        for batch in dl:
            out = model.predict(batch["hl_frames"].to(device))
            p = out["phase"].cpu().numpy()
            t = batch["phase"].numpy()
            pred_phase.append(p)
            true_phase.append(t)
            flags.append(out["flag"].cpu().numpy())
            ts.append(batch["t"].numpy())
            correct += (p == t).sum()
    ts = np.concatenate(ts)
    order = np.argsort(ts)
    ts, pred_phase = ts[order], np.concatenate(pred_phase)[order]
    true_phase = np.concatenate(true_phase)[order]
    flags = np.concatenate(flags)[order]
    acc = correct / len(ts)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].scatter(ts, pred_phase, s=6, c="tab:red", label="predicted")
    axes[0].scatter(ts, true_phase, s=6, c="tab:blue", alpha=0.4, label="ground truth")
    axes[0].set_yticks(range(len(PHASES)))
    axes[0].set_yticklabels(PHASES)
    axes[0].legend(); axes[0].set_title(f"HL 子任务识别（acc={acc:.3f}）")
    axes[1].scatter(ts, flags, s=6, c="tab:green")
    axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["正常", "纠错"])
    axes[1].set_xlabel("t (s)")
    out_png = os.path.join(args.out, "hl_rollout.png")
    fig.tight_layout(); fig.savefig(out_png, dpi=150)
    print(f"准确率 {acc:.4f}，图已保存 {out_png}")

    # ---- VAM 动作可视化（有解码器权重与运动学时）
    dec_ckpt = os.path.join(cfg.train.out_dir, "vam", "decoder_best.pt")
    vid_ckpt = os.path.join(cfg.train.out_dir, "vam", "video_best.pt")
    if os.path.exists(dec_ckpt) and any(r.kinematics is not None for r in recs):
        from tendon_vam.flow_decoder import FlowMatchingActionDecoder
        from tendon_vam.video_encoder import VideoBackbone
        video = VideoBackbone(cfg.vam.video_backbone, cfg.vam.latent_dim,
                              cfg.vam.partial_denoise_t).to(device).eval()
        video.load_state_dict(torch.load(vid_ckpt, map_location=device))
        dec = FlowMatchingActionDecoder(cfg, cfg.vam.latent_dim).to(device).eval()
        dec.load_state_dict(torch.load(dec_ckpt, map_location=device))
        batch = next(iter(dl))
        x = batch["hl_frames"].to(device).flip(1)[:, :8]
        with torch.no_grad():
            lat = video.encode(x)
            z = dec.build_cond(lat[:, -1], batch["proprio"].to(device),
                               batch["phase"].to(device), batch["flag"].to(device),
                               batch["correction"].to(device))
            act = dec.sample(z, batch["proprio"].to(device),
                             batch["phase"].to(device), batch["flag"].to(device),
                             batch["correction"].to(device),
                             video_latents=lat).cpu().numpy()
        fig2, axs = plt.subplots(1, 3, figsize=(12, 3))
        for j, axname in enumerate(["dx", "dy", "dz"]):
            axs[j].plot(act[0, :, j], label="left arm")
            axs[j].plot(act[0, :, 10 + j], label="right arm")
            axs[j].set_title(axname); axs[j].legend()
        out_png2 = os.path.join(args.out, "vam_action_chunk.png")
        fig2.tight_layout(); fig2.savefig(out_png2, dpi=150)
        print(f"动作块可视化 {out_png2}")


if __name__ == "__main__":
    main()
