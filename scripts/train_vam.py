"""训练完整 VAM：视频骨干 + 局部去噪潜在视觉计划 + Flow Matching 动作解码器。

用法::

    python scripts/train_vam.py --config configs/default.yaml

三段式损失：
  1) plan_loss   —— 潜在视觉计划的 rectified flow（局部去噪预演）
  2) flow_loss   —— 动作解码器速度场 MSE
  3) 联合微调    —— 两项加权联合
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

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tendon_vam.config import load_config  # noqa: E402
from tendon_vam.dataset import (PairedVideoDataset, load_recordings,  # noqa: E402
                                split_recordings)
from tendon_vam.flow_decoder import FlowMatchingActionDecoder  # noqa: E402
from tendon_vam.utils import EMA, cosine_lr  # noqa: E402
from tendon_vam.video_encoder import VideoBackbone  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--video_T", type=int, default=8,
                    help="视频骨干输入的过去帧数（控制频率时序窗口）")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.data.seed)
    T = args.video_T

    recs = load_recordings(cfg.data.root, cfg.data.frame_rate)
    with_kin = [r for r in recs if r.kinematics is not None]
    if not with_kin:
        print("VAM 动作解码训练需要运动学数据。"
              "当前阶段请先运行 train_hl.py（纯视频）与视频表征预训练。")
        return
    train_r, val_r = split_recordings(with_kin, cfg)
    train_dl = DataLoader(PairedVideoDataset(train_r, cfg, "train"),
                          cfg.train.batch_size, shuffle=True,
                          num_workers=cfg.train.num_workers, drop_last=True)
    val_dl = DataLoader(PairedVideoDataset(val_r, cfg, "val"),
                        cfg.train.batch_size, num_workers=2)

    video = VideoBackbone(cfg.vam.video_backbone, cfg.vam.latent_dim,
                          cfg.vam.partial_denoise_t).to(device)
    decoder = FlowMatchingActionDecoder(cfg, cfg.vam.latent_dim,
                                        video_T=T).to(device)
    ema = EMA(decoder, cfg.vam.ema)

    params = list(video.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    total = cfg.train.epochs_vam * max(len(train_dl), 1)
    warm = cfg.train.warmup_epochs * max(len(train_dl), 1)
    scaler = torch.amp.GradScaler(enabled=cfg.train.amp)
    out_dir = os.path.join(cfg.train.out_dir, "vam")
    os.makedirs(out_dir, exist_ok=True)

    def past_future(batch, device):
        """由当前帧构造 (过去 T 帧, 未来 1 帧) 的视频张量。

        简化实现：把 hl_frames 的历史窗作为过去帧、当前帧作为未来目标。
        正式训练建议把 dataset 的采样窗口加大（T 帧连续序列）。
        """
        x = batch["hl_frames"].to(device)      # (B, 1+K, 3, H, W)
        past = x.flip(1)[:, :T]                # 最近 T 帧按时间序
        fut = x[:, :1]
        return past, fut

    step = 0
    best = float("inf")
    for epoch in range(cfg.train.epochs_vam):
        video.train(); decoder.train()
        for batch in train_dl:
            lr = cosine_lr(step, total, warm, cfg.train.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            has_kin = batch["has_kin"].to(device)
            with torch.amp.autocast("cuda", enabled=cfg.train.amp and device == "cuda"):
                past, fut = past_future(batch, device)
                l_plan = video.plan_loss(past, fut)
                # 动作 flow 损失（仅运动学样本）
                z_ctx = video.encode(past)[:, -1]
                latents = video.encode(past)
                cond_mask = has_kin[:, None].float()
                z_plan_ctx = torch.where(
                    cond_mask > 0, z_ctx, torch.zeros_like(z_ctx))
                proprio = batch["proprio"].to(device)
                # 无运动学样本用零动作（被 mask 掉，不产生梯度噪声）
                actions = batch["actions"].to(device) * cond_mask[:, None, None]
                l_flow = decoder.compute_loss(
                    z_plan_ctx, proprio * cond_mask[:, :, None],
                    batch["phase"].to(device), batch["flag"].to(device),
                    batch["correction"].to(device), actions, video_latents=latents)
                loss = l_plan + cond_mask.mean() * l_flow
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(decoder)
            step += 1
            if step % 50 == 0:
                print(f"ep {epoch} step {step} plan {l_plan.item():.4f} "
                      f"flow {l_flow.item():.4f}")

        if val_dl:
            video.eval(); decoder.eval()
            tot = n = 0
            with torch.no_grad():
                for batch in val_dl:
                    past, fut = past_future(batch, device)
                    lp = video.plan_loss(past, fut).item()
                    tot += lp * len(batch["phase"]); n += len(batch["phase"])
            v = tot / max(n, 1)
            print(f"[val] epoch {epoch} plan loss {v:.4f}")
            if v < best:
                best = v
                torch.save(video.state_dict(), os.path.join(out_dir, "video_best.pt"))
                torch.save(decoder.state_dict(), os.path.join(out_dir, "decoder_best.pt"))
        torch.save(video.state_dict(), os.path.join(out_dir, "video_last.pt"))
        torch.save(decoder.state_dict(), os.path.join(out_dir, "decoder_last.pt"))
    print(f"VAM 训练完成，权重在 {out_dir}")


if __name__ == "__main__":
    main()
