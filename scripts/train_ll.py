"""训练低层策略（SRT-H 式语言条件 ACT，需运动学数据）。

用法::

    python scripts/train_ll.py --config configs/default.yaml

数据缺失运动学时会自动退出并提示（LL 训练需要动作标签）。
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tendon_vam.config import load_config  # noqa: E402
from tendon_vam.dataset import (PairedVideoDataset, load_recordings,  # noqa: E402
                                split_recordings)
from tendon_vam.ll_policy import LowLevelPolicy  # noqa: E402
from tendon_vam.utils import cosine_lr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.data.seed)

    recs = load_recordings(cfg.data.root, cfg.data.frame_rate)
    with_kin = [r for r in recs if r.kinematics is not None]
    print(f"录像 {len(recs)} 组，其中含运动学数据 {len(with_kin)} 组")
    if not with_kin:
        print("LL 训练需要运动学数据（kinematics.csv）。"
              "当前 18 条视频先用于 HL 子任务规划器与视频表征训练；"
              "待运动学数据到位后重跑本脚本。")
        return
    train_r, val_r = split_recordings(with_kin, cfg)
    train_ds = PairedVideoDataset(train_r, cfg, "train", need_actions=True)
    val_ds = PairedVideoDataset(val_r, cfg, "val", need_actions=True)
    train_dl = DataLoader(train_ds, cfg.train.batch_size, shuffle=True,
                          num_workers=cfg.train.num_workers, drop_last=True)
    val_dl = DataLoader(val_ds, cfg.train.batch_size, num_workers=2)

    model = LowLevelPolicy(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    total = cfg.train.epochs_ll * max(len(train_dl), 1)
    warm = cfg.train.warmup_epochs * max(len(train_dl), 1)
    scaler = torch.amp.GradScaler(enabled=cfg.train.amp)
    out_dir = os.path.join(cfg.train.out_dir, "ll")
    os.makedirs(out_dir, exist_ok=True)

    step = 0
    best = float("inf")
    for epoch in range(cfg.train.epochs_ll):
        model.train()
        for batch in train_dl:
            lr = cosine_lr(step, total, warm, cfg.train.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.train.amp and device == "cuda"):
                loss = model.compute_loss(batch, device)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % 50 == 0:
                print(f"ep {epoch} step {step} L1 {loss.item():.5f}")

        if val_dl:
            model.eval()
            tot = n = 0
            with torch.no_grad():
                for batch in val_dl:
                    loss = model.compute_loss(batch, device)
                    tot += loss.item() * len(batch["phase"])
                    n += len(batch["phase"])
            v = tot / max(n, 1)
            print(f"[val] epoch {epoch} 动作 L1 {v:.5f}")
            if v < best:
                best = v
                torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
        torch.save(model.state_dict(), os.path.join(out_dir, "last.pt"))
    print(f"训练完成，最优 val L1 {best:.5f}，权重在 {out_dir}")


if __name__ == "__main__":
    main()
