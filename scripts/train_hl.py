"""训练高层子任务规划器（纯视频即可训练，18 条视频现状下第一步）。

用法::

    python scripts/train_hl.py --config configs/default.yaml
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
from tendon_vam.hl_policy import HighLevelPolicy  # noqa: E402
from tendon_vam.utils import cosine_lr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.data.seed)

    recs = load_recordings(cfg.data.root, cfg.data.frame_rate)
    if not recs:
        print(f"在 {cfg.data.root} 下未找到录像（需先抽帧）。")
        return
    train_r, val_r = split_recordings(recs, cfg)
    labeled = [r for r in train_r if r.subtasks]
    val = [r for r in val_r if r.subtasks]
    print(f"录像 {len(recs)} 组 | 有标注: train {len(labeled)} / val {len(val)}")
    if not labeled:
        print("没有 subtasks.json 标注，请先运行 make_subtask_labels.py。")
        return

    train_ds = PairedVideoDataset(labeled, cfg, "train")
    val_ds = PairedVideoDataset(val, cfg, "val")
    train_dl = DataLoader(train_ds, cfg.train.batch_size, shuffle=True,
                          num_workers=cfg.train.num_workers, drop_last=True,
                          persistent_workers=cfg.train.num_workers > 0)
    val_dl = DataLoader(val_ds, cfg.train.batch_size, shuffle=False,
                        num_workers=cfg.train.num_workers)

    model = HighLevelPolicy(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    steps_per_epoch = max(len(train_dl), 1)
    total = cfg.train.epochs_hl * steps_per_epoch
    warm = cfg.train.warmup_epochs * steps_per_epoch
    scaler = torch.amp.GradScaler(enabled=cfg.train.amp)
    out_dir = os.path.join(cfg.train.out_dir, "hl")
    os.makedirs(out_dir, exist_ok=True)

    step = 0
    best = 0.0
    for epoch in range(cfg.train.epochs_hl):
        model.train()
        for batch in train_dl:
            lr = cosine_lr(step, total, warm, cfg.train.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.train.amp and device == "cuda"):
                m = model.compute_loss(
                    batch["hl_frames"].to(device),
                    batch["phase"].to(device),
                    batch["flag"].to(device),
                    batch["correction"].to(device))
            scaler.scale(m["loss"]).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % 50 == 0:
                print(f"ep {epoch} step {step} loss {m['loss'].item():.4f} "
                      f"(phase {m['phase'].item():.4f})")

        # ---- 验证：子任务分类准确率
        if val_dl:
            model.eval()
            correct = total_n = 0
            with torch.no_grad():
                for batch in val_dl:
                    out = model.predict(batch["hl_frames"].to(device))
                    correct += (out["phase"] == batch["phase"].to(device)).sum().item()
                    total_n += len(batch["phase"])
            acc = correct / max(total_n, 1)
            print(f"[val] epoch {epoch} 子任务准确率 {acc:.4f}")
            if acc > best:
                best = acc
                torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
        torch.save(model.state_dict(), os.path.join(out_dir, "last.pt"))
    print(f"训练完成，最优子任务准确率 {best:.4f}，权重在 {out_dir}")


if __name__ == "__main__":
    main()
