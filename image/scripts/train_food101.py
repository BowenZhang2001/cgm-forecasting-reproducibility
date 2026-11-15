"""Stage A: Food-101 category-supervised fine-tuning of ImageNet ConvNeXt-Base.

Splits (no leakage):
  - official train (75,750) -> fit set + a stratified 5% validation hold-out (seeded) used only for
    checkpoint selection;
  - official test (25,250) -> touched once, after training, for the final accuracy report.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (ConvNeXtBase, environment_info, eval_transform, load_image,  # noqa: E402
                        seed_everything, sha256_lines, train_transform, write_json)


class Food101(Dataset):
    def __init__(self, root, items, classes, tf):
        self.root, self.items, self.tf = Path(root), items, tf
        self.cls = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        return self.tf(load_image(self.root / "images" / f"{it}.jpg")), self.cls[it.split("/")[0]]


def split_train_val(items, frac, seed):
    rng = np.random.default_rng(seed)
    by_cls = {}
    for it in items:
        by_cls.setdefault(it.split("/")[0], []).append(it)
    fit, val = [], []
    for c in sorted(by_cls):
        lst = sorted(by_cls[c])
        rng.shuffle(lst)
        k = round(len(lst) * frac)
        val += lst[:k]
        fit += lst[k:]
    return sorted(fit), sorted(val)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = top5 = n = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
        loss_sum += ce(out.float(), y).item()
        correct += (out.argmax(1) == y).sum().item()
        top5 += (out.topk(5, 1).indices == y[:, None]).any(1).sum().item()
        n += len(y)
    return dict(top1=correct / n, top5=top5 / n, loss=loss_sum / n, n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="food-101 directory (images/, meta/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr-mult", type=float, default=10.0)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=0.5)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop after N steps")
    a = ap.parse_args()

    seed_everything(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(a.root)
    classes = (root / "meta/classes.txt").read_text().split()
    train_items = (root / "meta/train.txt").read_text().split()
    test_items = (root / "meta/test.txt").read_text().split()
    assert not set(train_items) & set(test_items)
    fit, val = split_train_val(train_items, a.val_frac, a.seed)
    (out / "fit_items.txt").write_text("\n".join(fit))
    (out / "val_items.txt").write_text("\n".join(val))
    write_json(out / "config.json", dict(
        args=vars(a), n_fit=len(fit), n_val=len(val), n_test=len(test_items),
        fit_sha256=sha256_lines(fit), val_sha256=sha256_lines(val), test_sha256=sha256_lines(test_items),
        classes=classes, env=environment_info(),
        model="torchvision convnext_base IMAGENET1K_V1 init, stochastic_depth 0.2, head Linear(1024,101)",
        selection_rule="highest validation top-1 accuracy (5% stratified hold-out of official train)"))

    device = "cuda"
    dl_kw = dict(num_workers=a.workers, pin_memory=True, persistent_workers=True)
    fit_dl = DataLoader(Food101(root, fit, classes, train_transform()), a.batch, shuffle=True,
                        drop_last=True, **dl_kw)
    val_dl = DataLoader(Food101(root, val, classes, eval_transform()), 96, **dl_kw)

    model = ConvNeXtBase(len(classes)).to(device).to(memory_format=torch.channels_last)
    head = list(model.head.parameters())
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    decay = lambda ps: [p for p in ps if p.ndim > 1]  # noqa: E731
    no_decay = lambda ps: [p for p in ps if p.ndim <= 1]  # noqa: E731
    opt = torch.optim.AdamW([
        dict(params=decay(body), lr=a.lr, weight_decay=a.wd),
        dict(params=no_decay(body), lr=a.lr, weight_decay=0.0),
        dict(params=decay(head), lr=a.lr * a.head_lr_mult, weight_decay=a.wd),
        dict(params=no_decay(head), lr=a.lr * a.head_lr_mult, weight_decay=0.0),
    ])
    base_lrs = [g["lr"] for g in opt.param_groups]
    total = a.epochs * len(fit_dl)
    warm = int(a.warmup_epochs * len(fit_dl))
    scaler = torch.amp.GradScaler()
    crit = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)

    history, best, step = [], -1.0, 0
    for ep in range(a.epochs):
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for x, y in fit_dl:
            f = step / max(warm, 1) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * f
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * len(y)
            seen += len(y)
            step += 1
            if step % 200 == 0:
                print(f"ep {ep} step {step}/{total} loss {run_loss / seen:.4f} "
                      f"{seen / (time.time() - t0):.1f} img/s", flush=True)
            if a.max_steps and step >= a.max_steps:
                break
        v = evaluate(model, val_dl, device)
        rec = dict(epoch=ep, train_loss=run_loss / max(seen, 1), val=v, sec=time.time() - t0)
        history.append(rec)
        print(json.dumps(rec), flush=True)
        torch.save(dict(model=model.state_dict(), epoch=ep, val=v), out / "last.pt")
        if v["top1"] > best:
            best = v["top1"]
            torch.save(dict(model=model.state_dict(), epoch=ep, val=v, classes=classes), out / "best.pt")
        write_json(out / "history.json", history)
        if a.max_steps and step >= a.max_steps:
            break

    if a.max_steps:
        return  # debug runs never touch the test set
    # final, single use of the official test set with the selected checkpoint
    ck = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    test_dl = DataLoader(Food101(root, test_items, classes, eval_transform()), 96, **dl_kw)
    t = evaluate(model, test_dl, device)
    write_json(out / "test_metrics.json", dict(selected_epoch=ck["epoch"], val=ck["val"], test=t))
    print("TEST", json.dumps(t), flush=True)


if __name__ == "__main__":
    main()
