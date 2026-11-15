"""Stage B: Nutrition5k dish-level nutrition regression (calories, carb, protein, fat).

Init: Food-101 category-supervised ConvNeXt-Base checkpoint (backbone only; new 4-output head).
Loss: mean over the 4 nutrients of the MSE on z-scored targets (scaler fitted on fit dishes only).
Splits (no leakage):
  - official rgb_train -> fit + validation (10% of capture sessions held out as whole sessions);
  - implausible-label dishes dropped from fit/val (rule in src/n5k/data.py, label-only);
  - official rgb_test -> touched once at the end with the selected checkpoint.
Dish-level prediction = mean of view-level predictions over all of the dish's views.
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (ConvNeXtBase, environment_info, eval_transform, regression_metrics,  # noqa: E402
                        seed_everything, sha256_file, sha256_lines, train_transform, write_json)
from src.n5k.data import (TargetScaler, ViewDataset, all_view_items, build_dish_table,  # noqa: E402
                          sample_epoch_items)
from src.n5k.metadata import TARGETS  # noqa: E402


@torch.no_grad()
def predict_views(model, items, targets, device, workers, batch=96):
    dl = DataLoader(ViewDataset(items, targets, eval_transform()), batch, num_workers=workers, pin_memory=True)
    model.eval()
    preds = np.zeros((len(items), len(TARGETS)), np.float32)
    for x, _, idx in dl:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            preds[idx.numpy()] = model(x).float().cpu().numpy()
    return preds


def dish_level(items, preds):
    acc = {}
    for (d, t, _), p in zip(items, preds):
        acc.setdefault(d, []).append(p)
    return {d: np.mean(v, 0) for d, v in acc.items()}


def evaluate(df, items, view_preds_z, scaler, subset_name):
    """Metrics in original units + normalized loss; dish-level (all views) and per view type."""
    truth = df.set_index("dish_id")[TARGETS]
    res = {}
    groups = {"all_views": items, "overhead_only": [i for i in items if i[1] == "overhead"],
              "side_only": [i for i in items if i[1] == "side"]}
    pz = {id(it): p for it, p in zip(items, view_preds_z)}
    for g, its in groups.items():
        if not its:
            continue
        dl = dish_level(its, [pz[id(i)] for i in its])
        ids = sorted(dl)
        z_pred = np.stack([dl[d] for d in ids])
        y_true = truth.loc[ids].to_numpy(np.float32)
        y_pred = scaler.inverse(z_pred)
        z_true = scaler.transform(y_true)
        res[g] = dict(
            n_dishes=len(ids),
            normalized_mse_mean=float(((z_pred - z_true) ** 2).mean()),
            normalized_mse_per_target=dict(zip(TARGETS, ((z_pred - z_true) ** 2).mean(0).tolist())),
            metrics=regression_metrics(y_true, y_pred, TARGETS),
        )
    return {subset_name: res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n5k-root", required=True)
    ap.add_argument("--init", required=True, help="Food-101 best.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--views-per-dish", type=int, default=4)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr-mult", type=float, default=10.0)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop early, never touch test")
    a = ap.parse_args()

    seed_everything(a.seed)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    df, views = build_dish_table(a.n5k_root, a.val_frac, a.seed)
    df["n_views"] = [len(views[d]) for d in df.dish_id]
    df["used"] = (df.n_views > 0) & ~((df.split != "test") & df.implausible)
    df.to_csv(out / "dish_table.csv", index=False)
    fit_ids = df[(df.split == "fit") & df.used].dish_id.tolist()
    val_ids = df[(df.split == "val") & df.used].dish_id.tolist()
    test_df = df[(df.split == "test") & (df.n_views > 0)]
    assert not set(fit_ids) & set(val_ids)
    assert not set(df[df.split == "fit"].session) & set(df[df.split == "val"].session)

    scaler = TargetScaler.fit(df.set_index("dish_id").loc[fit_ids, TARGETS].to_numpy(np.float32))
    all_targets = {d: scaler.transform(np.asarray(r, np.float32))
                   for d, r in zip(df.dish_id, df[TARGETS].to_numpy(np.float32))}
    val_items = all_view_items(val_ids, views)

    write_json(out / "config.json", dict(
        args=vars(a), env=environment_info(), targets=TARGETS, target_scaler=scaler.to_dict(),
        init_checkpoint=a.init, init_sha256=sha256_file(a.init),
        n_dishes=dict(fit=len(fit_ids), val=len(val_ids), test=len(test_df),
                      test_plausible=int((~test_df.implausible).sum()),
                      dropped_implausible_fit_val=int(((df.split != "test") & df.implausible).sum()),
                      no_views=int((df.n_views == 0).sum())),
        n_val_views=len(val_items),
        fit_ids_sha256=sha256_lines(sorted(fit_ids)), val_ids_sha256=sha256_lines(sorted(val_ids)),
        loss="mean over targets of MSE on z-scored targets (scaler fitted on fit dishes)",
        selection_rule="lowest validation dish-level normalized MSE (mean over 4 targets)"))

    model = ConvNeXtBase(len(TARGETS))
    ck = torch.load(a.init, map_location="cpu")
    sd = {k: v for k, v in ck["model"].items() if not k.startswith("head.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert set(missing) == {"head.weight", "head.bias"} and not unexpected, (missing, unexpected)
    model = model.to(device).to(memory_format=torch.channels_last)

    head = list(model.head.parameters())
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    groups = []
    for ps, lr in ((body, a.lr), (head, a.lr * a.head_lr_mult)):
        groups += [dict(params=[p for p in ps if p.ndim > 1], lr=lr, weight_decay=a.wd),
                   dict(params=[p for p in ps if p.ndim <= 1], lr=lr, weight_decay=0.0)]
    opt = torch.optim.AdamW(groups)
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_epoch = len(fit_ids) * a.views_per_dish // a.batch
    total, warm = a.epochs * steps_per_epoch, int(a.warmup_epochs * steps_per_epoch)
    scaler_amp = torch.amp.GradScaler()
    mse = nn.MSELoss()

    history, best, step = [], math.inf, 0
    tf = train_transform()
    for ep in range(a.epochs):
        items = sample_epoch_items(fit_ids, views, a.views_per_dish, rng)
        dl = DataLoader(ViewDataset(items, all_targets, tf), a.batch, shuffle=True, drop_last=True,
                        num_workers=a.workers, pin_memory=True)
        model.train()
        t0, run, seen = time.time(), 0.0, 0
        for x, y, _ in dl:
            f = step / max(warm, 1) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * f
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(x)
            loss = mse(pred.float(), y)  # mean over batch and the 4 targets
            opt.zero_grad(set_to_none=True)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(opt)
            scaler_amp.update()
            run += loss.item() * len(y)
            seen += len(y)
            step += 1
            if a.max_steps and step >= a.max_steps:
                break
        vp = predict_views(model, val_items, all_targets, device, a.workers)
        v = evaluate(df, val_items, vp, scaler, "val")["val"]
        vloss = v["all_views"]["normalized_mse_mean"]
        rec = dict(epoch=ep, train_loss=run / max(seen, 1), val_loss=vloss,
                   val_per_target=v["all_views"]["normalized_mse_per_target"],
                   val_r2={k: m["r2"] for k, m in v["all_views"]["metrics"].items()},
                   val_mae={k: m["mae"] for k, m in v["all_views"]["metrics"].items()},
                   sec=time.time() - t0)
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if vloss < best:
            best = vloss
            torch.save(dict(model=model.state_dict(), epoch=ep, val_loss=vloss, target_scaler=scaler.to_dict(),
                            targets=TARGETS), out / "best.pt")
        write_json(out / "history.json", history)
        if a.max_steps and step >= a.max_steps:
            return

    ck = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    res = dict(selected_epoch=ck["epoch"], val_loss=ck["val_loss"])
    vp = predict_views(model, val_items, all_targets, device, a.workers)
    res.update(evaluate(df, val_items, vp, scaler, "val"))
    # final, single use of the official test split
    test_items = all_view_items(test_df.dish_id.tolist(), views)
    tp = predict_views(model, test_items, all_targets, device, a.workers)
    res.update(evaluate(df, test_items, tp, scaler, "test_official_all"))
    plaus = set(test_df[~test_df.implausible].dish_id)
    keep = [i for i, it in enumerate(test_items) if it[0] in plaus]
    res.update(evaluate(df, [test_items[i] for i in keep], tp[keep], scaler, "test_plausible"))
    write_json(out / "eval_metrics.json", res)
    dl = dish_level(test_items, tp)
    ids = sorted(dl)
    import pandas as pd
    pd.DataFrame(scaler.inverse(np.stack([dl[d] for d in ids])), columns=[f"pred_{t}" for t in TARGETS]) \
        .assign(dish_id=ids).merge(df[["dish_id", "implausible", *TARGETS]], on="dish_id") \
        .to_csv(out / "test_dish_predictions.csv", index=False)
    print("DONE", json.dumps({k: v for k, v in res.items() if k.startswith("test")}["test_official_all"]["all_views"]["metrics"]), flush=True)


if __name__ == "__main__":
    main()
