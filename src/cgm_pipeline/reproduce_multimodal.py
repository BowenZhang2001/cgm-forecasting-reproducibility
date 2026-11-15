"""Train the fixed CGMacros image, nutrition and timing variants in ID and LOSO."""

import argparse
import json
import math
import os
from pathlib import Path
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from cgm_pipeline.benchmark_inputs import SCOPES, load_scope
from cgm_pipeline.cgmacros_multimodal_data import NUTRIENTS
from cgm_pipeline.reproduce_baselines import CONFIG
from cgm_pipeline.reproduce_multimodal_data import image_inventory, prepare, raw_sources, review_hashes
from cgm_pipeline.reproduce_tsfms import Forecast, snapshot, training_loop


BRANCHES = ("cgm", "time", "image", "nutrition")
HORIZONS = (6, 12, 24)
PARENT_SEED = 20261011
BACKBONE_SEED = 20261012


def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validation_score(forecast, data):
    residual = forecast.astype(np.float64) - data.target.astype(np.float64)
    people = data.people
    by_person = np.array([
        [np.sqrt(np.mean(residual[people == person, :h] ** 2)) for h in HORIZONS]
        for person in sorted(set(people))
    ])
    return float(by_person.mean(axis=0).mean())


def parent_checkpoint(work, output, source, device):
    path = output / "ai_readi_parent"
    path.mkdir()
    scope = SCOPES["primary"]
    train, validation, _, _ = load_scope(work, scope, full_training=True)
    task = Forecast("chronos_bolt_tiny", source, 24, 120, device)
    seed_torch(PARENT_SEED)
    training_loop(task, 3e-5, PARENT_SEED, scope, 24, 120, train, validation, path)
    return path / "best.pt"


@torch.inference_mode()
def base_features(model, context, device):
    model.eval()
    results, embeddings = [], []
    median = model.config.chronos_config["quantiles"].index(.5)
    for start in range(0, len(context), 64):
        values = torch.as_tensor(context[start:start + 64], dtype=torch.float32, device=device)
        hidden, scale, inputs, attention = model.encode(context=values, mask=torch.isfinite(values))
        decoded = model.decode(inputs, attention, hidden)
        native = model.output_patch_embedding(decoded).view(len(values), model.num_quantiles, 64)
        raw = model.instance_norm.inverse(native.view(len(values), -1), scale).view_as(native)
        results.append(raw[:, median, :24].cpu().numpy())
        embeddings.append(hidden[:, -1].cpu().numpy())
    return np.concatenate(results), np.concatenate(embeddings)


def fit_backbone(source, parent, data, output, device):
    task = Forecast("chronos_bolt_tiny", source, 24, 120, device)
    model = task.model
    model.load_state_dict(torch.load(parent, map_location="cpu", weights_only=True), strict=True)
    seed_torch(BACKBONE_SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.,
                                  foreach=False, fused=False)
    scaler = torch.amp.GradScaler("cuda", init_scale=128., enabled=device.type == "cuda")
    train = data["train"]
    batches = math.ceil(len(train.rows) / 64)
    checks = {math.ceil(batches * quarter / 4) for quarter in (1, 2, 3, 4)}
    best, stale = math.inf, 0
    checkpoint = output / "backbone.pt"
    history = []
    for epoch in range(20):
        order = np.random.default_rng(np.random.SeedSequence([BACKBONE_SEED, epoch])).permutation(len(train.rows))
        for step, begin in enumerate(range(0, len(order), 64), 1):
            selected = order[begin:begin + 64]
            x = torch.as_tensor(train.context[selected], dtype=torch.float32, device=device)
            y = torch.as_tensor(train.target[selected], dtype=torch.float32, device=device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model(context=x, mask=torch.isfinite(x), target=y,
                             target_mask=torch.isfinite(y)).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite CGMacros backbone loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            if step not in checks:
                continue
            val = data["validation"]
            forecast, _ = base_features(model, val.context, device)
            score = validation_score(forecast, val)
            history.append({"epoch": epoch + 1, "step_in_epoch": step, "validation_score": score})
            if score < best:
                best, stale = score, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale += 1
            if stale >= 8:
                break
        if stale >= 8:
            break
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.requires_grad_(False)
    (output / "backbone_validation.json").write_text(json.dumps(history, indent=2) + "\n")
    return {part: base_features(model, subset.context, device) for part, subset in data.items()}


def image_embeddings(root):
    arrays = {}
    for encoder in ("food101", "nutrition5k"):
        with np.load(root / f"{encoder}-features.npz", allow_pickle=False) as data:
            arrays[encoder] = dict(zip(data["sha256"], data["embedding"]))
    return arrays


def image_transform(meals, embeddings, *, loso):
    """Fit and scale the eight retained PCs on each unique training photograph."""
    hashes = dict.fromkeys(meal["image_sha256"] for meal in sorted(meals.values(), key=lambda m: m["meal_id"])
                           if meal["partition"] == "train" and meal["image_sha256"])
    x = np.stack([embeddings[key] for key in hashes])
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    center = x.mean(axis=0)
    centered = x - center
    if loso:
        internal = centered.mean(axis=0)
        _, _, vectors = np.linalg.svd(centered - internal, full_matrices=False)
        signs = np.sign(vectors[np.arange(len(vectors)), np.abs(vectors).argmax(axis=1)])
        vectors *= signs[:, None]
        components = vectors[:8].copy()
    else:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=8, svd_solver="full").fit(centered)
        components, internal = pca.components_, pca.mean_
    train = centered @ components.T - internal @ components.T
    mean = train.mean(axis=0, dtype=np.float64)
    sd = train.std(axis=0, dtype=np.float64)

    def transform(key):
        vector = embeddings[key]
        normalized = vector / np.linalg.norm(vector)
        scores = (normalized - center) @ components.T - internal @ components.T
        return ((scores - mean) / sd).astype(np.float32)

    return {key: transform(key) for key in {meal["image_sha256"] for meal in meals.values()} - {""}}


def modalities(data, meals, embeddings, *, loso):
    train_meals = {row["last_meal_id"] for row in data["train"].rows if row["nutrition_available"]}
    entries = {key: np.log1p([float(meal[n]) for n in NUTRIENTS])
               for key, meal in meals.items() if meal["nutrients_valid"]}
    train_nutrition = np.stack([entries[key] for key in sorted(train_meals)])
    mean, sd = train_nutrition.mean(0), train_nutrition.std(0)
    transforms = {encoder: image_transform(meals, raw, loso=loso)
                  for encoder, raw in embeddings.items()}
    result = {}
    for partition, part in data.items():
        sizes = {"time": (len(part.rows), 2), "nutrition": (len(part.rows), 6),
                 "food101": (len(part.rows), 9), "nutrition5k": (len(part.rows), 9)}
        values = {name: np.zeros(shape, dtype=np.float32) for name, shape in sizes.items()}
        for i, row in enumerate(part.rows):
            meal = meals.get(row["last_meal_id"])
            recent = False
            hours = 6.
            if meal:
                hours_since = (pd.Timestamp(row["origin_timestamp"]) -
                               pd.Timestamp(meal["original_meal_timestamp"])).total_seconds() / 3600
                recent = hours_since <= 6
                if recent:
                    hours = hours_since
            values["time"][i] = hours, recent
            if row["nutrition_available"]:
                values["nutrition"][i, :-1] = (entries[row["last_meal_id"]] - mean) / sd
                values["nutrition"][i, -1] = 1
            if row["image_candidate_available_pending_acquisition"] and meal["image_sha256"]:
                for encoder, vectors in transforms.items():
                    values[encoder][i, :-1] = vectors[meal["image_sha256"]]
                    values[encoder][i, -1] = 1
        result[partition] = values
    return result


class ResidualModel(nn.Module):
    def __init__(self, name, settings):
        super().__init__()
        self.width = settings["projection_dimension"]
        self.active = settings["modalities"]
        # M1 retains the inactive six-input time projection to preserve the original seeded initialization.
        for branch, size in zip(BRANCHES, (256, 6 if name == "M1" else 2, 9, 6)):
            projection = nn.Sequential(nn.Linear(size, self.width), nn.LayerNorm(self.width), nn.GELU())
            self.add_module(branch, projection)
            if branch not in self.active:
                projection.requires_grad_(False)
        width = 4 * self.width
        layers = [nn.LayerNorm(width)]
        for units in settings["hidden_units"]:
            layers.extend([nn.Linear(width, units), nn.GELU(), nn.Dropout(settings["dropout"])])
            width = units
        self.head = nn.Sequential(*layers, nn.Linear(width, 24))

    def forward(self, x):
        outputs = []
        for branch in BRANCHES:
            if branch not in self.active:
                out = x["cgm"].new_zeros((len(x["cgm"]), self.width))
            else:
                features = x[branch]
                out = getattr(self, branch)(features)
                if branch in ("image", "nutrition"):
                    out = torch.where(features[:, -1:].bool(), out, torch.zeros_like(out))
            outputs.append(out)
        return self.head(torch.cat(outputs, dim=1))


def participant_balanced_sample(ids, seed, epoch):
    groups = [np.flatnonzero(ids == person) for person in sorted(set(ids))]
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    selected = rng.integers(0, len(groups), size=len(ids))
    order = np.empty(len(ids), dtype=np.int64)
    for index, items in enumerate(groups):
        positions = np.flatnonzero(selected == index)
        order[positions] = items[rng.integers(0, len(items), size=len(positions))]
    return order


def fit_residual(name, settings, seed, data, base, all_features, residual_stats, output, device, *, loso):
    seed_torch(seed)
    model = ResidualModel(name, settings).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=settings["learning_rate"], weight_decay=settings["weight_decay"], foreach=False)
    mean, sd = residual_stats
    features = {
        part: {
            "cgm": torch.as_tensor(base[part][1], device=device),
            "time": torch.as_tensor(v["time"], device=device),
            "image": torch.as_tensor(v[settings.get("image_encoder", "food101")], device=device),
            "nutrition": torch.as_tensor(v["nutrition"], device=device),
        } for part, v in all_features.items()
    }
    base_predictions = {part: torch.as_tensor(pair[0], device=device) for part, pair in base.items()}
    residual = data["train"].target.astype(np.float64) - base["train"][0].astype(np.float64)
    target = torch.as_tensor(((residual - mean) / sd).astype(np.float32), device=device)
    batch_size = settings["batch_size"]
    checks = 1 if name == "M1" and not loso else 4
    patience = 10 if checks == 1 else 20
    batches = math.ceil(len(target) / batch_size)
    validation_steps = {math.ceil(batches * step / checks) for step in range(1, checks + 1)}

    @torch.inference_mode()
    def predict(partition):
        model.eval()
        return np.concatenate([
            (base_predictions[partition][i:i + 64] +
             model({key: value[i:i + 64] for key, value in features[partition].items()}) * sd + mean).cpu().numpy()
            for i in range(0, len(data[partition].rows), 64)
        ])

    path = output / "best.pt"
    best, stale = math.inf, 0
    history = []
    for epoch in range(100):
        order = participant_balanced_sample(data["train"].people, seed, epoch)
        for step, start in enumerate(range(0, len(order), batch_size), 1):
            indices = torch.as_tensor(order[start:start + batch_size], device=device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            predicted = model({key: value[indices] for key, value in features["train"].items()})
            loss = F.mse_loss(predicted, target[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite residual MLP loss")
            loss.backward()
            optimizer.step()
            if step not in validation_steps:
                continue
            score = validation_score(predict("validation"), data["validation"])
            history.append({"epoch": epoch + 1, "step_in_epoch": step, "validation_score": score})
            if score < best:
                best, stale = score, 0
                torch.save(model.state_dict(), path)
            else:
                stale += 1
            if stale >= patience:
                break
        if stale >= patience:
            break
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    (output / "validation.json").write_text(json.dumps(history, indent=2) + "\n")
    return predict("test")


def forecast_cohort(source, parent, sources, inventory, excluded, embeddings, output, held_out, device):
    output.mkdir(parents=True)
    data, meals, curves = prepare(sources, inventory, excluded, held_out)
    base = fit_backbone(source, parent, data, output, device)
    features = modalities(data, meals, embeddings, loso=held_out is not None)
    config = json.loads((CONFIG / "multimodal.json").read_text())
    residual = data["train"].target.astype(np.float64) - base["train"][0].astype(np.float64)
    stats = float(residual.mean()), float(residual.std())
    test = data["test"]
    pd.DataFrame([{**row, **curves.get(row["origin_id"], {})} for row in test.rows]).to_parquet(
        output / "test_origins.parquet", index=False)
    np.save(output / "test_target.npy", test.target, allow_pickle=False)
    np.save(output / "M0.npy", base["test"][0], allow_pickle=False)
    for name, settings in config["variants"].items():
        if name == "M0":
            continue
        for seed in config["residual_training"]["seeds"]:
            destination = output / name / str(seed)
            destination.mkdir(parents=True)
            predicted = fit_residual(name, settings, seed, data, base, features, stats,
                                     destination, device, loso=held_out is not None)
            np.save(destination / "prediction.npy", predicted, allow_pickle=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--cgmacros", required=True, type=Path)
    parser.add_argument("--image-qc", type=Path, default=CONFIG / "cgmacros_photo_exclusions.json",
                        help="Override the bundled meal-photo exclusion list")
    parser.add_argument("--image-encoder-root", required=True, type=Path)
    parser.add_argument("--checkpoint-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--analysis", choices=("id", "loso", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    sources = raw_sources(args.cgmacros)
    inventory = image_inventory(sources)
    excluded = review_hashes(args.image_qc)
    embeddings = image_embeddings(args.image_encoder_root)
    device = torch.device(args.device)
    source = snapshot("chronos_bolt_tiny", args.checkpoint_cache)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    parent = parent_checkpoint(args.work_root, args.output, source, device)
    settings = [(None, args.output / "ID")] if args.analysis in ("id", "all") else []
    if args.analysis in ("loso", "all"):
        settings += [(person, args.output / "LOSO" / person) for person in sources]
    for held_out, output in settings:
        forecast_cohort(source, parent, sources, inventory, excluded, embeddings, output, held_out, device)


if __name__ == "__main__":
    main()
