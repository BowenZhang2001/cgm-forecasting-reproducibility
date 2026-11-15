"""Train the CGM-specific benchmark models in uninterrupted runs."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn

from cgm_pipeline.benchmark_inputs import MODEL_SEEDS, SCOPES, load_scope
from cgm_pipeline.reproduce_baselines import CONFIG, export_predictions
from cgm_pipeline.training_data import CONTEXTS, HORIZONS
from cgm_pipeline.vendor.cruise.encoder import Encoder
from cgm_pipeline.vendor.cruise.gluformer import GluFormer


def released_files(cache):
    spec = json.loads((CONFIG / "cgm_pretrained.json").read_text())
    paths = {}
    for name, sha in spec["files"].items():
        path = Path(hf_hub_download(repo_id=spec["repo_id"], revision=spec["revision"],
                                    filename=name, cache_dir=cache))
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != sha:
                raise ValueError(f"Released file checksum differs: {name}")
        paths[name] = path
    return paths


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ForecastJEPA(nn.Module):
    def __init__(self, files, hours, horizon, seed):
        super().__init__()
        self.hours = hours
        config = json.loads(files["cgm_jepa/config.json"].read_text())
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
            self.encoder = Encoder(**config).float()
            self.encoder.load_state_dict(load_file(str(files["cgm_jepa/model.safetensors"])), strict=True)
            self.encoder.proj.requires_grad_(False)
            self.encoder.data_embedding.timefeature_embedding.requires_grad_(False)
            before_head = torch.get_rng_state()
            self.encoder.eval()
            with torch.no_grad():
                tokens = self.tokens(torch.full((1, hours * 12), 100., dtype=torch.float32))
            torch.set_rng_state(before_head)
            self.head = nn.Linear(math.prod(tokens.shape[1:]), horizon)
            self.encoder.train()

    def tokens(self, values):
        tokens, _ = self.encoder(values.contiguous().view(len(values), self.hours, 12),
                                 x_mark=None, mask=None)
        return tokens

    def forward(self, values):
        return self.head(self.tokens(values).flatten(1))


def encode(values):
    width = 280. / 278.
    return np.clip(np.floor((np.clip(values, 40., 320.) - 40.) / width), 0, 277).astype(np.int64)


def gluformer_forecast(model, values, horizon, device):
    sequence = torch.as_tensor(encode(values), dtype=torch.long, device=device)
    generated = []
    for _ in range(horizon):
        next_token = model(sequence)[:, -1, :].argmax(-1, keepdim=True)
        generated.append(next_token)
        sequence = torch.cat((sequence, next_token), dim=1)
    ids = torch.cat(generated, dim=1).cpu().numpy().astype(np.float64)
    return 40. + (ids + .5) * (280. / 278.)


def validation_score(predict, validation, hours, minutes, batch_size, *, jepa):
    keys = [(row["dataset"], row["participant_id"]) for donor in validation for row in donor.rows]
    people = {key: i for i, key in enumerate(sorted(set(keys)))}
    sums = np.zeros(len(people), dtype=np.float64)
    counts = np.zeros(len(people), dtype=np.int64)
    for donor in validation:
        x, y = donor.task(hours, minutes)
        for begin in range(0, len(x), batch_size):
            sl = slice(begin, begin + batch_size)
            indices = [people[row["dataset"], row["participant_id"]] for row in donor.rows[sl]]
            targets = y[sl].astype(np.float32).astype(np.float64) if jepa else y[sl]
            residuals = predict(x[sl]).astype(np.float64) - targets
            squared_errors = np.square(residuals).sum(axis=1)
            if jepa:
                np.add.at(sums, indices, squared_errors)
                np.add.at(counts, indices, residuals.shape[1])
            else:
                sums += np.bincount(indices, weights=squared_errors, minlength=len(people))
                counts += np.bincount(indices, minlength=len(people)) * residuals.shape[1]
    metrics = np.sqrt(sums / counts)
    return math.fsum(metrics) / len(people) if jepa else float(metrics.mean())


def train_model(name, files, train, validation, hours, minutes, seed, device, destination):
    jepa = name == "CGM-JEPA"
    x, y = train.task(hours, minutes)
    if jepa:
        model = ForecastJEPA(files, hours, HORIZONS[minutes], seed).to(device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=1e-4, weight_decay=1e-5, foreach=False, fused=False)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        batch_size, epochs = 64, math.ceil(20000 / math.ceil(len(x) / 64))
    else:
        seed_all(seed)
        model = GluFormer(vocab_size=278, embed_dim=96, nhead=6, num_layers=3,
                          max_seq_length=25000, dim_feedforward=192, dropout=0., pad_token=278)
        original = torch.load(files["baselines/gluformer.pt"], weights_only=True, map_location="cpu")
        model.load_state_dict(original["encoder"], strict=True)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False, fused=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=.99)
        tokens = encode(np.concatenate([x, y], axis=1))
        batch_size, epochs = 128, 50
    seed_all(seed)

    @torch.inference_mode()
    def predict(values):
        model.eval()
        if jepa:
            return model(torch.as_tensor(values, dtype=torch.float32, device=device)).cpu().numpy()
        return gluformer_forecast(model, values, HORIZONS[minutes], device)

    best, best_step, stale_checks, step = math.inf, 0, 0, 0
    history = []
    checkpoint = destination / "best.pt"
    stop = False
    for epoch in range(1, epochs + 1):
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(x))
        for begin in range(0, len(x), batch_size):
            indices = order[begin:begin + batch_size]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if jepa:
                values = torch.as_tensor(x[indices], dtype=torch.float32, device=device)
                targets = torch.as_tensor(y[indices], dtype=torch.float32, device=device)
                loss = F.mse_loss(model(values), targets)
            else:
                batch = torch.as_tensor(tokens[indices], dtype=torch.long, device=device)
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, 278), batch[:, 1:].reshape(-1), ignore_index=278)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            completed = begin + len(indices) == len(x)
            if jepa:
                due = step % (100 if step <= 10000 else 500) == 0
            else:
                due = completed and epoch % 5 == 0
            if due:
                score = validation_score(predict, validation, hours, minutes,
                                         256 if jepa else 128, jepa=jepa)
                if not math.isfinite(score):
                    raise FloatingPointError("Nonfinite participant-macro validation RMSE")
                if score < best:
                    best, best_step, stale_checks = score, step, 0
                    temporary = checkpoint.with_suffix(".tmp")
                    torch.save(model.state_dict(), temporary)
                    temporary.replace(checkpoint)
                else:
                    stale_checks += 1
                history.append({"update": step, "epoch": epoch, "validation_rmse": score})
                if jepa and (epoch > 1 or completed):
                    stop = stale_checks >= 10 if step <= 10000 else step - best_step >= 1000
            if jepa and (stop or step == 20000):
                stop = True
                break
        if stop:
            break
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    (destination / "training.json").write_text(json.dumps(
        {"updates": step, "best_update": best_step, "validation": history}, indent=2) + "\n")
    return lambda values, _: np.concatenate([
        predict(values[start:start + batch_size]) for start in range(0, len(values), batch_size)
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-cache", required=True, type=Path)
    parser.add_argument("--scope", nargs="+", choices=tuple(SCOPES), default=tuple(SCOPES))
    parser.add_argument("--models", nargs="+", choices=("CGM-JEPA", "Compact-GluFormer"),
                        default=("CGM-JEPA", "Compact-GluFormer"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    files = released_files(args.checkpoint_cache)
    for scope_name in args.scope:
        scope = SCOPES[scope_name]
        train, validation, _, _ = load_scope(args.work_root, scope)
        for hours in CONTEXTS:
            for minutes in HORIZONS:
                for name in args.models:
                    for seed in MODEL_SEEDS:
                        output = args.output / scope_name / f"{hours}h-{minutes}m" / name / str(seed)
                        output.mkdir(parents=True, exist_ok=False)
                        predict = train_model(name, files, train, validation, hours, minutes,
                                              seed, args.device, output)
                        export_predictions(args.work_root, output, scope, hours, minutes, predict)


if __name__ == "__main__":
    main()
