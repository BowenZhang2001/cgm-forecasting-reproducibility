"""Zero-shot and fixed-learning-rate adaptation of the four general TSFMs."""

import argparse
import json
from pathlib import Path
import math
import random

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from cgm_pipeline.benchmark_inputs import MODEL_SEEDS, SCOPES, load_scope
from cgm_pipeline.chronos_gradient_retry import POLICY, finite_batch, gradient_failures
from cgm_pipeline.chronos_primitives import (
    _finite_native_scaling, check_finite_optimizer, install_fp32_output_head, optimizer_batch,
)
from cgm_pipeline.formal_training import digest, file_digest
from cgm_pipeline.reproduce_baselines import CONFIG, configs, export_predictions
from cgm_pipeline.training_data import CONTEXTS, HORIZONS
from cgm_pipeline.validation_metrics import ParticipantMacroRMSE


SPECS = json.loads((CONFIG / "tsfm_pretrained.json").read_text())


def snapshot(name, cache):
    spec = SPECS[name]
    for filename, key in (("config.json", "config_sha256"),
                          ("model.safetensors", "checkpoint_sha256")):
        path = Path(hf_hub_download(repo_id=spec["repo_id"], revision=spec["revision"],
                                    filename=filename, cache_dir=cache))
        if file_digest(path) != spec[key]:
            raise ValueError(f"Released checkpoint checksum differs for {filename}")
    return path.parent


class Forecast:
    def __init__(self, name, source, hours, minutes, device, *, adaptation=False):
        self.name, self.device = name, torch.device(device)
        self.context, self.horizon = CONTEXTS[hours], HORIZONS[minutes]
        self.bolt = name.startswith("chronos_bolt")
        self.timesfm = name == "timesfm_2_5_rank8"
        self.config = {"execution": {}}
        if self.bolt:
            from chronos import ChronosBoltPipeline

            self.pipeline = ChronosBoltPipeline.from_pretrained(
                str(source), device_map=str(self.device), torch_dtype=torch.float32, local_files_only=True,
            )
            self.model = self.pipeline.model
            install_fp32_output_head(self.model)
            self.model.instance_norm.register_forward_hook(_finite_native_scaling)
            if name == "chronos_bolt_small" and self.device.type == "cuda":
                self.config["execution"]["gradient_retry"] = POLICY
        elif self.timesfm:
            from safetensors.torch import load_file
            from transformers import TimesFm2_5Config, TimesFm2_5ModelForPrediction

            config = TimesFm2_5Config.from_pretrained(source, local_files_only=True)
            config._attn_implementation = "sdpa"
            model = TimesFm2_5ModelForPrediction(config).float()
            weights = load_file(str(source / "model.safetensors"))
            weights = {key.replace(".mlp.ff0.", ".mlp.fc1.").replace(".mlp.ff1.", ".mlp.fc2."): value
                       for key, value in weights.items()}
            model.load_state_dict(weights, strict=True)
            if adaptation:
                from peft import LoraConfig, get_peft_model

                model = get_peft_model(model, LoraConfig(
                    r=8, lora_alpha=16, lora_dropout=.05, bias="none",
                    target_modules=SPECS[name]["lora_modules"], modules_to_save=None,
                ))
                if sum(p.numel() for p in model.parameters() if p.requires_grad) != 2457600:
                    raise ValueError("TimesFM adapter parameter count differs")
            self.model = model.to(self.device)
        else:
            from chronos import Chronos2Pipeline

            self.pipeline = Chronos2Pipeline.from_pretrained(
                str(source), device_map="cpu", torch_dtype=torch.float32, local_files_only=True,
            )
            self.model = self.pipeline.model.to(self.device)

    @property
    def amp(self):
        return self.bolt and self.device.type == "cuda"

    def native(self, x, y=None, *, final=False):
        if self.timesfm:
            arguments = {"truncate_negative": False, "window_size": None} if final else {}
            return self.model(past_values=list(x.unbind()), future_values=y,
                              forecast_context_len=math.ceil(self.context / 32) * 32, **arguments)
        if self.bolt:
            return self.model(context=x, target=y)
        return self.model(context=x, future_target=y,
                          num_output_patches=math.ceil(self.horizon / 16),
                          group_ids=torch.arange(len(x), device=self.device))

    @torch.inference_mode()
    def predict(self, context, _, *, final=True):
        self.model.eval()
        x = torch.as_tensor(np.array(context, dtype=np.float32, copy=True), device=self.device)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
            if self.bolt:
                _, point = self.pipeline.predict_quantiles(
                    x, prediction_length=self.horizon, quantile_levels=[.5], limit_prediction_length=True,
                )
            elif self.timesfm:
                point = self.native(x, final=final).mean_predictions[:, :self.horizon]
            elif final:
                forecasts, _ = self.pipeline.predict_quantiles(
                    x.cpu()[:, None, :], prediction_length=self.horizon, quantile_levels=[.5],
                    batch_size=64, cross_learning=False, limit_prediction_length=True,
                )
                point = torch.cat([forecast[0, :, 0][None, :] for forecast in forecasts])
            else:
                quantiles = self.model.config.chronos_config["quantiles"]
                point = self.native(x).quantile_preds[:, quantiles.index(.5), :self.horizon]
        result = point.cpu().numpy()
        if result.shape != (len(x), self.horizon) or not np.isfinite(result).all():
            raise FloatingPointError("Malformed or nonfinite native point forecast")
        return result


class FineTuning:
    def __init__(self, task, lr, output):
        self.task = task
        self.optimizer = torch.optim.AdamW(
            [p for p in task.model.parameters() if p.requires_grad],
            lr=lr, weight_decay=0., fused=not task.bolt, foreach=False,
        )
        self.scheduler = None
        if not task.bolt and not task.timesfm:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                                lambda step: max(0., 1. - step / 1000))
        self.scaler = torch.amp.GradScaler("cuda", init_scale=128., enabled=task.amp)
        self.retry_state = {"total_retries": 0}
        self.output = output
        self.steps = 0

    def step(self, x, y, rows):
        task = self.task
        task.model.train()
        values = torch.as_tensor(np.array(x, dtype=np.float32, copy=True), device=task.device)
        target = torch.as_tensor(np.array(y, dtype=np.float32, copy=True), device=task.device)

        def forward_backward():
            with torch.autocast(task.device.type, dtype=torch.float16, enabled=task.amp):
                loss = task.native(values, target).loss
            if loss is None or not torch.isfinite(loss).all():
                raise FloatingPointError("Nonfinite native loss")
            self.scaler.scale(loss).backward()
            return float(loss.detach())

        if task.config["execution"].get("gradient_retry"):
            def record_retry(record):
                with (self.output / "amp_retries.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")

            finite_batch(self, forward_backward, batch_sha256=digest([row["origin_id"] for row in rows]),
                         emit=record_retry, check_state=check_finite_optimizer)
        else:
            self.optimizer.zero_grad(set_to_none=True)
            forward_backward()
            self.scaler.unscale_(self.optimizer)
            missing, invalid = gradient_failures(task.model)
            if missing or invalid:
                raise FloatingPointError(f"Invalid gradients: missing={missing}, nonfinite={invalid}")
            if not task.bolt and not task.timesfm:
                torch.nn.utils.clip_grad_norm_(task.model.parameters(), 1., error_if_nonfinite=True)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        if self.scheduler:
            self.scheduler.step()
        self.steps += 1


def training_loop(task, lr, seed, scope, hours, minutes, train, validation, output):
    training = FineTuning(task, lr, output)
    x, y = train.task(hours, minutes)
    batch_size = 256 if task.timesfm else 512
    history = []
    best, stale = math.inf, 0
    best_path = output / "best.pt"
    for step in range(1, 1001):
        indices = optimizer_batch(len(x), seed=seed, step=step)
        training.step(x[indices], y[indices], [train.rows[i] for i in indices])
        if step % 100:
            continue
        metric = ParticipantMacroRMSE(
            scope=scope, horizon_minutes=minutes,
            expected_rows=[row for donor in validation for row in donor.rows],
        )
        for donor in validation:
            vx, vy = donor.task(hours, minutes)
            for begin in range(0, len(vx), batch_size):
                sl = slice(begin, begin + batch_size)
                rows = donor.rows[sl]
                metric.update(task.predict(vx[sl], rows, final=False), vy[sl], origin_rows=rows, unit="mg/dL")
        score = metric.result()["participant_macro_rmse_mgdl"]
        history.append({"step": step, "validation_rmse": score})
        if score < best:
            best, stale = score, 0
            parameters = {name: p.detach().cpu() for name, p in task.model.named_parameters() if p.requires_grad}
            state = parameters if task.timesfm else task.model.state_dict()
            temporary = best_path.with_suffix(".tmp")
            torch.save(state, temporary)
            temporary.replace(best_path)
        else:
            stale += 1
            if stale == 3:
                break
    selected = torch.load(best_path, weights_only=True, map_location="cpu")
    if task.timesfm:
        with torch.no_grad():
            for name, parameter in task.model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(selected[name])
    else:
        task.model.load_state_dict(selected)
    (output / "validation.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-cache", type=Path, required=True)
    parser.add_argument("--scope", nargs="+", choices=tuple(SCOPES), default=tuple(SCOPES))
    parser.add_argument("--model", choices=tuple(SPECS), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    source = snapshot(args.model, args.checkpoint_cache)
    learning_rates = configs("tsfm_learning_rates.csv", ("scope", "task"))
    for name in args.scope:
        scope = SCOPES[name]
        train, validation, _, _ = load_scope(args.work_root, scope)
        for hours in CONTEXTS:
            for minutes in HORIZONS:
                tag = f"{hours}h-{minutes}m"
                root = args.output / name / tag
                for seed in (None, *MODEL_SEEDS):
                    mode = "zs" if seed is None else "adapted"
                    output = root / f"{args.model}-{mode}" / str(seed or "deterministic")
                    output.mkdir(parents=True, exist_ok=False)
                    batch_size = 64 if args.model == "chronos2_small" else 512
                    if seed is not None:
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                    task = Forecast(args.model, source, hours, minutes, args.device,
                                    adaptation=seed is not None)
                    if seed is not None:
                        if not task.timesfm:
                            random.seed(seed)
                            np.random.seed(seed)
                            torch.manual_seed(seed)
                            torch.cuda.manual_seed_all(seed)
                        lr = float(learning_rates[name, tag][args.model])
                        training_loop(task, lr, seed, scope, hours, minutes, train, validation, output)
                    export_predictions(args.work_root, output, scope, hours, minutes, task.predict,
                                       batch_size=batch_size)
                    del task


if __name__ == "__main__":
    main()
