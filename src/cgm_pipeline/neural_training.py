"""Task-specific neural steps, audit metadata and inference on frozen scaling.

The caller controls validation, checkpoint selection, and early stopping.
Raw matrices and audit rows follow the existing Elastic Net interface.
"""

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import random

import numpy as np
import torch

from cgm_pipeline.elastic_net import CONTEXTS, HORIZONS


@dataclass(frozen=True)
class NeuralSettings:
    seed: int
    learning_rate: float
    weight_decay: float
    micro_batch_size: int
    accumulation_steps: int
    deterministic: bool
    betas: tuple = (.9, .999)
    eps: float = 1e-8

    def __post_init__(self):
        if (not isinstance(self.seed, int) or not 0 <= self.seed < 2**32 or
                self.learning_rate not in (1e-4, 3e-4, 1e-3) or
                not np.isfinite(self.weight_decay) or self.weight_decay < 0 or
                not isinstance(self.micro_batch_size, int) or self.micro_batch_size < 1 or
                not isinstance(self.accumulation_steps, int) or self.accumulation_steps < 1 or
                self.micro_batch_size * self.accumulation_steps > 64 or
                len(self.betas) != 2 or not all(0 <= b < 1 for b in self.betas) or
                not np.isfinite(self.eps) or self.eps <= 0 or
                not isinstance(self.deterministic, bool)):
            raise ValueError("invalid neural-training settings")


def seed_runtime(settings):
    """Explicit process-wide RNG/backend configuration before model creation."""
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.use_deterministic_algorithms(settings.deterministic)
    torch.backends.cudnn.deterministic = settings.deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def horizon_mse(prediction, target, horizon_points):
    """Mean over every origin and every step of this task's horizon only."""
    if (prediction.ndim != 2 or prediction.shape != target.shape or
            prediction.shape[1] != horizon_points or not len(prediction)):
        raise ValueError("loss requires matching nonempty task-specific target matrices")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise FloatingPointError("nonfinite training prediction or target")
    return (prediction.float() - target.float()).square().mean()


def environment_identity(device):
    return {"versions": {name: str(importlib.metadata.version(name)) for name in
                         ("torch", "numpy", "scipy", "scikit-learn")},
            "cuda_runtime": torch.version.cuda, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cudnn": torch.backends.cudnn.version(), "cpu_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}


@dataclass(frozen=True)
class NeuralPredictions:
    predictions: np.ndarray
    origin_rows: tuple
    unit: str = "mg/dL"


class NeuralTrainer:
    """One task/scope, finite-gradient steps, exclusive checkpoints and inference.

    The caller supplies an initialized model and recorded settings. This class
    implements no epoch loop or validation-dependent decisions.
    """

    def __init__(self, model, *, normalization, context_hours, horizon_minutes,
                 settings, architecture, provenance, model_name, device="cpu"):
        if context_hours not in CONTEXTS or horizon_minutes not in HORIZONS:
            raise ValueError("invalid forecasting task")
        self.normalization, self.settings = normalization, settings
        self.context_hours, self.horizon_minutes = context_hours, horizon_minutes
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("only verified CPU/CUDA backends are supported")
        self.model = model.to(device=self.device, dtype=torch.float32)
        self.architecture = dict(architecture)
        self.model_name = model_name
        # Only JSON-safe provenance is allowed in weights-only checkpoints.
        self.provenance = json.loads(json.dumps(provenance, allow_nan=False))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=settings.learning_rate,
            weight_decay=settings.weight_decay, betas=settings.betas, eps=settings.eps,
            amsgrad=False, foreach=False, fused=False, maximize=False, capturable=False)
        self.steps, self.step_audit, self.failed = 0, [], False
        self.audit_mode, self.audit_chain_sha256, self.audit_sink = "full", "", None
        self.environment = environment_identity(self.device)

    def _values(self, values, width, name):
        raw = np.asarray(values, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != width or not len(raw) or not np.isfinite(raw).all():
            raise ValueError(f"{name} requires finite nonempty (origins, {width}) raw mg/dL values")
        scaled = np.asarray(self.normalization.transform(raw), dtype=np.float32)
        if not np.isfinite(scaled).all():
            raise ValueError("nonfinite normalized input")
        return torch.from_numpy(scaled)

    def context_tensor(self, context):
        """Only glucose context is accepted; no target, time or static covariate."""
        return self._values(context, CONTEXTS[self.context_hours], "context").unsqueeze(-1)

    def attach_audit_sink(self, sink):
        """Opt-in external audit records; retain only last step and a cumulative chain.

        Restored external-audit checkpoints require reattachment before updates.
        """
        if not callable(sink) or (self.audit_mode == "full" and self.steps):
            raise ValueError("attach an audit sink before the first update")
        self.audit_mode, self.audit_sink = "external", sink

    def train_step(self, context, target, *, origin_rows, loss_denominator=None):
        if self.failed:
            raise RuntimeError("cannot continue training after a numerical failure")
        if self.audit_mode == "external" and self.audit_sink is None:
            raise RuntimeError("resume requires the external audit sink")
        x = self.context_tensor(context)
        y = self._values(target, HORIZONS[self.horizon_minutes], "target")
        if len(x) != len(y):
            raise ValueError("context and target rows disagree")
        rows = self.normalization.scope.check_rows(origin_rows, len(x), training=True)
        if len(x) > self.settings.micro_batch_size * self.settings.accumulation_steps:
            raise ValueError("step exceeds declared effective batch size")
        denominator = len(x) if loss_denominator is None else loss_denominator
        if (not isinstance(denominator, int) or denominator < len(x) or
                denominator > self.settings.micro_batch_size*self.settings.accumulation_steps):
            raise ValueError("invalid explicit optimizer-loss denominator")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total, micro_batches = 0., 0
        try:
            for start in range(0, len(x), self.settings.micro_batch_size):
                stop = min(start + self.settings.micro_batch_size, len(x))
                prediction = self.model(x[start:stop].to(self.device))
                loss = horizon_mse(prediction, y[start:stop].to(self.device), HORIZONS[self.horizon_minutes])
                # Weight each origin equally; optional nominal denominator scales a final short step.
                weighted = loss * ((stop-start) / denominator)
                if not torch.isfinite(weighted):
                    raise FloatingPointError("nonfinite training objective")
                weighted.backward()
                total += float(weighted.detach().cpu())
                micro_batches += 1
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.model.parameters() if p.requires_grad):
                raise FloatingPointError("missing/nonfinite gradient; update rejected")
            self.optimizer.step()
            if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                raise FloatingPointError("nonfinite updated parameters; trainer stopped")
        except Exception:
            self.failed = True
            raise
        self.steps += 1
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest()
        audit = {"step": self.steps, "origin_rows_sha256": digest, "origins": len(rows),
                 "datasets": sorted({r["dataset"] for r in rows}),
                 "participants": len({(r["dataset"], r["participant_id"]) for r in rows}),
                 "micro_batches": micro_batches, "actual_effective_batch_size": len(x),
                 "normalized_training_mse": total*denominator/len(x),
                 "optimizer_loss": total, "loss_denominator": denominator, "finite_gradients": True}
        chain = hashlib.sha256((self.audit_chain_sha256 +
            json.dumps(audit, sort_keys=True, allow_nan=False)).encode()).hexdigest()
        if self.audit_mode == "external":
            try:
                self.audit_sink(dict(audit, audit_chain_sha256=chain))
            except Exception:
                self.failed = True
                raise
            self.step_audit[:] = [audit]
        else:
            self.step_audit.append(audit)
        self.audit_chain_sha256 = chain
        return dict(audit)

    def predict(self, context, *, origin_rows, batch_size=64):
        if self.failed:
            raise RuntimeError("failed trainer cannot generate forecasts")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("inference batch size must be positive")
        # Validate before splitting; duplicate or mixed-selection rows must fail.
        raw = np.asarray(context, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != CONTEXTS[self.context_hours] or not len(raw) or not np.isfinite(raw).all():
            raise ValueError("invalid task context")
        rows = self.normalization.scope.check_rows(origin_rows, len(raw), training=False)
        was_training = self.model.training
        self.model.eval()
        predictions = []
        try:
            with torch.inference_mode():
                for start in range(0, len(raw), batch_size):
                    x = self.context_tensor(raw[start:start+batch_size]).to(self.device)
                    z = self.model(x).float().cpu().numpy()
                    if z.shape != (len(x), HORIZONS[self.horizon_minutes]) or not np.isfinite(z).all():
                        raise FloatingPointError("invalid forecast; no fallback")
                    predictions.append(self.normalization.inverse(z).astype(np.float32))
        finally:
            self.model.train(was_training)
        result = np.concatenate(predictions)
        if not np.isfinite(result).all():
            raise FloatingPointError("nonfinite inverse-transformed forecast")
        return NeuralPredictions(result, rows)

    @property
    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.model.parameters()),
                "trainable": sum(p.numel() for p in self.model.parameters() if p.requires_grad)}

    def manifest(self):
        return {"schema_version": 1, "model": self.model_name, "architecture": self.architecture,
                "context_hours": self.context_hours, "horizon_minutes": self.horizon_minutes,
                "normalization": asdict(self.normalization), "settings": asdict(self.settings),
                "optimizer": {"family": "AdamW", "learning_rate": self.settings.learning_rate,
                    "weight_decay": self.settings.weight_decay, "betas": list(self.settings.betas),
                    "eps": self.settings.eps, "amsgrad": False, "foreach": False, "fused": False,
                    "maximize": False, "capturable": False},
                "scheduler": None, "gradient_clipping": None, "precision": "float32",
                "loss": "task-horizon normalized MSE, mean over origins and steps",
                "parameter_counts": self.parameter_counts, "environment": self.environment,
                "provenance": self.provenance, "steps": self.steps, "step_audit": self.step_audit,
                "step_audit_mode": self.audit_mode, "audit_chain_sha256": self.audit_chain_sha256,
                "checkpoint_role": self.provenance.get("checkpoint_role", "implementation_check; no validation selection")}

    def save_checkpoint(self, destination):
        """Save own weights exclusively under work, or to an in-memory buffer."""
        if self.failed or any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise RuntimeError("cannot checkpoint a failed/nonfinite model")
        numpy_state = np.random.get_state()
        state = {"manifest": self.manifest(),
                 "state_dict": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                 "optimizer_state": self.optimizer.state_dict(),
                 "rng": {"torch_cpu": torch.get_rng_state(), "python": random.getstate(),
                     "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
                     "torch_cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []}}
        buffer = io.BytesIO()
        torch.save(state, buffer)
        data = buffer.getvalue()
        if isinstance(destination, (str, Path)):
            path = Path(destination).resolve()
            with path.open("xb") as output:
                output.write(data)
        else:
            destination.write(data)
        return hashlib.sha256(data).hexdigest()

    def restore_state(self, payload, *, resume=False):
        self.model.load_state_dict(payload["state_dict"], strict=True)
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError("checkpoint contains nonfinite parameters")
        self.steps = payload["manifest"]["steps"]
        self.step_audit = payload["manifest"]["step_audit"]
        self.audit_mode = payload["manifest"].get("step_audit_mode", "full")
        self.audit_chain_sha256 = payload["manifest"].get("audit_chain_sha256", "")
        if not self.audit_chain_sha256:
            for audit in self.step_audit:
                self.audit_chain_sha256 = hashlib.sha256((self.audit_chain_sha256 +
                    json.dumps(audit, sort_keys=True, allow_nan=False)).encode()).hexdigest()
        if resume:
            self.optimizer.load_state_dict(payload["optimizer_state"])
            rng = payload["rng"]
            torch.set_rng_state(rng["torch_cpu"])
            random.setstate(rng["python"])
            n = rng["numpy"]
            np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
            if self.device.type == "cuda":
                if not rng["torch_cuda"]:
                    raise ValueError("CUDA resume requires the recorded CUDA RNG state")
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
        self.model.eval()
