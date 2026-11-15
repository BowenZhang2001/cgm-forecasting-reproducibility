"""Fixed official supervised PatchTST adaptation; global normalization only."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from cgm_pipeline.elastic_net import CONTEXTS, HORIZONS
from cgm_pipeline.neural_training import NeuralSettings, NeuralTrainer, seed_runtime
from cgm_pipeline.vendor.patchtst.PatchTST_backbone import PatchTST_backbone

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/patchtst_small.json"


def fixed_configuration():
    data = CONFIG_PATH.read_bytes()
    config = json.loads(data)
    for name, identity in config["sources"].items():
        source = Path(__file__).parent / "vendor/patchtst" / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != identity["adapted_sha256"]:
            raise ValueError("adapted upstream component identity mismatch")
    return config, hashlib.sha256(data).hexdigest()


class PatchTSTForecaster(nn.Module):
    def __init__(self, *, context_hours, horizon_minutes):
        super().__init__()
        if context_hours not in CONTEXTS or horizon_minutes not in HORIZONS:
            raise ValueError("invalid PatchTST task")
        self.context_points = CONTEXTS[context_hours]
        self.horizon_points = HORIZONS[horizon_minutes]
        config, digest = fixed_configuration()
        self.backbone = PatchTST_backbone(context_window=self.context_points,
            target_window=self.horizon_points, **config["backbone"])
        self._architecture = dict(config, config_sha256=digest,
            context_points=self.context_points, horizon_points=self.horizon_points,
            patch_count=(self.context_points-16)//8+2,
            representation="flatten embedding-major (D, patches) into Linear(D*patches, H)",
            attention="full visible-context scaled dot product; previous-layer pre-softmax residual scores; no mask",
            initialization="upstream PyTorch defaults; learned zeros PE is Uniform(-0.02,0.02)",
            internal_norm="post-residual BatchNorm1d(32), eps=1e-5, momentum=0.1, affine/running stats enabled",
            attention_scale="8**-0.5, one nontrainable scalar parameter per encoder layer")

    def forward(self, context):
        if (context.ndim != 3 or context.shape[1:] != (self.context_points, 1) or
                not len(context) or not torch.isfinite(context).all()):
            raise ValueError("PatchTST requires finite (origins, task-context, 1) input")
        return self.backbone(context.transpose(1, 2)).squeeze(1)

    @property
    def architecture(self):
        return json.loads(json.dumps(self._architecture))


class PatchTSTTask(NeuralTrainer):
    def __init__(self, *, normalization, context_hours, horizon_minutes,
                 settings, provenance, device="cpu"):
        seed_runtime(settings)
        model = PatchTSTForecaster(context_hours=context_hours, horizon_minutes=horizon_minutes)
        super().__init__(model, normalization=normalization, context_hours=context_hours,
                         horizon_minutes=horizon_minutes, settings=settings,
                         architecture=model.architecture, provenance=provenance, model_name="PatchTST", device=device)

    @classmethod
    def load_checkpoint(cls, source, *, normalization, context_hours, horizon_minutes,
                        device="cpu", resume=False):
        payload = torch.load(source, map_location="cpu", weights_only=True)
        m = payload["manifest"]
        if (m.get("schema_version") != 1 or m.get("model") != "PatchTST" or
                m.get("context_hours") != context_hours or m.get("horizon_minutes") != horizon_minutes or
                m.get("normalization") != asdict(normalization)):
            raise ValueError("checkpoint task/scope/normalization identity mismatch")
        settings = NeuralSettings(**m["settings"])
        task = cls(normalization=normalization, context_hours=context_hours,
                   horizon_minutes=horizon_minutes, settings=settings,
                   provenance=m["provenance"], device=device)
        if task.architecture != m["architecture"] or task.parameter_counts != m["parameter_counts"]:
            raise ValueError("checkpoint architecture or parameter counts mismatch")
        expected = task.manifest()
        for key in ("optimizer", "scheduler", "gradient_clipping", "precision", "loss"):
            if m.get(key) != expected[key]:
                raise ValueError(f"checkpoint {key} convention mismatch")
        task.restore_state(payload, resume=resume)
        return task
