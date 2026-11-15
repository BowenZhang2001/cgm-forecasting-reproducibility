"""Two-layer LSTM with a linear forecasting head on the final hidden state."""

from dataclasses import asdict

import torch
from torch import nn

from cgm_pipeline.elastic_net import CONTEXTS, HORIZONS
from cgm_pipeline.neural_training import NeuralSettings, NeuralTrainer, seed_runtime


class LSTMForecaster(nn.Module):
    def __init__(self, *, context_hours, horizon_minutes):
        super().__init__()
        if context_hours not in CONTEXTS or horizon_minutes not in HORIZONS:
            raise ValueError("invalid LSTM task")
        self.context_points = CONTEXTS[context_hours]
        self.horizon_points = HORIZONS[horizon_minutes]
        self.recurrent = nn.LSTM(input_size=1, hidden_size=128, num_layers=2,
                                 dropout=.2, batch_first=True, bias=True,
                                 bidirectional=False, proj_size=0)
        self.head = nn.Linear(128, self.horizon_points, bias=True)

    def forward(self, context):
        if (context.ndim != 3 or context.shape[1:] != (self.context_points, 1) or
                not len(context) or not torch.isfinite(context).all()):
            raise ValueError("LSTM requires finite (origins, task-context, 1) input")
        # h/c start at zero for every window; no carried state or future decoder.
        _, (hidden, _) = self.recurrent(context)
        return self.head(hidden[-1])

    @property
    def architecture(self):
        return {"input_size": 1, "hidden_size": 128, "num_layers": 2, "dropout": .2,
                "dropout_placement": "between recurrent layers only", "batch_first": True,
                "bias": True, "bidirectional": False, "proj_size": 0,
                "representation": "final hidden state of second recurrent layer",
                "head": "Linear(128, horizon_points), bias=True",
                "horizon_points": self.horizon_points, "context_points": self.context_points,
                "initial_state": "zero independently for each window",
                "initialization": "PyTorch LSTM/Linear defaults; no forget-bias override"}


class LSTMTask(NeuralTrainer):
    def __init__(self, *, normalization, context_hours, horizon_minutes,
                 settings, provenance, device="cpu"):
        seed_runtime(settings)
        model = LSTMForecaster(context_hours=context_hours, horizon_minutes=horizon_minutes)
        super().__init__(model, normalization=normalization, context_hours=context_hours,
                         horizon_minutes=horizon_minutes, settings=settings,
                         architecture=model.architecture, provenance=provenance, model_name="LSTM", device=device)

    @classmethod
    def load_checkpoint(cls, source, *, normalization, context_hours, horizon_minutes,
                        device="cpu", resume=False):
        payload = torch.load(source, map_location="cpu", weights_only=True)
        m = payload["manifest"]
        if (m.get("schema_version") != 1 or m.get("model") != "LSTM" or
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
