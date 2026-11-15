"""Streaming validation-only participant-macro RMSE in raw mg/dL."""
import math
import numpy as np
from cgm_pipeline.elastic_net import HORIZONS


class ParticipantMacroRMSE:
    """Pool squared errors within each participant and average the resulting RMSEs.

    Supply the complete expected validation index, separately from glucose values.
    Dataset-qualified participant keys prevent identically named donors merging.
    """
    def __init__(self, *, scope, horizon_minutes, expected_rows):
        if horizon_minutes not in HORIZONS:
            raise ValueError("invalid horizon")
        self.scope, self.width = scope, HORIZONS[horizon_minutes]
        self.expected = {}
        for row in expected_rows:
            checked = scope.check_rows([row], 1, training=False)[0]
            if checked['partition'] != 'validation':
                raise ValueError("metric accepts validation only")
            key = checked['origin_id']
            if key in self.expected:
                raise ValueError("duplicate expected origin")
            self.expected[key] = (checked['dataset'], checked['participant_id'])
        if not self.expected:
            raise ValueError("empty validation index")
        self.seen, self.sums = set(), {}

    def update(self, prediction, target, *, origin_rows, unit):
        if unit != 'mg/dL':
            raise ValueError("metric requires inverse-transformed mg/dL")
        p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
        if p.ndim != 2 or p.shape != y.shape or p.shape[1] != self.width or not len(p):
            raise ValueError("wrong validation geometry")
        rows = self.scope.check_rows(origin_rows, len(p), training=False)
        keys = [r['origin_id'] for r in rows]
        for r in rows:
            if (r['partition'] != 'validation' or r['origin_id'] in self.seen or
                    self.expected.get(r['origin_id']) != (r['dataset'], r['participant_id'])):
                raise ValueError("duplicate, unexpected or misaligned validation origin")
        with np.errstate(over='ignore', invalid='ignore'):
            squared = (p-y)**2
            per_row = squared.sum(axis=1)
        if not np.isfinite(p).all() or not np.isfinite(y).all() or not np.isfinite(per_row).all():
            raise FloatingPointError("nonfinite validation value/error; no exclusion")
        # Validate entire batch before modifying accumulator.
        for r, error in zip(rows, per_row):
            person = (r['dataset'], r['participant_id'])
            total, count = self.sums.get(person, (0., 0))
            self.sums[person] = (total+float(error), count+self.width)
        self.seen.update(keys)

    def result(self):
        if self.seen != set(self.expected):
            raise ValueError("incomplete validation origins; cannot select a checkpoint")
        scores = [math.sqrt(total/count) for _, (total,count) in sorted(self.sums.items())]
        score = math.fsum(scores)/len(scores)
        if not math.isfinite(score):
            raise FloatingPointError("nonfinite aggregate metric")
        return {'participant_macro_rmse_mgdl': score, 'participants': len(scores),
                'origins': len(self.seen), 'horizon_points': self.width}
