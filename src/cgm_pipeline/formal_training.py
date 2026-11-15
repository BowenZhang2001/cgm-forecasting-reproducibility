"""Continuous baseline training and validation selection."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from cgm_pipeline.validation_metrics import ParticipantMacroRMSE


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                   separators=(',', ':')).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def epoch_batches(count, *, seed, epoch, batch_size):
    if count < 1 or epoch < 1 or batch_size < 1 or seed < 0:
        raise ValueError('invalid epoch geometry/seed')
    # Epoch-addressable shuffling does not depend on unrelated task RNG progress.
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(count)
    for start in range(0, count, batch_size):
        yield order[start:start+batch_size]


def verify_training_index(scope, observed_rows, expected_rows):
    """Require the complete pinned sampled index, including participant identities."""
    observed = scope.check_rows(observed_rows, len(observed_rows), training=True)
    expected = scope.check_rows(expected_rows, len(expected_rows), training=True)
    def index(rows):
        return {r['origin_id']:(r['dataset'],r['participant_id'],r.get('natural_segment_id')) for r in rows}
    if not expected or index(observed) != index(expected):
        raise ValueError('training origin index differs from pinned sampled index')
    return digest(index(expected))


def train_epoch(task, context, target, *, origin_rows, seed, epoch, partial_policy):
    """All training origins once; equal weights, no donor/participant rebalancing.

    A partial batch uses its actual size for the mean loss.
    """
    if partial_policy != 'actual':
        raise ValueError('partial batches must use their actual size for mean loss')
    rows = task.normalization.scope.check_rows(origin_rows, len(context), training=True)
    if len(target) != len(context):
        raise ValueError('training rows disagree')
    size = task.settings.micro_batch_size*task.settings.accumulation_steps
    total, steps, seen = 0., 0, []
    for ids in epoch_batches(len(rows), seed=seed, epoch=epoch, batch_size=size):
        audit = task.train_step(context[ids], target[ids], origin_rows=[rows[i] for i in ids],
                               loss_denominator=None)
        total += audit['normalized_training_mse']*len(ids)
        seen.extend(rows[i]['origin_id'] for i in ids)
        steps += 1
    if len(set(seen)) != len(rows):
        raise RuntimeError('epoch origin coverage mismatch')
    return {'epoch':epoch, 'origins':len(rows), 'steps':steps,
            'origin_order_sha256':digest(seen), 'normalized_training_mse':total/len(rows),
            'partial_policy':partial_policy, 'final_batch_size':len(ids)}


@dataclass
class EarlyStopping:
    max_epochs: int
    patience: int
    min_delta_mgdl: float
    best_score: float | None = None
    best_epoch: int | None = None
    stale_checks: int = 0
    last_epoch: int = 0
    patience_reference_score: float | None = None

    def __post_init__(self):
        if not 1 <= self.max_epochs <= 100 or not 1 <= self.patience <= 10 or self.min_delta_mgdl < 0 or not math.isfinite(self.min_delta_mgdl):
            raise ValueError('invalid explicit stopping policy')

    def observe(self, epoch, score):
        if epoch != self.last_epoch+1 or epoch > self.max_epochs or not math.isfinite(score) or score < 0:
            raise ValueError('requires consecutive complete finite validation checks')
        # Always retain the true minimum checkpoint, independent of patience tolerance.
        best = self.best_score is None or score < self.best_score
        significant = self.patience_reference_score is None or score < self.patience_reference_score-self.min_delta_mgdl
        if significant:
            self.patience_reference_score = score
        if best:
            self.best_score, self.best_epoch = score, epoch
        self.stale_checks = 0 if significant else self.stale_checks+1
        self.last_epoch = epoch
        return {'save_best':best, 'stop':self.stale_checks>=self.patience or epoch==self.max_epochs,
                'state':asdict(self)}


def validate_epoch(task, *, batches, expected_rows):
    """All validation origins once; targets go only to the metric, never prediction."""
    metric = ParticipantMacroRMSE(scope=task.normalization.scope,
        horizon_minutes=task.horizon_minutes, expected_rows=expected_rows)
    for context, target, rows, unit in batches():
        if unit != 'mg/dL':
            raise ValueError('validation batches must be raw mg/dL')
        checked = task.normalization.scope.check_rows(rows,len(context),training=False)
        if any(r['partition'] != 'validation' for r in checked):
            raise ValueError('reject test inputs before prediction')
        prediction = task.predict(context,origin_rows=rows)
        if prediction.origin_rows != tuple(rows):
            raise ValueError('prediction origin alignment changed')
        metric.update(prediction.predictions,target,origin_rows=rows,unit=prediction.unit)
    return metric.result()


def fit_neural_epochs(task, context, target, *, origin_rows, validation_batches,
                      validation_index, expected_training_index, stop_policy, partial_policy, on_epoch):
    """Explicit-policy loop; caller restores checkpoint and stopping state on resume.

    on_epoch must durably publish checkpoint/curve/audit artifacts before returning.
    The callback receives the checkpoint-selection and early-stopping flags.
    """
    verify_training_index(task.normalization.scope, origin_rows, expected_training_index)
    records = []
    if stop_policy.stale_checks >= stop_policy.patience:
        return records
    for epoch in range(stop_policy.last_epoch+1, stop_policy.max_epochs+1):
        train = train_epoch(task,context,target,origin_rows=origin_rows,
            seed=task.settings.seed,epoch=epoch,partial_policy=partial_policy)
        validation = validate_epoch(task,batches=validation_batches,expected_rows=validation_index)
        decision = stop_policy.observe(epoch,validation['participant_macro_rmse_mgdl'])
        record = {'training':train,'validation':validation,'decision':decision}
        on_epoch(task,record)
        records.append(record)
        if decision['stop']:
            break
    return records
