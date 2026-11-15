"""Task-specific, joint multi-response Elastic Net on pinned training scaling.

Objective: ||Y - 1 b^T - X B||_F^2/(2N)
  + lambda * (mixing * sum_j ||B[j, :]||_2 + (1-mixing)*||B||_F^2/2).
No predictor/response standardization beyond the supplied frozen global scale.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import time
import warnings

import numpy as np


CONTEXTS = {4: 48, 12: 144, 24: 288}
HORIZONS = {30: 6, 60: 12, 120: 24}
MIXING_GRID = (0., .25, .5, .75, 1.)
FOLDS = {"A": ("Anderson_2016", ("Brown_2019", "Lynch_2022")),
         "B": ("Brown_2019", ("Anderson_2016", "Lynch_2022")),
         "C": ("Lynch_2022", ("Anderson_2016", "Brown_2019"))}


@dataclass(frozen=True)
class TrainingScope:
    cohort: str
    fold: str | None = None

    def __post_init__(self):
        if self.cohort == "T1D":
            if self.fold not in FOLDS:
                raise ValueError("T1D requires a pinned fold")
        elif self.cohort != "Prediabetes & T2D" or self.fold is not None:
            raise ValueError("invalid cohort/fold")

    @property
    def development_datasets(self):
        return FOLDS[self.fold][1] if self.cohort == "T1D" else ("AI_READI_2025",)

    def check_rows(self, rows, size, *, training):
        rows = tuple(deepcopy(dict(r)) for r in rows)
        if len(rows) != size or len({r.get("origin_id") for r in rows}) != size:
            raise ValueError("origin rows must align and contain unique common-origin IDs")
        for r in rows:
            if (not r.get("origin_id") or not r.get("participant_id") or
                    r.get("cohort") != self.cohort or r.get("fold") != self.fold or
                    r.get("normalization") != "raw"):
                raise ValueError("scope or raw-input provenance mismatch")
            partition, dataset = r.get("partition"), r.get("dataset")
            if training and partition != "train":
                raise ValueError("fitting and lambda paths require training origins only")
            if partition in ("train", "validation", "internal_test"):
                allowed = self.development_datasets
            elif partition == "external_test" and not training:
                allowed = ((FOLDS[self.fold][0],) if self.cohort == "T1D" else
                           ("Broll_2021", "Hall_2018", "CGMacros_2025"))
            else:
                raise ValueError("invalid partition")
            if dataset not in allowed:
                raise ValueError("dataset violates cohort/fold boundaries")
        # A prediction batch is one isolated dataset/partition; training may pool
        # the two development T1D datasets while keeping each origin's identity.
        if not training and len({(r["dataset"], r["partition"]) for r in rows}) > 1:
            raise ValueError("prediction batches must keep dataset/partition boundaries")
        return rows


@dataclass(frozen=True)
class TrainingNormalization:
    scope: TrainingScope
    mean_mgdl: float
    std_mgdl: float
    statistics_sha256: str

    def __post_init__(self):
        if not np.isfinite(self.mean_mgdl) or not np.isfinite(self.std_mgdl) or self.std_mgdl <= 0:
            raise ValueError("invalid frozen global normalization")

    @classmethod
    def from_loader(cls, selection):
        """Bind the pinned loader's observed-training statistics to its scope."""
        if selection.partition != "train":
            raise ValueError("normalization must be obtained from a training selection")
        scope = TrainingScope(selection.cohort, selection.fold)
        if selection.dataset not in scope.development_datasets:
            raise ValueError("normalization donor is not a development dataset")
        stats = selection.normalization
        if (not stats or stats.get("source_partition") != "train" or
                stats.get("interpolated_values_included") is not False or
                stats.get("ddof") != 0 or stats.get("observed_training_positions", 0) < 2):
            raise ValueError("normalization is not observed training only")
        if scope.cohort == "T1D":
            if (set(stats.get("source_datasets", [])) != set(scope.development_datasets) or
                    stats.get("heldout_dataset") != FOLDS[scope.fold][0]):
                raise ValueError("normalization belongs to another T1D fold")
        elif stats.get("is_observed_required") is not True:
            raise ValueError("AI-READI statistics must require observed positions")
        digest = hashlib.sha256(json.dumps(stats, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return cls(scope, float(stats["mean_mgdl"]), float(stats["std_mgdl"]), digest)

    def transform(self, values):
        return (values - self.mean_mgdl) / self.std_mgdl

    def inverse(self, values):
        return values * self.std_mgdl + self.mean_mgdl


@dataclass(frozen=True)
class ElasticNetPredictions:
    predictions: np.ndarray
    origin_rows: tuple
    unit: str = "mg/dL"


@dataclass(frozen=True)
class PreparedElasticNetTraining:
    """One validated, normalized and centered task matrix shared by a path."""
    x_centered: np.ndarray
    y_centered: np.ndarray
    x_mean: np.ndarray
    y_mean: np.ndarray
    rows: tuple
    context_hours: int
    horizon_minutes: int
    normalization_sha256: str


class PathCandidateEstimator:
    """Minimal fitted-estimator interface for one immutable path snapshot."""
    def __init__(self, *, coef, intercept, alpha, l1_ratio,
                 n_iter=None, dual_gap=None, eps=None):
        self.coef_ = np.array(coef, dtype=np.float64, order="F", copy=True)
        self.intercept_ = np.array(intercept, dtype=np.float64, copy=True)
        self.alpha, self.l1_ratio = float(alpha), float(l1_ratio)
        self.n_iter_, self.dual_gap_, self.eps_ = n_iter, dual_gap, eps

    def predict(self, values):
        values = np.asarray(values, dtype=np.float64)
        return values @ self.coef_.T + self.intercept_


def _matrix(values, width, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != width or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite, nonempty (origins, {width}) matrix")
    return values


def _mixing(value):
    if value not in MIXING_GRID:
        raise ValueError("mixing must match one of the configured fractions")


def prepare_elastic_net_training(context, target, *, origin_rows, normalization,
                                 context_hours, horizon_minutes):
    """Validate, normalize and center a task exactly once in path-solver layout."""
    if context_hours not in CONTEXTS or horizon_minutes not in HORIZONS:
        raise ValueError("invalid task")
    x = _matrix(context, CONTEXTS[context_hours], "context")
    y = _matrix(target, HORIZONS[horizon_minutes], "target")
    if len(x) != len(y):
        raise ValueError("context/target rows disagree")
    rows = normalization.scope.check_rows(origin_rows, len(x), training=True)
    x = np.asarray(normalization.transform(x), dtype=np.float64)
    y = np.asarray(normalization.transform(y), dtype=np.float64)
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    return PreparedElasticNetTraining(
        x_centered=np.asfortranarray(x - x_mean),
        y_centered=np.asfortranarray(y - y_mean),
        x_mean=x_mean,
        y_mean=y_mean,
        rows=rows,
        context_hours=context_hours,
        horizon_minutes=horizon_minutes,
        normalization_sha256=normalization.statistics_sha256)


def _lambda_path_prepared(prepared, mixing):
    _mixing(mixing)
    x, y = prepared.x_centered, prepared.y_centered
    maximum = np.linalg.norm(x.T @ y, axis=1).max() / (len(x) * max(mixing, .001))
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("degenerate training correlation cannot define an automatic lambda path")
    ratio = .01 if len(x) < x.shape[1] else .0001
    return np.geomspace(maximum, maximum * ratio, 100)


def lambda_path(context, target, *, origin_rows, normalization,
                context_hours, horizon_minutes, mixing):
    """Historical glmnet default path convention, computed from training only.

    100 geometric points; min/max ratio .01 if N<P else .0001.
    The ridge path uses mixing=.001 solely to define a finite lambda_max.
    Its fitted objective remains exactly pure ridge (mixing=0).
    """
    prepared = prepare_elastic_net_training(context, target, origin_rows=origin_rows,
        normalization=normalization, context_hours=context_hours,
        horizon_minutes=horizon_minutes)
    return _lambda_path_prepared(prepared, mixing)


class ElasticNetTask:
    """One independently fitted context/horizon and cohort/fold specification."""

    def __init__(self, *, normalization, context_hours, horizon_minutes, mixing,
                 lambda_value, max_iter=100_000, tol=1e-7):
        _mixing(mixing)
        if context_hours not in CONTEXTS or horizon_minutes not in HORIZONS:
            raise ValueError("invalid task")
        if not np.isfinite(lambda_value) or lambda_value <= 0:
            raise ValueError("lambda must be positive and finite")
        if not isinstance(max_iter, int) or max_iter < 1 or not np.isfinite(tol) or tol <= 0:
            raise ValueError("invalid solver budget/tolerance")
        self.normalization = normalization
        self.context_hours, self.horizon_minutes = context_hours, horizon_minutes
        self.mixing, self.lambda_value = float(mixing), float(lambda_value)
        self.max_iter, self.tol = max_iter, tol

    def _fit_scaled(self, x, y, rows, *, previous=None):
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import MultiTaskElasticNet, Ridge
        if self.mixing == 0:
            estimator = Ridge(alpha=len(x) * self.lambda_value, fit_intercept=True,
                              solver="svd", copy_X=True)
        else:
            estimator = MultiTaskElasticNet(alpha=self.lambda_value, l1_ratio=self.mixing,
                                            fit_intercept=True, copy_X=True, max_iter=self.max_iter,
                                            tol=self.tol, selection="cyclic", warm_start=previous is not None)
            if previous is not None:
                estimator.coef_ = previous.coef_.copy()
                estimator.intercept_ = previous.intercept_.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(x, y)
        if (estimator.coef_.shape != (y.shape[1], x.shape[1]) or
                estimator.intercept_.shape != (y.shape[1],) or
                not np.isfinite(estimator.coef_).all() or not np.isfinite(estimator.intercept_).all()):
            raise InvalidCandidateFit("invalid fitted coefficients")
        self.estimator_, self.training_origin_rows_ = estimator, rows
        return self

    def fit(self, context, target, *, origin_rows):
        if hasattr(self, "estimator_"):
            raise ValueError("create a new task object for another fit")
        x = _matrix(context, CONTEXTS[self.context_hours], "context")
        y = _matrix(target, HORIZONS[self.horizon_minutes], "target")
        if len(x) != len(y):
            raise ValueError("context/target rows disagree")
        rows = self.normalization.scope.check_rows(origin_rows, len(x), training=True)
        return self._fit_scaled(self.normalization.transform(x), self.normalization.transform(y), rows)

    def predict(self, context, *, origin_rows):
        if not hasattr(self, "estimator_"):
            raise ValueError("task has not been fitted")
        x = _matrix(context, CONTEXTS[self.context_hours], "context")
        rows = self.normalization.scope.check_rows(origin_rows, len(x), training=False)
        predictions = self.normalization.inverse(self.estimator_.predict(self.normalization.transform(x)))
        if predictions.shape != (len(x), HORIZONS[self.horizon_minutes]) or not np.isfinite(predictions).all():
            raise RuntimeError("nonfinite/malformed forecast; no fallback")
        return ElasticNetPredictions(predictions, rows)

    @property
    def coefficient_counts(self):
        estimator = self.estimator_
        return {"possible_including_intercepts": int(estimator.coef_.size + estimator.intercept_.size),
                "nonzero_slopes": int(np.count_nonzero(estimator.coef_)),
                "intercepts": int(estimator.intercept_.size)}


def fit_lambda_path(context, target, *, origin_rows, normalization,
                    context_hours, horizon_minutes, mixing):
    """Yield independent model snapshots along one descending training-only path."""
    rows = tuple(origin_rows)
    path = lambda_path(context, target, origin_rows=rows, normalization=normalization,
                       context_hours=context_hours, horizon_minutes=horizon_minutes, mixing=mixing)
    x = normalization.transform(_matrix(context, CONTEXTS[context_hours], "context"))
    y = normalization.transform(_matrix(target, HORIZONS[horizon_minutes], "target"))
    rows = normalization.scope.check_rows(rows, len(x), training=True)
    previous = None
    for value in path:
        task = ElasticNetTask(normalization=normalization, context_hours=context_hours,
                              horizon_minutes=horizon_minutes, mixing=mixing, lambda_value=value)
        task._fit_scaled(x, y, rows, previous=previous)
        previous = task.estimator_
        yield task


class InvalidCandidateFit(RuntimeError):
    """A solver returned an object that cannot be considered for selection."""


def _diagnostic(value):
    if value is None:
        return None
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array.tolist()


def fit_lambda_path_audited_reference(context, target, *, origin_rows, normalization,
                                      context_hours, horizon_minutes, mixing,
                                      max_iter=100_000, retry_max_iter=200_000, tol=1e-7,
                                      on_attempt=None, on_candidate=None):
    """Fit a complete path while retaining failed candidates and bounded retries.

    A convergence warning gets one same-objective retry at 2x iterations, seeded
    from the preceding successful stronger-lambda solution. A candidate that
    still fails is recorded and excluded while the path continues from the last
    valid stronger-lambda snapshot. Data/schema/programming and infrastructure
    errors propagate.
    """
    from sklearn.exceptions import ConvergenceWarning

    rows = tuple(origin_rows)
    path = lambda_path(context, target, origin_rows=rows, normalization=normalization,
                       context_hours=context_hours, horizon_minutes=horizon_minutes,
                       mixing=mixing)
    x = normalization.transform(_matrix(context, CONTEXTS[context_hours], "context"))
    y = normalization.transform(_matrix(target, HORIZONS[horizon_minutes], "target"))
    rows = normalization.scope.check_rows(rows, len(x), training=True)
    previous = None
    results = []
    def coverage():
        attempted = len(results)
        successful = sum(x['status']=='completed' for x in results)
        failed = sum(x['status']=='failed' for x in results)
        complete = attempted == len(path)
        return dict(expected=len(path), attempted=attempted, successful=successful,
                    failed=failed, unattempted=len(path)-attempted, complete=complete,
                    selection_eligible=complete and successful > 0,
                    retries=sum(max(0,len(x['attempts'])-1) for x in results))
    def publish(item):
        results.append(item)
        if on_candidate is not None:
            public={k:v for k,v in item.items() if k!='task'}
            on_candidate(public,coverage())
    for index, value in enumerate(path):
        attempts = []
        budgets = [max_iter]
        for budget in budgets:
            task = ElasticNetTask(normalization=normalization, context_hours=context_hours,
                horizon_minutes=horizon_minutes, mixing=mixing, lambda_value=float(value),
                max_iter=budget, tol=tol)
            try:
                task._fit_scaled(x, y, rows, previous=previous)
            except ConvergenceWarning as exc:
                event=dict(index=index,mixing=float(mixing),lambda_value=float(value),
                    attempt=len(attempts)+1,max_iter=budget,status="convergence_warning",
                    error=f"{type(exc).__name__}: {exc}")
                attempts.append({k:v for k,v in event.items() if k not in ('index','mixing','lambda_value','attempt')})
                if on_attempt is not None: on_attempt(event)
                if budget == max_iter and retry_max_iter > max_iter:
                    budgets.append(retry_max_iter)
                    continue
                publish(dict(index=index, mixing=float(mixing), lambda_value=float(value),
                    status="failed", attempts=attempts, error_class="convergence_warning"))
                break
            except (InvalidCandidateFit, FloatingPointError, np.linalg.LinAlgError,
                    OverflowError) as exc:
                event=dict(index=index,mixing=float(mixing),lambda_value=float(value),
                    attempt=len(attempts)+1,max_iter=budget,status="numerical_failure",
                    error=f"{type(exc).__name__}: {exc}")
                attempts.append({k:v for k,v in event.items() if k not in ('index','mixing','lambda_value','attempt')})
                if on_attempt is not None: on_attempt(event)
                publish(dict(index=index, mixing=float(mixing), lambda_value=float(value),
                    status="failed", attempts=attempts, error_class="numerical_failure"))
                break
            else:
                estimator = task.estimator_
                fitted = estimator.predict(x)
                if fitted.shape != y.shape or not np.isfinite(fitted).all():
                    event=dict(index=index,mixing=float(mixing),lambda_value=float(value),
                        attempt=len(attempts)+1,max_iter=budget,status="numerical_failure",
                        error="nonfinite or malformed fitted values")
                    attempts.append({k:v for k,v in event.items() if k not in ('index','mixing','lambda_value','attempt')})
                    if on_attempt is not None: on_attempt(event)
                    publish(dict(index=index, mixing=float(mixing), lambda_value=float(value),
                        status="failed", attempts=attempts, error_class="numerical_failure"))
                    break
                dual_gap = getattr(estimator, "dual_gap_", None)
                eps = getattr(estimator, "eps_", None)
                n_iter = getattr(estimator, "n_iter_", None)
                if mixing > 0 and (dual_gap is None or eps is None or n_iter is None or
                        not np.isfinite(np.asarray(dual_gap)).all() or
                        not np.isfinite(np.asarray(eps)).all() or int(n_iter) < 1):
                    event=dict(index=index,mixing=float(mixing),lambda_value=float(value),
                        attempt=len(attempts)+1,max_iter=budget,status="numerical_failure",
                        error="nonfinite or missing convergence diagnostics")
                    attempts.append({k:v for k,v in event.items()
                        if k not in ('index','mixing','lambda_value','attempt')})
                    if on_attempt is not None: on_attempt(event)
                    publish(dict(index=index,mixing=float(mixing),lambda_value=float(value),
                        status="failed",attempts=attempts,error_class="numerical_failure"))
                    break
                diagnostics = dict(n_iter=_diagnostic(getattr(estimator, "n_iter_", None)),
                                   dual_gap=_diagnostic(dual_gap), eps=_diagnostic(eps))
                event=dict(index=index,mixing=float(mixing),lambda_value=float(value),
                    attempt=len(attempts)+1,max_iter=budget,status="converged",
                    diagnostics=diagnostics)
                attempts.append({k:v for k,v in event.items() if k not in ('index','mixing','lambda_value','attempt')})
                if on_attempt is not None: on_attempt(event)
                previous = estimator
                publish(dict(index=index, mixing=float(mixing), lambda_value=float(value),
                    status="completed", attempts=attempts, task=task, diagnostics=diagnostics))
                break
    return dict(results=results, coverage=coverage())


def _fitted_values_are_finite(prepared, estimator, *, batch_size=4096):
    """Check all training fitted values without allocating the complete matrix."""
    for start in range(0, len(prepared.x_centered), batch_size):
        stop = min(start + batch_size, len(prepared.x_centered))
        fitted = prepared.x_centered[start:stop] @ estimator.coef_.T + prepared.y_mean
        if fitted.shape != (stop - start, prepared.y_centered.shape[1]) or not np.isfinite(fitted).all():
            return False
    return True


def _path_task(*, prepared, normalization, context_hours, horizon_minutes,
               mixing, lambda_value, max_iter, tol, coef, n_iter, dual_gap, eps):
    coef = np.asarray(coef, dtype=np.float64)
    intercept = prepared.y_mean - prepared.x_mean @ coef.T
    estimator = PathCandidateEstimator(coef=coef, intercept=intercept,
        alpha=lambda_value, l1_ratio=mixing, n_iter=n_iter,
        dual_gap=dual_gap, eps=eps)
    if (estimator.coef_.shape != (prepared.y_centered.shape[1], prepared.x_centered.shape[1]) or
            estimator.intercept_.shape != (prepared.y_centered.shape[1],) or
            not np.isfinite(estimator.coef_).all() or
            not np.isfinite(estimator.intercept_).all()):
        raise InvalidCandidateFit("invalid fitted coefficients")
    task = ElasticNetTask(normalization=normalization, context_hours=context_hours,
        horizon_minutes=horizon_minutes, mixing=mixing, lambda_value=lambda_value,
        max_iter=max_iter, tol=tol)
    task.estimator_, task.training_origin_rows_ = estimator, prepared.rows
    return task


def _positive_path_segment(prepared, alphas, *, mixing, max_iter, tol, coef_init):
    """Run sklearn's supported multi-output path and retain emitted warnings."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import enet_path

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        started = time.perf_counter()
        solved_alphas, coefs, dual_gaps, n_iters = enet_path(
            prepared.x_centered, prepared.y_centered, l1_ratio=mixing,
            alphas=np.asarray(alphas, dtype=np.float64), precompute=False,
            copy_X=False, coef_init=coef_init, verbose=False, return_n_iter=True,
            positive=False, check_input=False, max_iter=max_iter, tol=tol,
            selection="cyclic", do_screening=True)
        fit_seconds = time.perf_counter() - started
    if not np.array_equal(solved_alphas, np.asarray(alphas, dtype=np.float64)):
        raise RuntimeError("path solver returned unexpected lambda values or order")
    unexpected = [w for w in caught if not issubclass(w.category, ConvergenceWarning)]
    if unexpected:
        raise RuntimeError(f"unexpected path warning: {unexpected[0].message}")
    return dict(coefs=coefs, dual_gaps=np.asarray(dual_gaps),
        n_iters=np.asarray(n_iters, dtype=np.int64),
        warnings=[str(w.message) for w in caught], fit_seconds=fit_seconds)


def _ridge_path(prepared, path):
    """Solve the exact pure-ridge path with one SVD of the centered design."""
    started = time.perf_counter()
    u, singular, vt = np.linalg.svd(prepared.x_centered, full_matrices=False)
    uty = u.T @ prepared.y_centered
    decomposition_seconds = time.perf_counter() - started
    del u
    cutoff = singular > 1e-15
    solved = []
    started = time.perf_counter()
    for value in path:
        weights = np.zeros_like(singular)
        weights[cutoff] = singular[cutoff] / (
            singular[cutoff] ** 2 + len(prepared.x_centered) * float(value))
        solved.append((vt.T @ (weights[:, None] * uty)).T)
    solve_seconds = time.perf_counter() - started
    return solved, decomposition_seconds, solve_seconds


def fit_lambda_path_audited(context, target, *, origin_rows, normalization,
                            context_hours, horizon_minutes, mixing,
                            max_iter=100_000, retry_max_iter=200_000, tol=1e-7,
                            on_attempt=None, on_candidate=None, prepared=None):
    """Fit the exact audited grid with shared centering and pathwise solvers.

    Positive-mixing candidates use one warm-started ``enet_path`` segment. If a
    segment warns, the first iteration-limited candidate is checked separately,
    gets the existing one bounded retry from the preceding converged solution,
    and the remaining segment resumes only after that candidate succeeds.
    """
    from sklearn.exceptions import ConvergenceWarning

    _mixing(mixing)
    if not isinstance(max_iter, int) or max_iter < 1 or retry_max_iter < max_iter:
        raise ValueError("invalid solver iteration budgets")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("invalid solver tolerance")
    preparation_started = time.perf_counter()
    if prepared is None:
        prepared = prepare_elastic_net_training(context, target,
            origin_rows=tuple(origin_rows), normalization=normalization,
            context_hours=context_hours, horizon_minutes=horizon_minutes)
    elif (prepared.context_hours != context_hours or
          prepared.horizon_minutes != horizon_minutes or
          prepared.normalization_sha256 != normalization.statistics_sha256):
        raise ValueError("prepared training matrices belong to another task or normalization")
    elif tuple(origin_rows) != prepared.rows:
        raise ValueError("prepared training origin identity differs from requested path")
    preparation_seconds = time.perf_counter() - preparation_started
    path = _lambda_path_prepared(prepared, mixing)
    results = []
    timing = dict(preparation_seconds=preparation_seconds, fit_seconds=0.,
                  training_check_seconds=0., ridge_decomposition_seconds=0.,
                  ridge_solve_seconds=0.)

    def coverage():
        attempted = len(results)
        successful = sum(item["status"] == "completed" for item in results)
        failed = sum(item["status"] == "failed" for item in results)
        complete = attempted == len(path)
        return dict(expected=len(path), attempted=attempted, successful=successful,
                    failed=failed, unattempted=len(path) - attempted, complete=complete,
                    selection_eligible=complete and successful > 0,
                    retries=sum(max(0, len(item["attempts"]) - 1) for item in results))

    def publish(item):
        results.append(item)
        if on_candidate is not None:
            on_candidate({k: v for k, v in item.items() if k != "task"}, coverage())

    def publish_failure(index, value, attempts, error_class):
        publish(dict(index=index, mixing=float(mixing), lambda_value=float(value),
                     status="failed", attempts=attempts, error_class=error_class))

    def publish_success(index, value, *, budget, coef, n_iter, dual_gap, attempts):
        eps = float(tol * np.sum(prepared.y_centered ** 2)) if mixing > 0 else None
        try:
            if mixing > 0 and (n_iter is None or int(n_iter) < 1 or
                    dual_gap is None or not np.isfinite(float(dual_gap)) or
                    not np.isfinite(eps)):
                raise InvalidCandidateFit("nonfinite or missing convergence diagnostics")
            task = _path_task(prepared=prepared, normalization=normalization,
                context_hours=context_hours, horizon_minutes=horizon_minutes,
                mixing=mixing, lambda_value=float(value), max_iter=budget, tol=tol,
                coef=coef, n_iter=int(n_iter) if n_iter is not None else None,
                dual_gap=float(dual_gap) if dual_gap is not None else None, eps=eps)
            check_started = time.perf_counter()
            finite = _fitted_values_are_finite(prepared, task.estimator_)
            timing["training_check_seconds"] += time.perf_counter() - check_started
        except (InvalidCandidateFit, FloatingPointError, np.linalg.LinAlgError,
                OverflowError) as exc:
            event = dict(index=index, mixing=float(mixing), lambda_value=float(value),
                attempt=len(attempts) + 1, max_iter=budget, status="numerical_failure",
                error=f"{type(exc).__name__}: {exc}")
            attempts.append({k: v for k, v in event.items()
                             if k not in ("index", "mixing", "lambda_value", "attempt")})
            if on_attempt is not None:
                on_attempt(event)
            publish_failure(index, value, attempts, "numerical_failure")
            return None
        if not finite:
            event = dict(index=index, mixing=float(mixing), lambda_value=float(value),
                attempt=len(attempts) + 1, max_iter=budget, status="numerical_failure",
                error="nonfinite or malformed fitted values")
            attempts.append({k: v for k, v in event.items()
                             if k not in ("index", "mixing", "lambda_value", "attempt")})
            if on_attempt is not None:
                on_attempt(event)
            publish_failure(index, value, attempts, "numerical_failure")
            return None
        diagnostics = dict(n_iter=_diagnostic(task.estimator_.n_iter_),
                           dual_gap=_diagnostic(task.estimator_.dual_gap_),
                           eps=_diagnostic(task.estimator_.eps_))
        event = dict(index=index, mixing=float(mixing), lambda_value=float(value),
            attempt=len(attempts) + 1, max_iter=budget, status="converged",
            diagnostics=diagnostics)
        attempts.append({k: v for k, v in event.items()
                         if k not in ("index", "mixing", "lambda_value", "attempt")})
        if on_attempt is not None:
            on_attempt(event)
        publish(dict(index=index, mixing=float(mixing), lambda_value=float(value),
            status="completed", attempts=attempts, task=task, diagnostics=diagnostics))
        return task.estimator_.coef_

    if mixing == 0:
        try:
            coefficients, decomposition, solving = _ridge_path(prepared, path)
            timing["ridge_decomposition_seconds"] = decomposition
            timing["ridge_solve_seconds"] = solving
            timing["fit_seconds"] = decomposition + solving
            for index, (value, coef) in enumerate(zip(path, coefficients)):
                publish_success(index, value, budget=max_iter, coef=coef,
                                n_iter=None, dual_gap=None, attempts=[])
        except (InvalidCandidateFit, FloatingPointError, np.linalg.LinAlgError,
                OverflowError) as exc:
            for index in range(len(results), len(path)):
                value = path[index]
                event = dict(index=index, mixing=0., lambda_value=float(value), attempt=1,
                    max_iter=max_iter, status="numerical_failure",
                    error=f"{type(exc).__name__}: {exc}")
                if on_attempt is not None:
                    on_attempt(event)
                publish_failure(index, value,
                    [{k: v for k, v in event.items()
                      if k not in ("index", "mixing", "lambda_value", "attempt")}],
                    "numerical_failure")
        return dict(results=results, coverage=coverage(), timing=timing)

    index = 0
    preceding_coef = None
    while index < len(path):
        segment = _positive_path_segment(prepared, path[index:], mixing=mixing,
            max_iter=max_iter, tol=tol, coef_init=preceding_coef)
        timing["fit_seconds"] += segment["fit_seconds"]
        if not segment["warnings"]:
            for local in range(len(path) - index):
                absolute = index + local
                fitted_coef = publish_success(absolute, path[absolute], budget=max_iter,
                    coef=segment["coefs"][:, :, local], n_iter=segment["n_iters"][local],
                    dual_gap=segment["dual_gaps"][local], attempts=[])
                if fitted_coef is not None:
                    preceding_coef = fitted_coef
            break

        suspects = np.flatnonzero(segment["n_iters"] >= max_iter)
        if not len(suspects):
            raise RuntimeError("path emitted a convergence warning without an iteration-limited candidate")
        suspect = int(suspects[0])
        for local in range(suspect):
            absolute = index + local
            fitted_coef = publish_success(absolute, path[absolute], budget=max_iter,
                coef=segment["coefs"][:, :, local], n_iter=segment["n_iters"][local],
                dual_gap=segment["dual_gaps"][local], attempts=[])
            if fitted_coef is not None:
                preceding_coef = fitted_coef
        index += suspect
        single = _positive_path_segment(prepared, path[index:index + 1], mixing=mixing,
            max_iter=max_iter, tol=tol, coef_init=preceding_coef)
        timing["fit_seconds"] += single["fit_seconds"]
        if not single["warnings"]:
            fitted_coef = publish_success(index, path[index], budget=max_iter,
                coef=single["coefs"][:, :, 0], n_iter=single["n_iters"][0],
                dual_gap=single["dual_gaps"][0], attempts=[])
            if fitted_coef is not None:
                preceding_coef = fitted_coef
            index += 1
            continue

        attempts = []
        warning_event = dict(index=index, mixing=float(mixing),
            lambda_value=float(path[index]), attempt=1, max_iter=max_iter,
            status="convergence_warning",
            error=f"{ConvergenceWarning.__name__}: {single['warnings'][0]}")
        attempts.append({k: v for k, v in warning_event.items()
                         if k not in ("index", "mixing", "lambda_value", "attempt")})
        if on_attempt is not None:
            on_attempt(warning_event)
        if retry_max_iter == max_iter:
            publish_failure(index, path[index], attempts, "convergence_warning")
            index += 1
            continue
        retry = _positive_path_segment(prepared, path[index:index + 1], mixing=mixing,
            max_iter=retry_max_iter, tol=tol, coef_init=preceding_coef)
        timing["fit_seconds"] += retry["fit_seconds"]
        if retry["warnings"]:
            event = dict(index=index, mixing=float(mixing), lambda_value=float(path[index]),
                attempt=2, max_iter=retry_max_iter, status="convergence_warning",
                error=f"{ConvergenceWarning.__name__}: {retry['warnings'][0]}")
            attempts.append({k: v for k, v in event.items()
                             if k not in ("index", "mixing", "lambda_value", "attempt")})
            if on_attempt is not None:
                on_attempt(event)
            publish_failure(index, path[index], attempts, "convergence_warning")
            index += 1
            continue
        fitted_coef = publish_success(index, path[index], budget=retry_max_iter,
            coef=retry["coefs"][:, :, 0], n_iter=retry["n_iters"][0],
            dual_gap=retry["dual_gaps"][0], attempts=attempts)
        if fitted_coef is not None:
            preceding_coef = fitted_coef
        index += 1
    return dict(results=results, coverage=coverage(), timing=timing)


def fit_frozen_selected_candidate(context, target, *, origin_rows, normalization,
                                  context_hours, horizon_minutes, mixing,
                                  lambda_value, max_iter=100_000, tol=1e-7):
    """Materialize one already-selected candidate without rerunning its grid.

    The fit uses the same centered objective and solver implementations as the
    production path code, but supplies exactly one frozen lambda.  Selection,
    validation, retry, and fallback decisions deliberately remain outside this
    function.
    """
    from sklearn.exceptions import ConvergenceWarning

    _mixing(mixing)
    if (not np.isfinite(lambda_value) or lambda_value <= 0 or
            not isinstance(max_iter, int) or max_iter < 1 or
            not np.isfinite(tol) or tol <= 0):
        raise ValueError("invalid frozen candidate specification")
    prepared = prepare_elastic_net_training(
        context, target, origin_rows=tuple(origin_rows),
        normalization=normalization, context_hours=context_hours,
        horizon_minutes=horizon_minutes)
    if mixing == 0:
        coefficients, _, _ = _ridge_path(
            prepared, np.asarray([lambda_value], dtype=np.float64))
        coefficient = coefficients[0]
        n_iter = dual_gap = eps = None
    else:
        segment = _positive_path_segment(
            prepared, np.asarray([lambda_value], dtype=np.float64),
            mixing=mixing, max_iter=max_iter, tol=tol, coef_init=None)
        if segment["warnings"]:
            raise ConvergenceWarning(
                "frozen selected candidate did not converge in its recorded bound")
        coefficient = segment["coefs"][:, :, 0]
        n_iter = int(segment["n_iters"][0])
        dual_gap = float(segment["dual_gaps"][0])
        eps = float(tol * np.sum(prepared.y_centered ** 2))
    task = _path_task(
        prepared=prepared, normalization=normalization,
        context_hours=context_hours, horizon_minutes=horizon_minutes,
        mixing=mixing, lambda_value=lambda_value, max_iter=max_iter, tol=tol,
        coef=coefficient, n_iter=n_iter, dual_gap=dual_gap, eps=eps)
    if not _fitted_values_are_finite(prepared, task.estimator_):
        raise FloatingPointError("frozen selected candidate has invalid fitted values")
    return task
