"""Bounded same-batch AMP retries for Chronos-Bolt-Small."""
import hashlib

import torch


POLICY = dict(max_retries=3, backoff_factor=0.5, minimum_scale=16.0,
              monitored_boundaries=['encoder.block.0.layer.0.dropout',
                                    'input_patch_embedding.act'],
              restore_rng=True, same_batch=True, require_retry_boundaries_finite=True)


def rng_identity(cpu, cuda):
    h = hashlib.sha256(cpu.cpu().numpy().tobytes())
    for state in cuda:
        h.update(state.cpu().numpy().tobytes())
    return h.hexdigest()


def nonfinite(tensor):
    return dict(nan=int(torch.isnan(tensor).sum()),
                positive_inf=int(torch.isposinf(tensor).sum()),
                negative_inf=int(torch.isneginf(tensor).sum()))


def gradient_failures(model):
    missing, invalid = [], {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not bool(torch.isfinite(parameter.grad).all()):
            invalid[name] = nonfinite(parameter.grad)
        elif parameter.grad.dtype != torch.float32:
            raise FloatingPointError(f'gradient must be FP32: {name} ({parameter.grad.dtype})')
    return missing, invalid


class BackwardBoundaries:
    def __init__(self, model, names):
        self.invalid, self.handles = {}, []
        modules = dict(model.named_modules())
        for name in names:
            if name not in modules:
                raise ValueError(f'missing monitored backward boundary: {name}')
            self.handles.append(modules[name].register_forward_hook(self.forward_hook(name)))

    def forward_hook(self, name):
        def hook(module, inputs, output):
            if not torch.is_tensor(output) or not output.requires_grad:
                raise ValueError(f'monitored backward boundary is not differentiable: {name}')
            def backward(gradient):
                if not bool(torch.isfinite(gradient).all()):
                    counts = nonfinite(gradient)
                    current = self.invalid.setdefault(name, dict(nan=0, positive_inf=0, negative_inf=0))
                    for key, value in counts.items():
                        current[key] += value
                return None
            output.register_hook(backward)
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def reduced_scaler(scaler, new_scale):
    """Discard the failed attempt's unscale state using public scaler APIs.

    Calling scaler.step would update at scale 64 when only an intermediate
    boundary overflowed. A fresh scaler avoids that update and avoids private
    found_inf mutation or update() before step(). Successful attempts retain
    the ordinary scaler.step()/update() growth path.
    """
    state = dict(scaler.state_dict(), scale=new_scale, _growth_tracker=0)
    replacement = torch.amp.GradScaler('cuda', enabled=scaler.is_enabled(), init_scale=new_scale)
    replacement.load_state_dict(state)
    return replacement


def finite_batch(tuner, forward_backward, *, batch_sha256, emit, check_state):
    policy = tuner.task.config['execution']['gradient_retry']
    cpu = torch.get_rng_state().clone()
    cuda = [state.clone() for state in torch.cuda.get_rng_state_all()]
    rng_sha = rng_identity(cpu, cuda)
    scales = []
    for attempt in range(policy['max_retries'] + 1):
        if attempt:
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state_all(cuda)
            if rng_identity(torch.get_rng_state(), torch.cuda.get_rng_state_all()) != rng_sha:
                raise RuntimeError('retry RNG restoration failed')
        tuner.optimizer.zero_grad(set_to_none=True)
        boundaries = BackwardBoundaries(tuner.task.model, policy['monitored_boundaries']) if attempt else None
        scale = tuner.scaler.get_scale()
        scales.append(scale)
        try:
            loss = forward_backward()
            tuner.scaler.unscale_(tuner.optimizer)
            missing, invalid = gradient_failures(tuner.task.model)
            boundary_invalid = boundaries.invalid if boundaries else {}
        except Exception as exc:
            missing, invalid = gradient_failures(tuner.task.model)
            emit(dict(optimizer_step=tuner.steps + 1, batch_sha256=batch_sha256,
                attempt=attempt, attempted_scales=list(scales), missing_parameters=missing,
                nonfinite_parameters=invalid, nonfinite_modules=dict(boundaries.invalid) if boundaries else {},
                rng_sha256=rng_sha, rng_restored=bool(attempt), optimizer_updates=0,
                outcome='failed_forward_or_backward', final_scale=None, error=f'{type(exc).__name__}: {exc}'))
            raise
        finally:
            if boundaries:
                boundaries.close()
        check_state(tuner.task.model, tuner.optimizer)
        record = dict(optimizer_step=tuner.steps + 1, batch_sha256=batch_sha256,
                      attempt=attempt, attempted_scales=list(scales), missing_parameters=missing,
                      nonfinite_parameters=invalid, nonfinite_modules=dict(boundary_invalid),
                      rng_sha256=rng_sha, rng_restored=bool(attempt), optimizer_updates=0)
        if missing:
            emit(dict(record, outcome='failed_missing_gradients', final_scale=None))
            raise FloatingPointError(f'missing full-fine-tuning gradients: {missing}')
        if not invalid and not boundary_invalid:
            tuner.scaler.step(tuner.optimizer)
            tuner.scaler.update()
            check_state(tuner.task.model, tuner.optimizer)
            if attempt:
                emit(dict(record, outcome='finite_update', final_scale=scale, optimizer_updates=1))
            tuner.retry_state = dict(total_retries=tuner.retry_state['total_retries'] + attempt,
                last_step=tuner.steps + 1, last_batch_sha256=batch_sha256,
                attempted_scales=list(scales), final_scale=scale, rng_sha256=rng_sha,
                outcome='finite_update')
            return loss
        terminal = attempt == policy['max_retries'] or scale * policy['backoff_factor'] < policy['minimum_scale']
        emit(dict(record, outcome='failed_retry_exhaustion' if terminal else 'retry', final_scale=None))
        tuner.optimizer.zero_grad(set_to_none=True)
        if terminal:
            raise FloatingPointError(f'bounded gradient retry exhausted at scales {scales}; '
                                     f'parameters={invalid}; modules={boundary_invalid}')
        tuner.scaler = reduced_scaler(tuner.scaler, scale * policy['backoff_factor'])
    raise AssertionError('unreachable bounded retry state')
