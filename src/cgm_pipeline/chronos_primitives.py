"""Native Chronos-Bolt precision and sampling helpers from the benchmark."""

from types import MethodType

import numpy as np
import torch


def require_finite_fp32(value, name):
    if value.dtype != torch.float32 or not torch.isfinite(value).all():
        raise FloatingPointError(f'{name} must be finite FP32')


def _fp32_head_forward(head, hidden):
    # Keep the official module and state-dict keys: only its execution dtype
    # changes. The differentiable cast retains FP16 backbone autocast.
    with torch.autocast(hidden.device.type, enabled=False):
        hidden = hidden.float()
        require_finite_fp32(hidden, 'output-head operand')
        for parameter in head.parameters():
            require_finite_fp32(parameter, 'output-head parameter')
        output = head._official_chronos_forward(hidden)
        # Check all native horizon positions BEFORE the unchanged native loss
        # can multiply padded-position errors by zero masks.
        require_finite_fp32(output, 'complete output-head loss input')
        return output


def install_fp32_output_head(model):
    """Execution-only shim; do not wrap/rekey or replace the official head."""
    head = model.output_patch_embedding
    if not getattr(head, '_chronos_fp32_head', False):
        head._official_chronos_forward = head.forward
        head.forward = MethodType(_fp32_head_forward, head)
        head._chronos_fp32_head = True


def check_finite_optimizer(model, optimizer):
    for name, parameter in model.named_parameters():
        require_finite_fp32(parameter, f'parameter {name}')
        for key, value in optimizer.state.get(parameter, {}).items():
            if torch.is_tensor(value):
                require_finite_fp32(value, f'optimizer state {name}/{key}')


def _finite_native_scaling(module, inputs, output):
    normalized, (location, scale) = output
    for name, value in [('normalized context/target', normalized),
                        ('native location', location), ('native scale', scale)]:
        require_finite_fp32(value, name)
    if not (scale > 0).all():
        raise FloatingPointError('native scale must remain positive')


def optimizer_batch(count, *, seed, step):
    """Addressable full-index permutation, including ordinary epoch wrapping."""
    if count < 64 or step < 1:
        raise ValueError('insufficient training origins/invalid step')
    offset = (step - 1) * 64
    epoch, start = divmod(offset, count)
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch+1])).permutation(count)
    ids = order[start:start+64]
    if len(ids) < 64:
        next_order = np.random.default_rng(np.random.SeedSequence([seed, epoch+2])).permutation(count)
        ids = np.concatenate([ids, next_order[:64-len(ids)]])
    return ids
