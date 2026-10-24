"""Operation-based compute accounting without changing tensor execution.

Counts are theoretical shape-based FLOPs, not hardware instructions. Dense
matmul/attention FLOPs use PyTorch's formulas (multiply-add = 2, dense attention
even when causal). Auxiliary arithmetic is a separate estimate: transcendental
functions count as one scalar operation. Unsupported floating ops are exposed.
No tensors/graphs are retained and no extra GPU synchronization is introduced.
"""

from collections import Counter
from functools import wraps

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.flop_counter import (
    flop_registry,
    sdpa_backward_flop_count,
    sdpa_flop_count,
)

PROTOCOL = {
    "version": "executed_ops_dense_flops_v1",
    "multiply_add_flops": 2,
    "attention": "dense equivalent, including padding; causal sparsity not discounted",
    "auxiliary": "estimated scalar arithmetic; transcendental/sqrt/division = one op",
    "scope": "executed forward, backward, checkpoint recomputation and optimizer; validation separate",
    "limitations": "not hardware instructions or memory traffic; unsupported float ops listed; fused kernels use formulas",
}


class ComputeCounter(TorchDispatchMode):
    """Observe eager operators without decomposing or replacing their kernels."""

    def __init__(self):
        super().__init__()
        self.matmul = 0
        self.auxiliary = 0
        self.operations = Counter()
        self.unsupported = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        packet = func._overloadpacket
        name = str(packet).removeprefix("aten.")
        # Registry formulas receive tensors but read only their shapes.
        if packet in flop_registry:
            n = int(flop_registry[packet](*args, **kwargs, out_val=out))
            self.matmul += n
            self.operations[name] += n
            return out
        if "scaled_dot_product" in name and "attention" in name:
            if "backward" in name:
                n = sdpa_backward_flop_count(*(x.shape for x in args[:4]))
            else:
                n = sdpa_flop_count(*(x.shape for x in args[:3]))
            self.matmul += int(n)
            self.operations[name] += int(n)
            return out
        values, _ = tree_flatten(out)
        floats = [
            x for x in values if isinstance(x, torch.Tensor) and x.is_floating_point()
        ]
        if not floats:
            return out
        size = floats[0].numel()
        source = args[0] if args and isinstance(args[0], torch.Tensor) else None
        n = None
        # Comparisons, allocation, casts, indexing and copies count as zero
        # arithmetic, despite potentially substantial bandwidth/runtime costs.
        zero = {
            "view",
            "_unsafe_view",
            "reshape",
            "as_strided",
            "detach",
            "alias",
            "transpose",
            "t",
            "permute",
            "slice",
            "select",
            "narrow",
            "expand",
            "unsqueeze",
            "squeeze",
            "split",
            "split_with_sizes",
            "unbind",
            "cat",
            "stack",
            "clone",
            "copy_",
            "_to_copy",
            "to",
            "contiguous",
            "empty",
            "empty_like",
            "empty_strided",
            "zeros",
            "zeros_like",
            "ones",
            "ones_like",
            "full",
            "full_like",
            "new_empty",
            "new_zeros",
            "zero_",
            "fill_",
            "lift_fresh",
            "lift_fresh_copy",
            "arange",
            "scalar_tensor",
            "new_empty_strided",
            "index",
            "index_select",
            "embedding",
            "gather",
            "where",
            "clamp",
            "clamp_min",
            "clamp_max",
            "maximum",
            "minimum",
            "bernoulli_",
            "uniform_",
            "normal_",
            "constant_pad_nd",
            "slice_backward",
            "select_backward",
            "masked_fill",
            "masked_fill_",
            "expand_as",
            "resolve_conj",
            "resolve_neg",
        }
        if name in zero:
            n = 0
        elif name.rstrip("_") in {
            "add",
            "sub",
            "mul",
            "div",
            "neg",
            "pow",
            "sqrt",
            "rsqrt",
            "sin",
            "cos",
            "exp",
            "log",
            "tanh",
            "reciprocal",
            "abs",
        }:
            n = size
        elif name.rstrip("_") in {"addcmul", "addcdiv", "lerp"}:
            n = 3 * size
        elif name in {"sum", "mean"}:
            n = max(0, source.numel() - size) + (size if name == "mean" else 0)
        elif name in {"amax", "max"}:
            n = 0
        elif name == "logsumexp":
            n = 3 * source.numel() + size  # subtraction, exp, sum, log/add
        elif name == "logaddexp":
            n = 6 * size
        elif name in {"linalg_vector_norm", "norm"}:
            order = args[1] if len(args) > 1 else kwargs.get("ord", kwargs.get("p", 2))
            if order in (None, 2):
                n = 2 * source.numel()  # square, reduction, sqrt
        elif name in {"_softmax", "_safe_softmax", "_log_softmax"}:
            n = 5 * size
        elif name == "tanh_backward":
            n = 3 * size
        elif name in {"_softmax_backward_data", "_log_softmax_backward_data"}:
            n = 5 * size
        elif name == "native_layer_norm":
            n = 8 * source.numel()
        elif name == "native_layer_norm_backward":
            n = 15 * source.numel()
        elif name in {"native_dropout", "native_dropout_backward"}:
            n = 2 * size
        elif name in {"index_add", "index_add_", "scatter_add", "scatter_add_"}:
            n = args[3].numel()
        elif name == "embedding_dense_backward":
            n = source.numel()
        elif name in {"index_put", "index_put_"}:
            accumulate = args[3] if len(args) > 3 else kwargs.get("accumulate", False)
            n = args[2].numel() if accumulate else 0
        elif name == "nll_loss_forward":
            n = args[1].numel()
        elif name == "nll_loss_backward":
            n = args[2].numel()
        if n is None:
            self.unsupported[name] += 1
        else:
            self.auxiliary += int(n)
        return out

    def summary(self):
        return {
            "protocol": PROTOCOL["version"],
            "matmul_attention_flops": self.matmul,
            "auxiliary_flops_estimate": self.auxiliary,
            "accounted_flops_estimate": self.matmul + self.auxiliary,
            "matmul_flops_by_operator": dict(self.operations),
            "unsupported_float_ops": dict(self.unsupported),
        }


def counted_update(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with ComputeCounter() as counter:
            result = fn(*args, **kwargs)
        return {**result, "compute": counter.summary()}

    return wrapped


def accumulate(state, measurement, phase):
    totals = state.setdefault("compute", {})
    total = totals.setdefault(
        phase,
        {
            "calls": 0,
            "matmul_attention_flops": 0,
            "auxiliary_flops_estimate": 0,
            "accounted_flops_estimate": 0,
            "unsupported_float_ops": {},
        },
    )
    total["calls"] += 1
    for name in (
        "matmul_attention_flops",
        "auxiliary_flops_estimate",
        "accounted_flops_estimate",
    ):
        total[name] += measurement[name]
    missing = Counter(total["unsupported_float_ops"])
    missing.update(measurement["unsupported_float_ops"])
    total["unsupported_float_ops"] = dict(missing)
    return totals
