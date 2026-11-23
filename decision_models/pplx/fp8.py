"""FP8 (e4m3) linear layers for GPUs with FP8 tensor cores (Ada sm89, Hopper sm90 and newer).

Weights are quantized once at load, with one scale per output channel; activations are quantized at run time, with
one scale per token (dynamic W8A8, "rowwise"). The product runs in torch._scaled_mm with bf16 output.

Scales are per token on purpose: one scale per tensor would depend on every token in the batch, so a question's
answer would change with the other questions packed next to it, which forked scoring exists to prevent. If this torch
build has no rowwise kernel for the GPU, the per-tensor kernel runs with unit scales and the per-token and
per-channel scales are applied to its output ("rowwise_emulated": same numbers, one extra elementwise pass).

Only the large projections are quantized (attention q/k/v/o, Gated DeltaNet in/out projections, MLP). Small
projections (the DeltaNet a/b gates, 48 outputs), norms, convolutions, embeddings and the readout stay in bf16.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

E4M3 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(E4M3).max  # 448
EPS = 1e-12
MODES = ("rowwise", "rowwise_emulated")


def supported(device) -> str | None:
    """'rowwise' (native kernel), 'rowwise_emulated' (per-tensor kernel) or None if FP8 does not run on `device`."""
    device = torch.device(device)
    if not hasattr(torch, "_scaled_mm"):
        return None
    if device.type == "cuda" and torch.cuda.get_device_capability(device) < (8, 9):
        return None  # Ampere (A6000) and older have no FP8 tensor cores
    a = torch.randn(32, 64, device=device).to(E4M3)
    b = torch.randn(48, 64, device=device).to(E4M3).t()
    probes = {"rowwise": (torch.ones(32, 1, device=device), torch.ones(1, 48, device=device)),
              "rowwise_emulated": (torch.ones((), device=device), torch.ones((), device=device))}
    for mode, (sa, sb) in probes.items():
        try:
            torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
            return mode
        except (RuntimeError, NotImplementedError):
            continue
    return None


def _scale(x):
    """Per-row quantization scale max |x| / 448, as float32 (rows, 1). Rounded to a bf16 value, so that dividing bf16
    activations by it uses exactly the scale the matrix multiply multiplies back."""
    amax = x.abs().amax(-1, keepdim=True)
    return (amax.float().clamp(min=EPS) / FP8_MAX).to(torch.bfloat16).float()


def _quantize(x, scale):
    return (x / scale.to(x.dtype)).clamp(-FP8_MAX, FP8_MAX).to(E4M3)


class FP8Linear(nn.Module):
    """Drop-in replacement for an nn.Linear, built from it (on the device it should run on)."""

    def __init__(self, linear: nn.Linear, mode: str):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown FP8 mode {mode}")
        self.in_features, self.out_features, self.mode = linear.in_features, linear.out_features, mode
        w = linear.weight.detach().float()
        scale = _scale(w)                                                  # (N, 1)
        self.register_buffer("weight", _quantize(w, scale))                # (N, K) e4m3, row-major
        self.register_buffer("weight_scale", scale.reshape(1, -1))         # (1, N)
        self.register_buffer("one", torch.ones((), device=w.device))
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach(), requires_grad=False)

    def forward(self, x):
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        rows = x2.shape[0]
        pad = (-rows) % 16  # FP8 GEMMs want every dimension a multiple of 16
        if pad:
            x2 = F.pad(x2, (0, 0, 0, pad))
        scale = _scale(x2)                                                 # (M, 1)
        xq = _quantize(x2, scale)
        if self.mode == "rowwise":
            out = torch._scaled_mm(xq, self.weight.t(), scale_a=scale, scale_b=self.weight_scale, bias=self.bias,
                                   out_dtype=x.dtype)
        else:  # unscaled products (|value| <= 448^2 * K, well inside bf16's range), then both scales in float32
            out = torch._scaled_mm(xq, self.weight.t(), scale_a=self.one, scale_b=self.one, out_dtype=x.dtype)
            out = (out.float() * scale * self.weight_scale).to(x.dtype)
            if self.bias is not None:
                out = out + self.bias
        return out[:rows].reshape(*lead, self.out_features)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, mode={self.mode}"


def quantize(module: nn.Module, mode: str, min_features: int = 256) -> int:
    """Replace every nn.Linear inside `module` that has at least `min_features` outputs (and dimensions that are
    multiples of 16) with an FP8Linear, in place. Returns how many were replaced."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if (child.out_features >= min_features and child.in_features % 16 == 0
                    and child.out_features % 16 == 0):
                setattr(module, name, FP8Linear(child, mode))
                count += 1
        else:
            count += quantize(child, mode, min_features)
    return count
