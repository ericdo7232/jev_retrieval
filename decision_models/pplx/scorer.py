"""pplx-decider-v1.2 yes/no scoring for DecisionRAG-style navigation, on one or several GPUs.

Three ways to ask N yes/no questions about one state:

  exact    Every question in its own request with the full state: the model's "one question per request" mode.
           Answers are independent, but the state is recomputed for every question (N x state tokens).
  joint    All questions in one request: the model's default mode. The state is read once, but the full-attention
           layers are noncausal, so every question attends to every other question and the answers depend on
           which other candidates are in the request. Requests over `joint_max_length` are split in halves.
  forked   The state is computed once and cached. Every question then runs as its own branch that attends to the
           cached state and to itself only, at the positions and with the tokens it has in a request of its own.
           Answers are independent of the other questions, at about the cost of joint.

forked differs from exact in one way only. In a request of its own, the state's tokens in the 16 full-attention
layers also attend to the question, so the state is encoded slightly differently for every question; here the
state is encoded once, without seeing any question. Everything else matches exactly:
  - tokens: each branch is the token sequence the question has in a request of its own (checked per request);
  - positions: each branch continues from the end of the state;
  - Gated DeltaNet layers (causal): each branch starts from the state's final recurrent state and convolution window,
    which is what it would see in a request of its own;
  - full-attention layers: each branch attends to the state's cached keys and values plus its own, merged with
    their log-sum-exp (exact softmax over the union).

The layer code follows the package's own fast path (source/src/pplx_decider/fast.py), which the model card reports
matches the reference implementation to bf16 noise. `official` runs the unmodified Transformers model (the
package's reference path) for checking this implementation; `official_masked` does the same with the forked
attention pattern, so it must match `forked` up to numerical noise.

Multi-GPU: the decoder layers are split across the given devices (embeddings and the first layers on the first
device, the last layers, final norm and readout on the last); hidden states move between devices between layers.

FP8 (fp8=True, Ada/Hopper only): the large projections run as FP8 matrix multiplies (fp8.py). The model then fits on
one 48 GB GPU, so each GPU can run its own copy on its own share of the queries instead of splitting one copy.
"""
from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

QUESTION_START = "\n\nQuestion 0:\n"
MODES = ("exact", "joint", "forked", "fast", "official", "official_masked")


def fork_text(instructions):
    """The start all instructions share, cut so it does not depend on which questions are asked: the text before
    the first digit if every instruction has the same one (templated questions like "... on Page 17?"), else the
    common prefix cut back to the last space."""
    first = instructions[0]
    digit = next((i for i, c in enumerate(first) if c.isdigit()), None)
    if digit is not None and all(t[:digit] == first[:digit] and len(t) > digit and t[digit].isdigit()
                                 for t in instructions):
        return first[:digit]
    common = first
    for t in instructions[1:]:
        i = 0
        while i < min(len(common), len(t)) and common[i] == t[i]:
            i += 1
        common = common[:i]
    return common[: common.rfind(" ") + 1]


def import_package(checkpoint):
    """Make the checkpoint's own `pplx_decider` package importable."""
    src = str(Path(checkpoint) / "source" / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


class Pack:
    """Independent sequences concatenated into one (padding-free) sequence."""

    def __init__(self, lengths):
        self.lengths = list(lengths)
        self.cu_cpu = F.pad(torch.tensor(self.lengths).cumsum(0), (1, 0))
        self.max = max(self.lengths)
        self.total = int(self.cu_cpu[-1])
        self._on = {}

    def on(self, device):
        """(cu_seqlens int32, cu_seqlens int64, position within its sequence) on `device`."""
        if device not in self._on:
            cu = self.cu_cpu.to(device)
            seqpos = torch.cat([torch.arange(n) for n in self.lengths]).to(device)
            self._on[device] = (cu.to(torch.int32), cu.long(), seqpos)
        return self._on[device]


def conv_with_tail(x, weight, seqpos, tail=None):
    """Depthwise causal convolution over a pack (1, C, T) that never reads across sequence starts (fp32 sums).

    tail (1, C, taps - 1): the last convolution inputs of a shared state that every sequence continues; without it
    the taps before a sequence start read zeros, as in the package's packed_conv.
    """
    taps = weight.shape[-1]
    xf = x.float()
    out = xf * weight[:, taps - 1, None].float()
    for shift in range(1, taps):
        shifted = F.pad(xf, (shift, 0))[..., : x.shape[-1]]
        inside = seqpos >= shift
        if tail is None:
            src = shifted * inside
        else:  # source position t - shift < 0 is state position P + t - shift, i.e. tail index taps - 1 + t - shift
            idx = (taps - 1 + seqpos - shift).clamp(0, taps - 2)
            src = torch.where(inside, shifted, tail.float()[:, :, idx])
        out = out + src * weight[:, taps - 1 - shift, None].float()
    return out.to(x.dtype)


def merge_attention(a, lse_a, b, lse_b):
    """Softmax attention over the union of two key sets from the two partial results. a, b: (T, H, D); lse: (T, H)."""
    m = torch.maximum(lse_a, lse_b)
    wa, wb = (lse_a - m).exp(), (lse_b - m).exp()
    out = (a.float() * wa[..., None] + b.float() * wb[..., None]) / (wa + wb)[..., None]
    return out.to(a.dtype)


def dense_attention(q, k, v, cu_q, cu_k, scale, want_lse, block=256):
    """Noncausal attention of each query segment over its key segment, in plain PyTorch (fp32).

    q (Tq, H, D), k/v (Tk, Hk, D); cu_q/cu_k: CPU boundaries of the matching segments.
    """
    heads, kv_heads = q.shape[1], k.shape[1]
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    lse = torch.empty(q.shape[0], heads, dtype=torch.float32, device=q.device)
    for i in range(len(cu_q) - 1):
        qs, qe, ks, ke = int(cu_q[i]), int(cu_q[i + 1]), int(cu_k[i]), int(cu_k[i + 1])
        kk = k[ks:ke].float().repeat_interleave(heads // kv_heads, 1)
        vv = v[ks:ke].float().repeat_interleave(heads // kv_heads, 1)
        for b in range(qs, qe, block):
            e = min(b + block, qe)
            s = torch.einsum("qhd,khd->hqk", q[b:e].float(), kk) * scale
            l = s.logsumexp(-1)
            out[b:e] = torch.einsum("hqk,khd->qhd", (s - l[..., None]).exp(), vv)
            lse[b:e] = l.t()
    return out.to(q.dtype), (lse if want_lse else None)


class Kernels:
    """flash-linear-attention and torch varlen FlashAttention on CUDA, plain PyTorch otherwise (or if missing)."""

    def __init__(self, backend="auto"):
        if backend not in ("auto", "cuda", "torch"):
            raise ValueError(f"unknown backend {backend}")
        self.fla = self.varlen = self.aux = None
        self.notes = []
        if backend != "torch":
            try:
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule
                self.fla = chunk_gated_delta_rule
            except ImportError as e:
                if backend == "cuda":
                    raise
                self.notes.append(f"flash-linear-attention unavailable ({e}); using the PyTorch delta rule")
            try:
                from torch.nn.attention.varlen import varlen_attn
                self.varlen = varlen_attn
                try:
                    from torch.nn.attention.varlen import AuxRequest
                    self.aux = AuxRequest(lse=True)
                except ImportError:
                    self.notes.append("varlen_attn cannot return log-sum-exp in this torch; forked branches use "
                                      "PyTorch attention for the shared state")
            except ImportError as e:
                if backend == "cuda":
                    raise
                self.notes.append(f"torch varlen attention unavailable ({e}); using PyTorch attention")
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
        self.torch_delta = torch_chunk_gated_delta_rule

    def delta(self, q, k, v, g, beta, pack, init, final):
        """Chunked gated delta rule per sequence of the pack; init: (N, HV, K, V) starting states or None."""
        if self.fla is not None and q.is_cuda:
            _, cu_long, _ = pack.on(q.device)
            return self.fla(q, k, v, g=g, beta=beta, initial_state=init, output_final_state=final,
                            use_qk_l2norm_in_kernel=True, cu_seqlens=cu_long, cu_seqlens_cpu=pack.cu_cpu)
        outs, states = [], []
        cu = pack.cu_cpu.tolist()
        for i in range(len(cu) - 1):
            s, e = cu[i], cu[i + 1]
            o, st = self.torch_delta(q[:, s:e], k[:, s:e], v[:, s:e], g=g[:, s:e], beta=beta[:, s:e],
                                     initial_state=None if init is None else init[i:i + 1],
                                     output_final_state=final, use_qk_l2norm_in_kernel=True)
            outs.append(o)
            states.append(st)
        return torch.cat(outs, 1), (torch.cat(states) if final else None)

    def attention(self, q, k, v, cu_q, cu_k, max_q, max_k, scale, want_lse):
        """Noncausal attention of query segment i over key segment i. q (Tq, H, D), k/v (Tk, Hk, D)."""
        gqa = k.shape[1] != q.shape[1]
        if self.varlen is not None and q.is_cuda and (not want_lse or self.aux is not None):
            cq = cu_q.to(q.device, torch.int32)
            ck = cu_k.to(q.device, torch.int32)
            if not want_lse:
                return self.varlen(q, k, v, cq, ck, max_q, max_k, scale=scale, window_size=(-1, -1),
                                   enable_gqa=gqa), None
            out, lse = self.varlen(q, k, v, cq, ck, max_q, max_k, scale=scale, window_size=(-1, -1),
                                   enable_gqa=gqa, return_aux=self.aux)
            lse = lse.float()
            if lse.shape == (q.shape[1], q.shape[0]):
                lse = lse.t()
            elif lse.shape != (q.shape[0], q.shape[1]):
                raise RuntimeError(f"unexpected log-sum-exp shape {tuple(lse.shape)} for queries {tuple(q.shape)}")
            return out, lse
        return dense_attention(q, k, v, cu_q, cu_k, scale, want_lse)


def _move(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _move(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_move(v, device) for v in x)
    return x


def _inputs_to(device):
    def hook(module, args, kwargs):
        return _move(args, device), _move(kwargs, device)
    return hook


class Scorer:
    """Loads the checkpoint (or takes a built backbone and readout, for tests) and scores yes/no questions."""

    def __init__(self, checkpoint, *, devices=None, backbone=None, readout=None, backend="auto", budget=16384,
                 joint_max_length=32768, dtype=torch.bfloat16, fp8=False, fp8_min_features=256, mlp_chunk=8192):
        """fp8: quantize the large projections to FP8 as each layer is placed (needs Ada/Hopper; see fp8.py).
        With fp8 the 27B model fits on one 48 GB GPU. mlp_chunk: tokens per MLP call (bounds peak memory on long
        states; per-token, so results do not change)."""
        checkpoint = Path(checkpoint)
        import_package(checkpoint)
        from transformers import AutoProcessor
        from pplx_decider.model import enable_noncausal_full_attention
        from pplx_decider.protocol import MAX_OPTIONS, PROTOCOL, Encoder, load_json

        if load_json(checkpoint / "joint_config.json") != PROTOCOL:
            raise ValueError("checkpoint does not declare the joint inline-readout protocol (version 3)")
        config = load_json(checkpoint / "decision_config.json")
        self.temperature = float(config["temperature"])
        self.codes = list(config["codes"])
        self.processor = AutoProcessor.from_pretrained(str(checkpoint))
        self.processor.tokenizer.padding_side = "left"
        self.tok = self.processor.tokenizer
        if any(self.tok.encode(c, add_special_tokens=False) != [t]
               for c, t in zip(self.codes, config["token_ids"], strict=True)):
            raise ValueError("checkpoint answer codes differ from its tokenizer")
        self.encoder = Encoder(self.processor, self.codes)
        self.marker_id = self.encoder.marker_id

        if torch.cuda.is_available():
            torch.backends.cuda.enable_cudnn_sdp(False)  # the package's evaluated SDPA kernel choice
        if backbone is None:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
            backbone = Qwen3_5Model.from_pretrained(str(checkpoint), dtype=dtype, attn_implementation="sdpa")
        if readout is None:
            from safetensors.torch import load_file
            hidden = backbone.config.text_config.hidden_size
            readout = torch.nn.Linear(hidden, MAX_OPTIONS, bias=False, dtype=dtype)
            readout.load_state_dict(load_file(str(checkpoint / "readout.safetensors")))
        enable_noncausal_full_attention(backbone.language_model)
        self.backbone = backbone.requires_grad_(False).eval()
        self.readout = readout.requires_grad_(False).eval()
        self.lm = backbone.language_model
        self.layers = list(self.lm.layers)
        self.taps = backbone.config.text_config.linear_conv_kernel_dim
        self.kernels = Kernels(backend)
        self.budget = budget
        self.joint_max_length = joint_max_length
        self.mlp_chunk = mlp_chunk
        self.fp8, self.fp8_min_features, self.fp8_mode, self.fp8_layers = fp8, fp8_min_features, None, 0
        if devices is None:
            n = torch.cuda.device_count()
            devices = [f"cuda:{i}" for i in range(n)] if n else ["cpu"]
        self.place(devices)

    # ------------------------------------------------------------------ placement

    def place(self, devices):
        """Split the decoder layers across `devices` in order; the first device also holds the embeddings."""
        devices = [torch.device(d) for d in devices]
        n, k = len(self.layers), len(devices)
        counts = [n // k + (1 if i < n % k else 0) for i in range(k)]
        if k > 1 and counts[0] > 1:  # the embeddings (~2.5 GB) and vision encoder also sit on the first device
            counts[0] -= 1
            counts[-1] += 1
        self.layer_devices = [d for d, c in zip(devices, counts) for _ in range(c)]
        self.devices, self.first, self.last = devices, devices[0], devices[-1]
        if self.fp8:
            try:
                from . import fp8
            except ImportError:  # run as a script from pplx/ (smoke.py), not as a package
                import fp8
            modes = {fp8.supported(d) for d in devices}
            if None in modes:
                raise ValueError(f"FP8 needs torch._scaled_mm on every device {[str(d) for d in devices]} "
                                 "(an Ada, Hopper or newer GPU; the A6000 is Ampere and has no FP8 tensor cores)")
            self.fp8_mode = "rowwise_emulated" if "rowwise_emulated" in modes else "rowwise"
            if self.fp8_mode == "rowwise_emulated":
                self.kernels.notes.append("FP8: no rowwise _scaled_mm kernel for this GPU in this torch build; "
                                          "per-token scales applied after a per-tensor kernel (slightly slower)")
        else:  # text-only scoring never runs the vision encoder; with fp8 it stays on the CPU to save GPU memory
            self.backbone.visual.to(self.first)
        self.lm.embed_tokens.to(self.first)
        self.lm.rotary_emb.to(self.first)
        for layer, d in zip(self.layers, self.layer_devices):
            layer.to(d)  # one bf16 layer at a time on the GPU, then quantized: the bf16 model never has to fit
            if self.fp8:
                self.fp8_layers += fp8.quantize(layer, self.fp8_mode, self.fp8_min_features)
        if self.fp8 and any(d.type == "cuda" for d in devices):
            torch.cuda.empty_cache()
        self.lm.norm.to(self.last)
        self.readout.to(self.last)
        if len(set(devices)) > 1:  # the official (Transformers) forward needs its inputs moved between devices
            for layer, d in zip(self.layers, self.layer_devices):
                layer.register_forward_pre_hook(_inputs_to(d), with_kwargs=True)
            self.lm.norm.register_forward_pre_hook(_inputs_to(self.last), with_kwargs=True)
        self.split_summary = {str(d): c for d, c in zip(devices, counts)}

    # ------------------------------------------------------------------ tokenization

    def question_text(self, question):
        """The text a question adds after the state in a request of its own, through the end of the template."""
        t = self.encoder.text({"state": "x", "questions": {"q": question}})
        i = t.find(QUESTION_START)
        if i < 0 or not t[:i].endswith("State:\nx"):
            raise RuntimeError("unexpected prompt layout; the package's template changed")
        return t[i:]

    def split(self, state, questions, checks=8):
        """Token ids of the shared state and of every question's branch.

        For every question i, shared + branches[i] is exactly the token sequence of the request {state, question i}.
        The shared part stops a few tokens before the end of the state, because the last state tokens can merge with
        the question's first tokens; those few tokens are repeated in every branch. Checked on `checks` questions.
        """
        return self._split(state, questions, checks)[:2]

    def _split(self, state, questions, checks=8):
        """split(), plus the text of the state's last few tokens that every branch starts with."""
        texts = [self.question_text(q) for q in questions]
        full0 = self.encoder.text({"state": state, "questions": {"q": questions[0]}})
        if not full0.endswith(texts[0]):
            raise RuntimeError("question text is not a suffix of its request")
        prefix = full0[: len(full0) - len(texts[0])]
        enc = self.tok(prefix, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        sample = sorted({0, len(questions) - 1, *range(0, len(questions), max(1, len(questions) // checks))})
        for back in (4, 8, 16, 32, 64):
            cut = max(len(ids) - back, self.taps)
            shared, tail = ids[:cut], prefix[offsets[cut][0]:]
            branches = self.tok([tail + t for t in texts], add_special_tokens=False)["input_ids"]
            full = self.tok([prefix + texts[i] for i in sample], add_special_tokens=False)["input_ids"]
            if all(f == shared + branches[i] for f, i in zip(full, sample)):
                break
        else:
            raise RuntimeError("could not split the state from the questions at a stable token boundary")
        reference = self.encoder.encode({"state": state, "questions": {"q": questions[0]}}).input_ids.tolist()
        if reference != shared + branches[0]:
            raise RuntimeError("split tokens differ from the package's encoder")
        if self.marker_id in shared or any(b.count(self.marker_id) != 1 for b in branches):
            raise RuntimeError("expected exactly one readout marker per branch and none in the state")
        return shared, branches, tail

    def shorten(self, shared, branches, tail, instructions, fork_at=None):
        """fast mode: move the question text every branch shares into the state, and end branches at the marker.

        fork_at: the start every instruction shares (e.g. "Is the information ... on Page "); default: the text
        before the first digit, if all instructions share it, else their common prefix cut to a word boundary.
        It depends only on the question template, not on which questions are in the request, so answers stay
        independent of the other questions. Approximations (full-attention layers only): the shared question
        tokens do not see the branch's own tokens, and the template tokens after the marker are dropped.
        """
        if fork_at is None:
            fork_at = fork_text(instructions)
        if not all(t.startswith(fork_at) for t in instructions):
            raise ValueError("every instruction must start with fork_at")
        ids = self.tok(tail + QUESTION_START + fork_at, add_special_tokens=False)["input_ids"]
        n = min(len(ids), min(len(b) for b in branches) - 1)
        while n > 0 and not all(b[:n] == ids[:n] for b in branches):  # last token may merge with the next text
            n -= 1
        cut = [b[n:b.index(self.marker_id) + 1] for b in branches]
        return shared + ids[:n], cut

    # ------------------------------------------------------------------ forward

    def _groups(self, lengths):
        """Consecutive groups of sequences with at most `budget` tokens each (a longer sequence runs alone)."""
        group, used = [], 0
        for i, n in enumerate(lengths):
            if group and used + n > self.budget:
                yield group
                group, used = [], 0
            group.append(i)
            used += n
        if group:
            yield group

    def _linear(self, mod, x, pack, prefix, capture):
        length = x.shape[1]
        if mod.conv1d.bias is not None:
            raise ValueError("unexpected convolution bias")
        mixed = mod.in_proj_qkv(x).transpose(1, 2)
        z = mod.in_proj_z(x).reshape(1, length, -1, mod.head_v_dim)
        b, a = mod.in_proj_b(x), mod.in_proj_a(x)
        init, tail = prefix if prefix is not None else (None, None)
        got_tail = mixed[:, :, -(self.taps - 1):].clone() if capture else None
        _, _, seqpos = pack.on(x.device)
        from transformers.activations import ACT2FN
        mixed = ACT2FN[mod.activation](conv_with_tail(mixed, mod.conv1d.weight.squeeze(1), seqpos, tail))
        mixed = mixed.transpose(1, 2)
        q, k, v = torch.split(mixed, [mod.key_dim, mod.key_dim, mod.value_dim], dim=-1)
        q = q.reshape(1, length, -1, mod.head_k_dim)
        k = k.reshape(1, length, -1, mod.head_k_dim)
        v = v.reshape(1, length, -1, mod.head_v_dim)
        beta = b.sigmoid()
        g = -mod.A_log.float().exp() * F.softplus(a.float() + mod.dt_bias)
        rep = mod.num_v_heads // mod.num_k_heads
        if rep > 1:
            q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
        if init is not None:
            init = init.expand(len(pack.lengths), -1, -1, -1).contiguous()
        core, final = self.kernels.delta(q, k, v, g, beta, pack, init, capture)
        core = mod.norm(core.reshape(-1, mod.head_v_dim), z.reshape(-1, mod.head_v_dim))
        return mod.out_proj(core.reshape(1, length, -1)), ((final, got_tail) if capture else None)

    def _full(self, attn, x, cos, sin, pack, prefix, capture):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        length, hd = x.shape[1], attn.head_dim
        shape = (1, length, -1, hd)
        query, gate = torch.chunk(attn.q_proj(x).view(1, length, -1, hd * 2), 2, dim=-1)
        gate = gate.reshape(1, length, -1)
        query = attn.q_norm(query.view(shape)).transpose(1, 2)
        key = attn.k_norm(attn.k_proj(x).view(shape)).transpose(1, 2)
        value = attn.v_proj(x).view(shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        q, k, v = (t[0].transpose(0, 1).contiguous() for t in (query, key, value))  # (T, H, D)
        if prefix is None:
            out, _ = self.kernels.attention(q, k, v, pack.cu_cpu, pack.cu_cpu, pack.max, pack.max, attn.scaling,
                                            False)
        else:
            pk, pv = prefix
            own, own_lse = self.kernels.attention(q, k, v, pack.cu_cpu, pack.cu_cpu, pack.max, pack.max,
                                                  attn.scaling, True)
            shared, shared_lse = self.kernels.attention(q, pk, pv, torch.tensor([0, length]),
                                                        torch.tensor([0, pk.shape[0]]), length, pk.shape[0],
                                                        attn.scaling, True)
            out = merge_attention(own, own_lse, shared, shared_lse)
        out = attn.o_proj(out.reshape(1, length, -1) * torch.sigmoid(gate))
        return out, ((k, v) if capture else None)

    @torch.inference_mode()
    def _stack(self, ids, positions, pack, prefix=None, capture=False):
        """All decoder layers over a pack. Returns the final-normed hidden states (on the last device) and, with
        capture (one-sequence pack only), every layer's cache for forked branches."""
        if capture and len(pack.lengths) != 1:
            raise ValueError("capture needs a single sequence")
        h = self.lm.embed_tokens(ids.to(self.first))[None]
        cos, sin = self.lm.rotary_emb(h, positions.to(self.first).view(1, 1, -1).expand(3, 1, -1))
        rope, cache = {}, []
        for i, layer in enumerate(self.layers):
            d = self.layer_devices[i]
            h = h.to(d)
            if d not in rope:
                rope[d] = (cos.to(d), sin.to(d))
            c, s = rope[d]
            with torch.cuda.device(d) if d.type == "cuda" else contextlib.nullcontext():
                x = layer.input_layernorm(h)
                p = None if prefix is None else prefix[i]
                if layer.block_type == "linear_attention":
                    x, got = self._linear(layer.linear_attn, x, pack, p, capture)
                else:
                    x, got = self._full(layer.self_attn, x, c, s, pack, p, capture)
                h = h + x
                for s in range(0, h.shape[1], self.mlp_chunk):  # per-token, so chunking changes nothing
                    part = h[:, s:s + self.mlp_chunk]
                    part += layer.mlp(layer.post_attention_layernorm(part))
            cache.append(got)
        return self.lm.norm(h.to(self.last)), (cache if capture else None)

    def _readout(self, hidden, positions):
        return self.readout(hidden[0, positions.to(hidden.device)]).float().cpu()

    def _run_packed(self, seqs, start_positions=None, prefix=None):
        """Logit rows at every marker of every sequence (list per sequence), packing up to `budget` tokens."""
        out = [None] * len(seqs)
        for group in self._groups([len(s) for s in seqs]):
            lengths = [len(seqs[i]) for i in group]
            pack = Pack(lengths)
            flat = torch.tensor([t for i in group for t in seqs[i]])
            offset = 0 if start_positions is None else start_positions
            positions = torch.cat([torch.arange(n) + offset for n in lengths])
            hidden, _ = self._stack(flat, positions, pack, prefix=prefix)
            marks = (flat == self.marker_id).nonzero().flatten()
            logits = self._readout(hidden, marks)
            owner = torch.bucketize(marks, pack.cu_cpu[1:], right=True)
            for j, i in enumerate(group):
                out[i] = logits[owner == j]
        return out

    @torch.inference_mode()
    def _official(self, seq, full_mask=None):
        """The unmodified Transformers forward (the package's reference path) on one sequence."""
        ids = torch.tensor(seq)[None].to(self.first)
        kwargs = {"input_ids": ids, "use_cache": False}
        if full_mask is None:
            kwargs["attention_mask"] = torch.ones_like(ids)
        else:
            kwargs["attention_mask"] = {"full_attention": full_mask[None, None].to(self.first),
                                        "linear_attention": None}
            kwargs["position_ids"] = torch.arange(len(seq), device=self.first).view(1, 1, -1).expand(3, 1, -1)
        hidden = self.backbone(**kwargs).last_hidden_state
        marks = (ids[0] == self.marker_id).nonzero().flatten()
        return self._readout(hidden, marks)

    # ------------------------------------------------------------------ public

    def score(self, state, instructions, mode="forked", fork_at=None):
        """Yes/no questions (instruction strings) about `state`.

        mode "fast" is forked with shorter branches (see shorten(); fork_at is passed to it): ~20 tokens per
        question instead of ~52, so about 2.5x fewer tokens when there are many short candidates.
        Returns logits [[false, true], ...], margin (true - false, for ranking), p (calibrated P(true)),
        tokens (tokens run through the model), seconds.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        questions = [{"type": "noul", "instructions": t} for t in instructions]
        self._sync()
        t0 = time.perf_counter()
        if mode == "joint":
            plan = self.encoder.plan({"state": state, "questions": {f"q{i}": q for i, q in enumerate(questions)}},
                                     self.joint_max_length)
            seqs = [c.input_ids.tolist() for c in plan]
            rows = torch.cat(self._run_packed(seqs))
            tokens, sequences = sum(len(s) for s in seqs), len(seqs)
        else:
            shared, branches, tail = self._split(state, questions)
            if mode == "fast":
                shared, branches = self.shorten(shared, branches, tail, instructions, fork_at)
            if mode in ("forked", "fast"):
                _, prefix = self._stack(torch.tensor(shared), torch.arange(len(shared)), Pack([len(shared)]),
                                        capture=True)
                rows = torch.cat(self._run_packed(branches, start_positions=len(shared), prefix=prefix))
                tokens, sequences = len(shared) + sum(len(b) for b in branches), 1
            elif mode == "exact":
                rows = torch.cat(self._run_packed([shared + b for b in branches]))
                tokens, sequences = sum(len(shared) + len(b) for b in branches), len(branches)
            else:
                rows = []
                for b in branches:
                    mask = None
                    if mode == "official_masked":
                        n, p = len(shared) + len(b), len(shared)
                        mask = torch.ones(n, n, dtype=torch.bool)
                        mask[:p, p:] = False  # the state does not see the question
                    rows.append(self._official(shared + b, mask))
                rows = torch.cat(rows)
                tokens, sequences = sum(len(shared) + len(b) for b in branches), len(branches)
        self._sync()
        seconds = time.perf_counter() - t0
        rows = rows[:, :2]
        p = (rows / self.temperature).softmax(-1)[:, 1]
        return {"mode": mode, "logits": rows.tolist(), "margin": (rows[:, 1] - rows[:, 0]).tolist(), "p": p.tolist(),
                "tokens": tokens, "sequences": sequences, "seconds": seconds}

    def _sync(self):
        for d in set(self.layer_devices):
            if d.type == "cuda":
                torch.cuda.synchronize(d)
