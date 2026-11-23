"""CPU tests of pplx/fp8.py and of the scorer's FP8 and chunked-MLP paths (no GPU needed).

The scorer tests use a tiny random Qwen3.5 with the real tokenizer and template. They need the checkpoint's small
files (tokenizer, configs, source/): set PPLX_TEST_CKPT to a directory holding them, or they are skipped.
Run:  PPLX_TEST_CKPT=... python -m pytest -q tests
"""
import os
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "pplx"))
import fp8  # noqa: E402


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("mode", fp8.MODES)
def test_fp8_linear_close_to_linear(mode):
    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 512, bias=False).to(torch.bfloat16)
    q = fp8.FP8Linear(lin, mode)
    assert q.weight.dtype == fp8.E4M3 and q.weight.shape == (512, 256)
    for shape in [(17, 256), (2, 33, 256), (1, 256)]:  # row counts that need padding to 16, and a 3-D input
        x = torch.randn(*shape, dtype=torch.bfloat16)
        y = q(x)
        assert y.shape == (*shape[:-1], 512) and y.dtype == torch.bfloat16
        assert rel_err(y, lin(x)) < 0.06  # e4m3 has a 3-bit mantissa: a few percent per matrix multiply


@pytest.mark.parametrize("mode", fp8.MODES)
def test_fp8_rows_do_not_depend_on_each_other(mode):
    """A token's output must not change with the other tokens in the batch (else forked answers would depend on the
    other questions packed with them), even next to a token with huge activations."""
    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 256, bias=False).to(torch.bfloat16)
    q = fp8.FP8Linear(lin, mode)
    x = torch.randn(32, 256, dtype=torch.bfloat16)
    alone = q(x[5:6])
    x[0] *= 1000
    assert torch.allclose(q(x)[5:6], alone, rtol=1e-2, atol=1e-2)
    assert rel_err(q(x)[1:], lin(x)[1:]) < 0.06


def test_emulated_rowwise_matches_native():
    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 512, bias=False).to(torch.bfloat16)
    x = torch.randn(40, 256, dtype=torch.bfloat16)
    assert rel_err(fp8.FP8Linear(lin, "rowwise_emulated")(x), fp8.FP8Linear(lin, "rowwise")(x)) < 1e-2


def test_fp8_scale_is_exact_in_bf16():
    x = torch.randn(8, 64, dtype=torch.bfloat16) * 3.7
    s = fp8._scale(x)
    assert torch.equal(s, s.to(torch.bfloat16).float())


def test_quantize_replaces_only_large_projections():
    m = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.Sequential(torch.nn.Linear(512, 48)),
                            torch.nn.Linear(48, 256), torch.nn.LayerNorm(256))
    assert fp8.quantize(m, "rowwise", min_features=256) == 2
    assert isinstance(m[0], fp8.FP8Linear) and isinstance(m[2], fp8.FP8Linear)
    assert isinstance(m[1][0], torch.nn.Linear)  # 48 outputs: kept
    assert m(torch.randn(3, 256)).shape == (3, 256)


def test_supported_on_cpu():
    assert fp8.supported("cpu") in fp8.MODES


# ---------------------------------------------------------------- scorer with a tiny model

CKPT = os.environ.get("PPLX_TEST_CKPT")
needs_ckpt = pytest.mark.skipif(not CKPT or not Path(CKPT, "source").exists(),
                                reason="set PPLX_TEST_CKPT to the checkpoint's small files")


def tiny_scorer(**kw):
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    from scorer import Scorer
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(CKPT)
    t = cfg.text_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 64, 128, 8
    t.layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 32
    t.linear_num_key_heads, t.linear_num_value_heads, t.linear_key_head_dim, t.linear_value_head_dim = 2, 4, 16, 16
    v = cfg.vision_config
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads, v.out_hidden_size = 1, 32, 64, 2, 64
    cfg._attn_implementation = "sdpa"
    model = Qwen3_5Model(cfg).float()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name or "dt_bias" in name:
                p.add_(0.1 * torch.randn_like(p))
    readout = torch.nn.Linear(64, 255, bias=False)
    torch.nn.init.normal_(readout.weight, std=0.5)
    return Scorer(CKPT, backbone=model, readout=readout, backend="torch", devices=["cpu"], budget=900,
                  dtype=torch.float32, **kw)


@pytest.fixture(scope="module")
def workload():
    from common import NEEDLES, needle_workload
    w = needle_workload(NEEDLES[1], 12, seed=3)
    return w["state"], w["questions"]


def max_dmargin(a, b):
    return max(abs(x - y) for x, y in zip(a["margin"], b["margin"]))


@needs_ckpt
def test_mlp_chunking_changes_nothing(workload):
    state, qs = workload
    whole = tiny_scorer().score(state, qs, "forked")
    chunked = tiny_scorer(mlp_chunk=7).score(state, qs, "forked")
    assert max_dmargin(whole, chunked) < 1e-4


@needs_ckpt
@pytest.mark.parametrize("mode", fp8.MODES)
def test_fp8_scorer_uses_fp8_everywhere_and_stays_isolated(workload, mode, monkeypatch):
    state, qs = workload
    full = tiny_scorer().score(state, qs, "forked")
    monkeypatch.setattr(fp8, "supported", lambda device: mode)
    s = tiny_scorer(fp8=True, fp8_min_features=16)
    assert s.fp8_mode == mode
    assert s.fp8_layers == 8 * 3 + 2 * 4 + 6 * 3  # MLP in every layer, q/k/v/o in full, qkv/z/out in linear
    assert str(next(s.backbone.visual.parameters()).device) == "cpu"
    forked = s.score(state, qs, "forked")
    # This random tiny model is chaotic under FP8: float32 round-off of 1e-7 between the two code paths flips FP8
    # roundings and moves margins by ~0.1 (on the real model the GPU smoke test bounds this). So the check is that
    # the forked code and the Transformers forward both run the FP8 modules: their gap is well below FP8's own error.
    fp8_error = max_dmargin(forked, full)
    assert max_dmargin(forked, s.score(state, qs, "official_masked")) < 0.5 * fp8_error
    # Isolation is exact: per-token scales, so the same questions in another subset and order give the same answers.
    idx = [9, 4, 0, 7, 2]
    sub = s.score(state, [qs[i] for i in idx], "forked")
    assert max(abs(forked["margin"][i] - m) for i, m in zip(idx, sub["margin"])) < 1e-4


# ---------------------------------------------------------------- fast mode (shortened branches)

def test_fork_text():
    from scorer import fork_text
    page = "Is the information needed to answer the user's question on Page {}?"
    assert fork_text([page.format(n) for n in (17, 3, 250)]) == page.split("{}")[0]
    assert fork_text([page.format(12), page.format(17)]) == page.split("{}")[0]  # not "... Page 1"
    assert fork_text([page.format(5)]) == page.split("{}")[0]                     # same for one question
    assert fork_text(["Is it red?", "Is it blue?"]) == "Is it "


@needs_ckpt
def test_fast_matches_transformers_with_the_same_shortcuts_and_is_isolated(workload):
    state, qs = workload
    s = tiny_scorer()
    fast = s.score(state, qs, "fast")
    # Reference: the unmodified Transformers forward on each shortened request, with the shared tokens (state and
    # shared question start) not attending to the branch.
    questions = [{"type": "noul", "instructions": t} for t in qs]
    shared, branches, tail = s._split(state, questions)
    shared, cut = s.shorten(shared, branches, tail, qs)
    assert all(len(c) <= 22 for c in cut) and all(len(b) >= 48 for b in branches)
    ref = []
    for b in cut:
        n, p = len(shared) + len(b), len(shared)
        mask = torch.ones(n, n, dtype=torch.bool)
        mask[:p, p:] = False
        ref.append(s._official(shared + b, mask))
    ref = torch.cat(ref)[:, :2]
    assert max(abs(m - (r[1] - r[0]).item()) for m, r in zip(fast["margin"], ref)) < 1e-4
    # Isolation: the shared question start depends only on the template, so subsets give the same answers.
    idx = [9, 4, 0, 7, 2]
    sub = s.score(state, [qs[i] for i in idx], "fast")
    assert max(abs(fast["margin"][i] - m) for i, m in zip(idx, sub["margin"])) < 1e-4
    # And it stays near forked on this random model (the approximation is in the full-attention layers only).
    forked = s.score(state, qs, "forked")
    assert fast["tokens"] < 0.75 * forked["tokens"]
