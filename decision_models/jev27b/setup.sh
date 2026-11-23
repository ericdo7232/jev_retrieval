#!/usr/bin/env bash
# Install vLLM for JEV-27B and download the weights (~55 GB). Run once, on a login node or CPU job.
#
# JEV-27B was tested on "a vLLM development build from September 2026"; no exact version is published.
# It needs Qwen3.5 (qwen3_5) support, LoRA on lm_head, --logprobs-mode and allowed_token_ids.
# If the check at the end fails, rerun with a newer build, for example:
#   VLLM_SPEC="vllm --pre --extra-index-url https://wheels.vllm.ai/nightly" bash jev27b/setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh"

command -v uv >/dev/null || { echo "uv not found. Install it first: https://docs.astral.sh/uv/"; exit 1; }
D="$WORK_DIR/jev27b"
mkdir -p "$D"
cd "$D"

[ -d .venv ] || uv venv --python 3.12 .venv
VLLM_SPEC="${VLLM_SPEC:-vllm}"
# shellcheck disable=SC2086  # VLLM_SPEC may hold extra pip flags on purpose
uv pip install --python .venv/bin/python $VLLM_SPEC "huggingface_hub[cli]"

.venv/bin/hf download autotrust/JEV-27B --local-dir "$D/JEV-27B"

echo "Checking that this vLLM has what serve_decide.py needs ..."
.venv/bin/python - "$D/JEV-27B" <<'EOF'
import importlib, json, re, sys
d = sys.argv[1]
ok = True
src = open(f"{d}/serve_decide.py").read()
mods = sorted(set(re.findall(r"^\s*from\s+(vllm[\w.]*)\s+import", src, re.M)
                  + re.findall(r"^\s*import\s+(vllm[\w.]*)", src, re.M)))
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  ok      {m}")
    except Exception as e:
        ok = False
        print(f"  MISSING {m}: {type(e).__name__}: {e}")
try:
    from vllm import ModelRegistry
    archs = json.load(open(f"{d}/config.json")).get("architectures", [])
    supported = set(ModelRegistry.get_supported_archs())
    for a in archs:
        print(f"  {'ok     ' if a in supported else 'MISSING'} architecture {a}")
        ok = ok and a in supported
except Exception as e:
    ok = False
    print(f"  could not check architectures: {e}")
import vllm
print(f"vLLM {vllm.__version__}: {'looks compatible' if ok else 'NOT compatible; see the note at the top of jev27b/setup.sh'}")
sys.exit(0 if ok else 1)
EOF
echo "JEV-27B setup done."
