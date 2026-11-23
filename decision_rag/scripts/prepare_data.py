"""Step 1: download DAPR NQ, build the page and paragraph tables and first sentences, write splits/nq_test500.txt.
CPU only. Run with slurm/prepare.sub."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decision_rag.config import load_config  # noqa: E402
from decision_rag.data import prepare  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    prepare(load_config(), workers=args.workers)
