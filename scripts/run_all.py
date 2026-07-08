"""Execute every RCG-Bench notebook in order (local or CI).

Runs each notebook with nbconvert, writing executed copies to results/executed/.
On a machine without a GPU/HF token, set RCG_MODEL_ID to a small CPU model, e.g.:

    RCG_MODEL_ID=sshleifer/tiny-gpt2 RCG_SKIP_INSTALL=1 python scripts/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
OUT = ROOT / "results" / "executed"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    notebooks = sorted(NB.glob("*.ipynb"))
    if not notebooks:
        print("No notebooks found. Run: python scripts/build_notebooks.py")
        return
    failures = []
    for path in notebooks:
        print(f"=== executing {path.name} ===")
        proc = subprocess.run(
            [
                sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
                "--output-dir", str(OUT), str(path),
            ],
            cwd=ROOT,
        )
        if proc.returncode != 0:
            failures.append(path.name)
    if failures:
        print("\nFAILED:", ", ".join(failures))
        sys.exit(1)
    print(f"\nAll {len(notebooks)} notebooks executed. Outputs in {OUT}")


if __name__ == "__main__":
    main()
