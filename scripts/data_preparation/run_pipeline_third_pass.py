from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "data_preparation" / "third_pass_multi_grain.py"


def main() -> None:
    print(f"[pipeline] Running {SCRIPT.relative_to(REPO_ROOT).as_posix()}")
    subprocess.run([sys.executable, str(SCRIPT)], cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
