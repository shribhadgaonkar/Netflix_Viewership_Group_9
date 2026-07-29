from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = [
    REPO_ROOT / "scripts" / "data_preparation" / "prepare_netflix.py",
    REPO_ROOT / "scripts" / "data_preparation" / "prepare_imdb.py",
    REPO_ROOT / "scripts" / "data_preparation" / "merge_datasets.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"[pipeline] Running {script.relative_to(REPO_ROOT).as_posix()}")
        subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
