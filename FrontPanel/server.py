from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOT_SERVER = ROOT / "server.py"

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    runpy.run_path(str(ROOT_SERVER), run_name="__main__")
