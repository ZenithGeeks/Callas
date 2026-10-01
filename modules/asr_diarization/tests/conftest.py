import sys
from pathlib import Path

MODULES = str(Path(__file__).resolve().parents[2])
if MODULES not in sys.path:
    sys.path.insert(0, MODULES)
