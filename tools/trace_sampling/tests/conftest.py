import sys
from pathlib import Path

# The trace_sampling modules import each other by bare name.
_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
