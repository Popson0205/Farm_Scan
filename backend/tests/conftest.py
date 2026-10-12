import os
import sys
import tempfile
from pathlib import Path

# isolate the database BEFORE any app module is imported
_TMP = tempfile.mkdtemp(prefix="farmscan_test_")
os.environ["FARMSCAN_DB"] = str(Path(_TMP) / "test.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
