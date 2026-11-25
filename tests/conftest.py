"""Test setup.

- Makes the app/ folder importable (mirrors how Streamlit runs the app).
- Points the app at a temporary SQLite file so tests never touch data/agrisort.db.
"""

import os
import sys
import tempfile
from pathlib import Path

_TEST_DB = Path(tempfile.mkdtemp(prefix="agrisort-tests-")) / "test.db"
os.environ["AGRISORT_DB_PATH"] = str(_TEST_DB)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
