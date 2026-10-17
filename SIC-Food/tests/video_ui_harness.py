"""Compatibility fixture entry; uses the current main app harness."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('ui_harness.py')), run_name='__main__')
