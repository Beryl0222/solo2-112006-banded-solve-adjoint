"""Pytest/unittest 兼容引导：确保仓库根目录在 sys.path 上。"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
