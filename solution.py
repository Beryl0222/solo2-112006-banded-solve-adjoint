# Compatibility alias module; authoritative implementation is band_solver.
from band_solver import *  # noqa: F401,F403
import band_solver as _bs

for _name in dir(_bs):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_bs, _name)
