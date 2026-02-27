"""Display bridge for core analysis.

Core code stays GUI-agnostic by calling this thin adapter. If web plotting
helpers are available, calls are forwarded so ``show=True`` displays figures.
If not, behavior degrades to a one-time warning.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional

_WARNED = False
_PLOTTING = None


def _warn_once(msg: str) -> None:
    global _WARNED
    if not _WARNED:
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        _WARNED = True


def _get_plotting():
    global _PLOTTING
    if _PLOTTING is not None:
        return _PLOTTING
    try:
        from ..web import plotting as mod
    except Exception:
        _PLOTTING = False
    else:
        _PLOTTING = mod
    return _PLOTTING


def _dispatch(fn_name: str, *args, **kwargs):
    mod = _get_plotting()
    if mod is False:
        _warn_once(
            "Web plotting helpers are unavailable; install plotly to enable "
            "visualization for show=True."
        )
        return None
    fn: Optional[Callable] = getattr(mod, fn_name, None)
    if fn is None:
        _warn_once(f"Plotting function '{fn_name}' is unavailable.")
        return None
    return fn(*args, **kwargs)


def show_lines(*args, **kwargs):
    return _dispatch("show_lines", *args, **kwargs)


def show_line(*args, **kwargs):
    return _dispatch("show_line", *args, **kwargs)


def show_image(*args, **kwargs):
    return _dispatch("show_image", *args, **kwargs)


def show_image_stack(*args, **kwargs):
    return _dispatch("show_image_stack", *args, **kwargs)


def show_mask_overlay(*args, **kwargs):
    return _dispatch("show_mask_overlay", *args, **kwargs)

