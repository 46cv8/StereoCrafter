"""Shared helpers for deterministic clip ordering across tools.

Sort policy:
1) Clips with a parsed clip/scene id come first, ascending by numeric id.
2) Ties (or unparsable names) fall back to case-insensitive lexical order.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

PathLike = Union[str, os.PathLike]

# Prefer explicit scene/clip tokens, then hyphen/underscore id groups, then any standalone digits.
_CLIP_ID_PATTERNS = [
    re.compile(r"(?i)(?:scene|clip)[-_\s]*0*([0-9]+)(?=[^0-9]|$)"),
    re.compile(r"(?i)[-_]0*([0-9]{1,8})(?=[^0-9]|$)"),
    re.compile(r"(?<![0-9])([0-9]{1,8})(?![0-9])"),
]


def extract_clip_id(value: PathLike) -> Optional[int]:
    """Best-effort extraction of clip id from a file/folder name."""
    text = Path(str(value)).name
    stem = os.path.splitext(text)[0]
    for pattern in _CLIP_ID_PATTERNS:
        match = pattern.search(stem)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                continue
    return None


def clip_sort_key(value: PathLike) -> Tuple[int, int, str, str]:
    """Deterministic key with clip-id priority, then lexical fallback."""
    path_text = str(value)
    name = Path(path_text).name
    stem = os.path.splitext(name)[0]
    clip_id = extract_clip_id(name)
    if clip_id is None:
        # Put unparsable names after parsed clip ids.
        return (1, 2**31 - 1, stem.lower(), name.lower())
    return (0, int(clip_id), stem.lower(), name.lower())


def sort_paths_by_clip_id(values: Iterable[PathLike]) -> List[str]:
    """Return a clip-id ordered list preserving all input entries."""
    out = [str(v) for v in values]
    out.sort(key=clip_sort_key)
    return out
