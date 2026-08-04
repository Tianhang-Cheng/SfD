"""Bridge to the repo-root download_assets helper.

Lets superpoint.py / superglue.py pull their checkpoints from the Hugging Face Hub on first use, no
matter which directory the preprocessing scripts are launched from. The import is deferred to call
time so that merely importing this package never fails.
"""

import sys
from pathlib import Path

__all__ = ['ensure_keypoint_weights']


def ensure_keypoint_weights(force=False):
    """Download the SuperPoint/SuperGlue checkpoints into weights/ if they are missing."""
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from download_assets import ensure_keypoint_weights as _ensure
    return _ensure(force=force)
