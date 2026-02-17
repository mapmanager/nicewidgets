"""File manager utility for NiceGUI apps.

Reveal a path in the OS file manager (Finder/Explorer/etc).
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path


def reveal_in_file_manager(path: str | os.PathLike) -> None:
    """Reveal a path in the OS file manager (Finder/Explorer/etc).

    - macOS: Finder reveals + selects the item
    - Windows: Explorer reveals + selects the item
    - Linux: opens the containing folder (selection support varies)
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))

    system = platform.system()

    if system == "Darwin":
        # Finder reveal (select)
        subprocess.run(["open", "-R", str(p)], check=False)

    elif system == "Windows":
        # Explorer reveal (select)
        subprocess.run(["explorer", f'/select,"{p}"'], check=False, shell=True)

    else:
        # Linux: open folder (best-effort)
        folder = p if p.is_dir() else p.parent
        subprocess.run(["xdg-open", str(folder)], check=False)
