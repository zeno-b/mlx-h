"""Allows `python -m local_llm` to run the app exactly like the old script.

`python -m` imports the package (whose __init__ imports everything, core
included) before this file runs, so core's import-time defaults (DEFAULT_MODEL
from MODEL_ID, APP_NAME, the login throttle...) are already computed when .env
could first be read here. When .env sets anything new, re-exec the same
command once with those variables in the environment, so `python -m local_llm`
and `python3 deploy.py` see the same configuration. The loader marks the
environment, so the re-executed process does not load or re-exec again.
"""

from __future__ import annotations

import os
import sys

from .envfile import DOTENV_LOADED_VAR, load_dotenv

if __name__ == "__main__" and not os.environ.get(DOTENV_LOADED_VAR):
    if load_dotenv():
        argv = list(getattr(sys, "orig_argv", []) or [])
        if not argv or "-m" not in argv:
            argv = [sys.executable, "-m", __package__ or "local_llm", *sys.argv[1:]]
        os.execv(sys.executable, [sys.executable, *argv[1:]])

from .cli import main  # noqa: E402

if __name__ == "__main__":
    main()
