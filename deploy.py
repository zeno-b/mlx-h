#!/usr/bin/env python3
"""Entry point. Keeps `python3 deploy.py ...` working after the split.

Every flag behaves exactly as before; the implementation now lives in the
local_llm package next to this file.

Before the package is imported, ./.env (or LOCAL_LLM_ENV_FILE) is applied to
the environment, because local_llm.core computes DEFAULT_MODEL and other
defaults at import time. Variables already set in the environment win. The
loader is imported by file path: importing it as local_llm.envfile would run
the package __init__, which imports everything, before the file was read.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_env_file() -> None:
    try:
        path = Path(__file__).resolve().parent / "local_llm" / "envfile.py"
        spec = importlib.util.spec_from_file_location("_local_llm_envfile_boot", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.load_dotenv()
    except Exception as exc:  # noqa: BLE001 - a broken .env must never stop the app
        print(f"[local-llm] .env not loaded: {exc}", file=sys.stderr)


_load_env_file()

from local_llm.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
