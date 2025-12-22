from __future__ import annotations

import os
import re
from pathlib import Path

APP_NAME = "Agent Service"
APP_VERSION = "3.2.0"

PORT = int(os.getenv("PORT", "9000"))
BIND = os.getenv("BIND", "0.0.0.0")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
API_KEY = os.getenv("API_KEY", "").strip()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAX_CONCURRENCY = max(1, int(os.getenv("MAX_CONCURRENCY", "4")))
MAX_CHROME_CONCURRENCY = max(1, int(os.getenv("MAX_CHROME_CONCURRENCY", "2")))

STRIP_ANSI = bool(int(os.getenv("STRIP_ANSI_IN_DB", "1")))
COLLAPSE_INTERNAL_SPACES = bool(int(os.getenv("COLLAPSE_INTERNAL_SPACES", "0")))
DEFAULT_AUTHOR_ID_RAW = os.getenv("DEFAULT_AUTHOR_ID", "").strip()
DEFAULT_AUTHOR_ID = int(DEFAULT_AUTHOR_ID_RAW) if DEFAULT_AUTHOR_ID_RAW.isdigit() else None

DB_URL = os.getenv("DB_URL", "").strip()

SCREENSHOTS_BASE = Path(os.getenv("SCREENSHOTS_BASE", "./screenshots")).resolve()
SCREENSHOTS_BASE.mkdir(parents=True, exist_ok=True)

RECORDINGS_BASE = Path("./screenshots").resolve()
RECORDINGS_BASE.mkdir(parents=True, exist_ok=True)


def find_chrome_binary() -> str:
    env = os.getenv("BROWSER_USE_CHROME_PATH")
    if env and Path(env).exists():
        return env
    for candidate in (
        "/usr/bin/google-chrome-stable",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/snap/bin/chromium",
    ):
        if Path(candidate).exists():
            return candidate
    raise RuntimeError("Chrome/Chromium not found. Set BROWSER_USE_CHROME_PATH or install Chromium.")


CHROME_BIN = find_chrome_binary()
BASE_PROFILE_DIR = Path("/tmp/agent-profiles")
BASE_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
CTRL_ZW_RE = re.compile("[" + "\u200B\u200C\u200D\u200E\u200F" + "\u2060" + "\uFEFF" + "]")
STEP_IMG_RE = re.compile(r"step_(\d{1,3})_.*\.(png|jpg|jpeg)$", re.IGNORECASE)
