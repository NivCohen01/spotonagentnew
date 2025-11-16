from __future__ import annotations

import asyncio
import inspect
import json
import socket
import time
from typing import Any

import httpx

from .service_config import BASE_PROFILE_DIR, CHROME_BIN


def pick_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def wait_for_cdp_json_version(port: int, timeout_s: float = 25.0) -> dict:
    url = f"http://127.0.0.1:{port}/json/version"
    start = time.time()
    async with httpx.AsyncClient() as client:
        while time.time() - start < timeout_s:
            try:
                resp = await client.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                pass
            await asyncio.sleep(0.2)
    raise RuntimeError(f"CDP /json/version not ready on port {port}")


async def launch_chrome(sess) -> str:
    port = pick_free_port()
    profile = BASE_PROFILE_DIR / f"profile-{sess.id}"
    profile.mkdir(parents=True, exist_ok=True)
    cmd = [
        CHROME_BIN,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-sync",
        "--disable-dev-shm-usage",
        "--disable-gpu",
    ]
    if sess.headless:
        cmd.append("--headless=new")
    sess.cdp_port = port
    sess.chrome_proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    ver = await wait_for_cdp_json_version(port, timeout_s=25.0)
    ws = ver.get("webSocketDebuggerUrl")
    if not ws:
        raise RuntimeError(f"webSocketDebuggerUrl missing: {json.dumps(ver)}")
    sess.cdp_ws_url = ws
    return ws


async def _safe_browser_stop(browser: Any) -> None:
    try:
        for name in ("stop", "kill", "close"):
            method = getattr(browser, name, None)
            if not callable(method):
                continue
            res = method()
            if inspect.isawaitable(res):
                await res
            return
    except Exception:
        pass
