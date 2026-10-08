"""Windows / Hub download helpers for LeRobot scripts."""

from __future__ import annotations

import functools
import os
import time
from pathlib import Path
from typing import Any


def prepare_hf_hub_env() -> None:
    """Avoid WinError 1314 symlink races during parallel Hub downloads.

    Hugging Face Hub tests symlink support per cache directory. LeRobot uses a
    separate cache under ``HF_LEROBOT_HOME/hub``. Concurrent snapshot threads can
    briefly treat that cache as symlink-capable on Windows without Developer
    Mode, then fail with privilege errors mid-download.
    """
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if os.name != "nt":
        return

    try:
        from huggingface_hub.file_download import (
            _are_symlinks_supported_in_dir,
            are_symlinks_supported,
        )
        from huggingface_hub import constants as hf_constants
    except ImportError:
        return

    cache_dirs: list[str] = [str(Path(hf_constants.HF_HUB_CACHE).expanduser())]
    try:
        from lerobot.utils.constants import HF_LEROBOT_HOME, HF_LEROBOT_HUB_CACHE

        cache_dirs.append(str(Path(HF_LEROBOT_HUB_CACHE).expanduser()))
        cache_dirs.append(str(Path(HF_LEROBOT_HOME).expanduser()))
    except ImportError:
        pass

    for raw in cache_dirs:
        path = Path(raw)
        path.mkdir(parents=True, exist_ok=True)
        resolved = str(path.resolve())
        # Force copy/move fallback before any parallel download starts.
        _are_symlinks_supported_in_dir[resolved] = False
        # Also run the official probe so nested paths inherit correctly.
        are_symlinks_supported(resolved)
        _are_symlinks_supported_in_dir[resolved] = False


def _is_rate_limit(exc: BaseException) -> bool:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


def _wrap_snapshot_download(orig, *, max_workers: int, retries: int, base_wait_s: float):
    @functools.wraps(orig)
    def wrapped(*args: Any, **kwargs: Any):
        kwargs.setdefault("max_workers", max_workers)
        last_exc: BaseException | None = None
        for attempt in range(retries):
            try:
                return orig(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — Hub raises several types
                last_exc = exc
                if not _is_rate_limit(exc) or attempt >= retries - 1:
                    raise
                wait = base_wait_s * (attempt + 1)
                print(
                    f"[hf] Rate limited (429). Waiting {wait:.0f}s then retry "
                    f"{attempt + 1}/{retries} with max_workers={kwargs.get('max_workers')} "
                    "(resume continues from cache)...",
                    flush=True,
                )
                time.sleep(wait)
        assert last_exc is not None
        raise last_exc

    return wrapped


def install_hf_download_guards(
    *,
    max_workers: int = 1,
    retries: int = 15,
    base_wait_s: float = 35.0,
) -> None:
    """Throttle Hub downloads and retry on 429.

    Community datasets have ~1k meta parquet files; the default parallel
    snapshot_download burns the free-tier quota (1000 API calls / 5 min).
    Call this after importing ``lerobot`` so already-bound ``snapshot_download``
    references are patched too.
    """
    configured_workers = os.environ.get("SO101_HF_DOWNLOAD_WORKERS")
    if configured_workers is not None:
        try:
            max_workers = int(configured_workers)
        except ValueError as exc:
            raise ValueError("SO101_HF_DOWNLOAD_WORKERS must be a positive integer") from exc
    if max_workers < 1:
        raise ValueError("Hub download workers must be positive")
    prepare_hf_hub_env()

    import huggingface_hub
    import huggingface_hub._snapshot_download as snap_mod

    if getattr(huggingface_hub.snapshot_download, "_robot101_guarded", False):
        return

    orig = snap_mod.snapshot_download
    guarded = _wrap_snapshot_download(
        orig, max_workers=max_workers, retries=retries, base_wait_s=base_wait_s
    )
    guarded._robot101_guarded = True  # type: ignore[attr-defined]

    snap_mod.snapshot_download = guarded
    huggingface_hub.snapshot_download = guarded

    # LeRobot binds the symbol at import time — patch those modules too.
    for mod_name in (
        "lerobot.datasets.dataset_metadata",
        "lerobot.datasets.lerobot_dataset",
    ):
        try:
            mod = __import__(mod_name, fromlist=["snapshot_download"])
        except ImportError:
            continue
        if hasattr(mod, "snapshot_download"):
            mod.snapshot_download = guarded

    print(
        f"[hf] Hub downloads: max_workers={max_workers}, "
        f"429 retries={retries} (x{base_wait_s:.0f}s backoff)",
        flush=True,
    )
