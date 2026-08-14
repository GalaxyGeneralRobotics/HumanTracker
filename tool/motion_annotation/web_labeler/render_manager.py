"""Background pre-rendering manager.

Renders all clip MP4s into the disk cache ahead of time on a dedicated worker
thread, so that by the time an annotator navigates to a pair the videos are
already on disk and play instantly. The pair the user is currently looking at
(and its immediate neighbours) is pushed to the front of the queue.

Rendering is serialized on a single worker thread because the MuJoCo renderer
and its EGL context are not thread-safe and are bound to their creating thread.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from web_labeler.video_cache import cache_filename, get_or_render


@dataclass
class RenderJob:
    npz_path: Path
    start: int
    end: int
    side_info: Optional[dict] = None
    key: str = field(default="", init=False)

    def __post_init__(self):
        self.key = cache_filename(self.npz_path, self.start, self.end)


class RenderManager:
    """Owns a background worker that renders clips into ``cache_dir``."""

    def __init__(
        self,
        cache_dir: Path | str,
        scene_xml: Optional[str] = None,
        prefetch_all: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.scene_xml = scene_xml
        self.prefetch_all = bool(prefetch_all)

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

        self._jobs: Dict[str, RenderJob] = {}   # key -> job (deduplicated)
        self._order: List[str] = []             # default render order
        self._priority: List[str] = []          # keys to render first (front = next)
        self._done: set[str] = set()
        self._failed: Dict[str, str] = {}
        self._current: Optional[str] = None

        self._worker: Optional[threading.Thread] = None
        self._stop = False

    # -- registration -------------------------------------------------------
    def add_jobs(self, jobs: List[RenderJob]) -> None:
        """Register clips to render (idempotent; duplicates are merged)."""
        with self._cv:
            for job in jobs:
                if job.key in self._jobs:
                    continue
                self._jobs[job.key] = job
                self._order.append(job.key)
                p = self.cache_dir / job.key
                if p.exists() and p.stat().st_size > 0:
                    self._done.add(job.key)
            self._cv.notify_all()

    def start(self) -> None:
        if self._worker is not None:
            return
        self._worker = threading.Thread(
            target=self._run, name="render-worker", daemon=True
        )
        self._worker.start()

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    # -- queries / scheduling ----------------------------------------------
    def _refresh_disk_ready_unlocked(self, key: str) -> bool:
        """Import a file completed by an external pre-render process."""
        if key not in self._jobs:
            return False
        path = self.cache_dir / key
        try:
            ready = path.is_file() and path.stat().st_size > 0
        except OSError:
            ready = False
        if ready:
            self._done.add(key)
            self._failed.pop(key, None)
        return ready

    def has_job(self, key: str) -> bool:
        with self._lock:
            return key in self._jobs

    def is_ready(self, key: str) -> bool:
        with self._lock:
            return key in self._done or self._refresh_disk_ready_unlocked(key)

    def error_for(self, key: str) -> Optional[str]:
        with self._lock:
            if self._refresh_disk_ready_unlocked(key):
                return None
            return self._failed.get(key)

    def prioritize(self, keys: List[str]) -> None:
        """Bump ``keys`` to the front of the queue (first key rendered first)."""
        with self._cv:
            for k in reversed(keys):
                if k not in self._jobs or k in self._done:
                    continue
                if k in self._priority:
                    self._priority.remove(k)
                self._priority.insert(0, k)
            self._cv.notify_all()

    def progress(self) -> Dict[str, int]:
        with self._lock:
            # Offline workers publish into the same cache, so refresh the
            # in-memory view before reporting progress to the UI.
            for key in self._jobs:
                if key not in self._done:
                    self._refresh_disk_ready_unlocked(key)
            return {
                "total": len(self._jobs),
                "done": len(self._done),
                "failed": len(self._failed),
            }

    # -- worker -------------------------------------------------------------
    def _next_key(self) -> Optional[str]:
        # Must be called holding the lock.
        while self._priority:
            k = self._priority.pop(0)
            if self._refresh_disk_ready_unlocked(k):
                continue
            if k in self._jobs and k not in self._done and k not in self._failed:
                return k
        # When a dedicated multi-GPU pre-renderer owns the full scan, the web
        # process only handles pair keys explicitly prioritized by a browser.
        if not self.prefetch_all:
            return None
        for k in self._order:
            if self._refresh_disk_ready_unlocked(k):
                continue
            if k not in self._done and k not in self._failed:
                return k
        return None

    def _run(self) -> None:
        while True:
            with self._cv:
                if self._stop:
                    return
                key = self._next_key()
                while key is None and not self._stop:
                    self._cv.wait()
                    key = self._next_key()
                if self._stop:
                    return
                job = self._jobs[key]
                self._current = key

            try:
                t0 = time.time()
                get_or_render(
                    job.npz_path, job.start, job.end, self.cache_dir,
                    self.scene_xml, side_info=job.side_info,
                )
                dt = time.time() - t0
                with self._lock:
                    self._done.add(key)
                    self._current = None
                done = len(self._done)
                total = len(self._jobs)
                print(f"[render] {done}/{total} {key} ({dt:.2f}s)")
            except Exception as e:  # noqa: BLE001 - keep worker alive
                with self._lock:
                    self._failed[key] = str(e)
                    self._current = None
                print(f"[render] FAILED {key}: {e}")
                traceback.print_exc()
