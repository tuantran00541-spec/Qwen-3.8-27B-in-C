#!/usr/bin/env python3
"""Windows composition of the proven exact Qwen3.8 current-best quant stack.

Linux current-best remains in qwen38_current_best_runtime.py and is untouched.
This helper uses the already cross-platform exact Q6 persistent-pool source,
which preserves the same two-thread static disjoint output-row scheduling and
Q8_0 noalloc bridge on Win32.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fast_quantize_runtime import FastActivationQuantizer, find_base_quant_runtime
from q6_persistent_pool_runtime import Q6PersistentPoolRuntime
from qwen38_current_best_runtime import (
    CURRENT_BEST_MAX_PROMPT_VECTORS,
    CURRENT_BEST_Q6_MAX_ROWS,
    CURRENT_BEST_Q6_WORKERS,
)


class Qwen38CurrentBestWin32QuantStack:
    """Own the exact portable-Q6 current-best overlay on native Windows."""

    def __init__(
        self,
        engine,
        portable_pool_lib: Path,
        *,
        q6_workers: int = CURRENT_BEST_Q6_WORKERS,
        max_rows: int = CURRENT_BEST_Q6_MAX_ROWS,
        max_vec: int = CURRENT_BEST_MAX_PROMPT_VECTORS,
    ) -> None:
        if sys.platform != "win32":
            raise RuntimeError("Qwen38CurrentBestWin32QuantStack requires native Windows")
        if int(q6_workers) != CURRENT_BEST_Q6_WORKERS:
            raise ValueError(
                f"current-best Q6 worker count is pinned to {CURRENT_BEST_Q6_WORKERS}, "
                f"got {q6_workers}")

        self.engine = engine
        self.base_runtime = engine.runtime
        self.base_quant = find_base_quant_runtime(self.base_runtime)
        self.fast_quant = FastActivationQuantizer(self.base_quant)
        self.runtime: Q6PersistentPoolRuntime | None = None
        self._closed = False

        self.fast_quant.install()
        try:
            self.runtime = Q6PersistentPoolRuntime(
                self.base_runtime,
                Path(portable_pool_lib),
                threads=CURRENT_BEST_Q6_WORKERS,
                max_rows=int(max_rows),
                max_vec=int(max_vec),
            )
            self.engine.runtime = self.runtime
        except Exception:
            self.fast_quant.restore()
            raise

    def report(self) -> dict:
        if self.runtime is None:
            raise RuntimeError("Win32 current-best quant stack is not initialized")
        return {
            "platform": sys.platform,
            "q6_backend": "portable-win32-native-sync",
            "q6_workers": CURRENT_BEST_Q6_WORKERS,
            "q6_pool": self.runtime.pool_report(),
            "quant_many": self.runtime.report(),
            "fast_quantize": self.fast_quant.report(),
            "q8_noalloc": True,
            "q6_static_disjoint_rows": True,
            "q6_within_row_reduction_change": False,
            "arithmetic_change": False,
            "linux_current_best_changed": False,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.engine.runtime = self.base_runtime
        if self.runtime is not None:
            self.runtime.close()
        self.fast_quant.restore()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def sanity() -> None:
    if CURRENT_BEST_Q6_WORKERS != 2:
        raise SystemExit("current-best Q6 worker pin changed")
    if CURRENT_BEST_Q6_MAX_ROWS != 17408:
        raise SystemExit("current-best Q6 max-row contract changed")
    if CURRENT_BEST_MAX_PROMPT_VECTORS != 11:
        raise SystemExit("current-best prompt-vector contract changed")
    print("QWEN38_CURRENT_BEST_WIN32_QUANT_SANITY PASS")


if __name__ == "__main__":
    sanity()
