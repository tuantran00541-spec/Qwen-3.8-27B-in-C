#!/usr/bin/env python3
"""Named current-best exact quant runtime for Qwen3.8 research.

Linux experimental current-best as of the Q6 persistent-pool validation:
  * fast activation-quantize input marshaling;
  * fast F32 output marshaling;
  * Q8_0 exact noalloc many-vector bridge;
  * Q6_K exact persistent static row pool with two workers.

The pool changes only independent output-row scheduling.  Q6 arithmetic,
within-row reduction order, Q8 arithmetic, model bytes, and K3 reader policy are
unchanged.  This helper centralizes the proven quant runtime composition so
future probes do not accidentally assemble different "current-best" stacks.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fast_quantize_runtime import FastActivationQuantizer, find_base_quant_runtime
from q6_persistent_pool_runtime import Q6PersistentPoolRuntime

CURRENT_BEST_Q6_WORKERS = 2
CURRENT_BEST_Q6_MAX_ROWS = 17408
CURRENT_BEST_MAX_PROMPT_VECTORS = 11


class Qwen38CurrentBestQuantStack:
    """Own the experimental current-best quant overlay for one engine."""

    def __init__(
        self,
        engine,
        pool_lib: Path,
        *,
        q6_workers: int = CURRENT_BEST_Q6_WORKERS,
        max_rows: int = CURRENT_BEST_Q6_MAX_ROWS,
        max_vec: int = CURRENT_BEST_MAX_PROMPT_VECTORS,
    ):
        if not sys.platform.startswith("linux"):
            raise RuntimeError(
                "current Q6 persistent-pool backend is Linux/pthread experimental; "
                "native Windows backend is not promoted yet")
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
                pool_lib,
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
            raise RuntimeError("current-best quant stack is not initialized")
        return {
            "platform": sys.platform,
            "q6_workers": CURRENT_BEST_Q6_WORKERS,
            "q6_pool": self.runtime.pool_report(),
            "quant_many": self.runtime.report(),
            "fast_quantize": self.fast_quant.report(),
            "q8_noalloc": True,
            "q6_static_disjoint_rows": True,
            "q6_within_row_reduction_change": False,
            "arithmetic_change": False,
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
    print("QWEN38_CURRENT_BEST_RUNTIME_SANITY PASS")


if __name__ == "__main__":
    sanity()
