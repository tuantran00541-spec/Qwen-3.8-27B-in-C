#!/usr/bin/env python3
"""Compact-prompt entrypoint for the staged Qwen3.8 four-math benchmark."""
import qwen38_mtp1_math4_benchmark as bench

bench.MATH4_PROMPT = """4 final answers only.
1 sqrt(x+6)+sqrt(x-3)=5,x>=3
2 5R4B3G urn,3 draws no replacement:P(exactly 2 colors)
3 7^2026 mod1000
4 positive a<=b,1/a+1/b=1/6:all pairs"""

if __name__ == "__main__":
    raise SystemExit(bench.main())
