#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "qwen38" / "bonsai2_prompt_spike.py"


def function_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function: {name}")


def parser_add_arguments(main_node: ast.FunctionDef) -> set[str]:
    flags: set[str] = set()
    for node in ast.walk(main_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        if not node.args:
            continue
        arg0 = node.args[0]
        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
            flags.add(arg0.value)
    return flags


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    run_node = function_node(tree, "run")
    run_args = [arg.arg for arg in run_node.args.args]
    for required in ("expected_text", "stream_text", "json_events"):
        if required not in run_args:
            raise AssertionError(
                f"run() missing generic generation argument: {required}"
            )

    for node in ast.walk(run_node):
        if isinstance(node, ast.Constant) and node.value == "Hello from Bonsai 2!":
            raise AssertionError("run() still hard-codes Hello expected text")

    main_node = function_node(tree, "main")
    flags = parser_add_arguments(main_node)
    for required in ("--expected-text", "--stream-text", "--no-json-events"):
        if required not in flags:
            raise AssertionError(f"CLI missing {required}")

    if '"schema": "qwen38-bonsai2-generation-v2"' not in source:
        raise AssertionError("generic generation schema v2 missing")

    if '"tokens_per_second"' not in source:
        raise AssertionError("generation throughput metric missing")

    if '"time_to_first_token_seconds"' not in source:
        raise AssertionError("time-to-first-token metric missing")

    if '"completion_truncated"' not in source:
        raise AssertionError("max-token truncation signal missing")

    print("QWEN38_BONSAI2_GENERATION_RUNNER_CONTRACT_PASS")


if __name__ == "__main__":
    main()
