#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qwen38"))

import bonsai2_prompt_spike as spike

ALIGN = 4096


def make_fixture(root: Path) -> tuple[Path, Path, dict]:
    trunk = root / "tiny.k3.bin"
    index = root / "tiny.k3.json"
    layers = []
    with trunk.open("wb") as handle:
        for layer in range(3):
            payload = bytes([layer + 1]) * ALIGN
            handle.write(payload)
            layers.append({
                "layer": layer,
                "file_offset": layer * ALIGN,
                "data_bytes": ALIGN,
                "read_bytes": ALIGN,
                "tensor_count": 0,
                "tensors": [],
            })
    manifest = {
        "schema": "qwen38-k3-trunk-v1",
        "model_id": "fixture",
        "revision": "fixture",
        "alignment": ALIGN,
        "tensor_alignment": 64,
        "layers": layers,
        "total_tensor_bytes": 3 * ALIGN,
        "packed_file_bytes": 3 * ALIGN,
    }
    index.write_text(json.dumps(manifest), encoding="utf-8")
    return trunk, index, manifest


def sweep(reader) -> None:
    for layer in range(3):
        view = reader.bind(layer)
        assert int(view[0]) == layer + 1
        view.release()


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        trunk, index, manifest = make_fixture(Path(td))

        stream = spike.make_decoder_reader(
            trunk,
            index,
            manifest,
            resident_decoder=False,
            prefer_direct_io=False,
        )
        try:
            sweep(stream)
            first_stream = stream.report()["bytes_read"]
            sweep(stream)
            second_stream = stream.report()["bytes_read"]
        finally:
            stream.close()
        assert first_stream == 3 * ALIGN, (first_stream, 3 * ALIGN)
        assert second_stream > first_stream, (first_stream, second_stream)

        resident = spike.make_decoder_reader(
            trunk,
            index,
            manifest,
            resident_decoder=True,
            prefer_direct_io=False,
        )
        try:
            sweep(resident)
            first_resident = resident.report()["bytes_read"]
            sweep(resident)
            second_resident = resident.report()["bytes_read"]
            report = resident.report()
        finally:
            resident.close()

        assert first_resident == 3 * ALIGN, (first_resident, 3 * ALIGN)
        assert second_resident == first_resident, (first_resident, second_resident)
        assert report["pinned_layers"] == [0, 1, 2], report
        assert report["ring_slots"] == 0, report
        assert report["resident_decoder"] is True, report
        assert report["decoder_resident_bytes"] == 3 * ALIGN, report

    print("QWEN38_BONSAI2_RESIDENT_K3_BEHAVIOR_PASS")


if __name__ == "__main__":
    main()
