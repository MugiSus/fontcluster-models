#!/usr/bin/env python3
"""Validate one FontCluster model release bundle and print its asset digests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


ASSET_NAMES = (
    "model.json",
    "model.onnx",
    "attribute_directions.json",
    "THIRD_PARTY_NOTICES.md",
)
ATTRIBUTE_NAMES = {
    "angular",
    "artistic",
    "attention-grabbing",
    "attractive",
    "bad",
    "boring",
    "calm",
    "capitals",
    "charming",
    "clumsy",
    "complex",
    "cursive",
    "delicate",
    "disorderly",
    "display",
    "dramatic",
    "formal",
    "fresh",
    "friendly",
    "gentle",
    "graceful",
    "happy",
    "italic",
    "legible",
    "modern",
    "monospace",
    "playful",
    "pretentious",
    "serif",
    "sharp",
    "sloppy",
    "soft",
    "strong",
    "technical",
    "thin",
    "warm",
    "wide",
}
EXPECTED_INFERENCE = {
    "inputName": "gray_image",
    "inputDtype": "float32",
    "inputShape": [8, 1, 224, 224],
    "outputName": "embedding",
    "outputDtype": "float32",
    "outputShape": [8, 512],
    "preprocessing": "fontcluster_grayscale_mask_v1",
    "l2Normalized": True,
}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("bundle", type=Path, help="directory containing the four release assets")
args = parser.parse_args()
bundle = args.bundle.resolve()

missing = [name for name in ASSET_NAMES if not (bundle / name).is_file()]
extras = sorted(path.name for path in bundle.iterdir() if path.is_file() and path.name not in ASSET_NAMES)
if missing or extras:
    raise SystemExit(f"invalid asset set: missing={missing}, extra={extras}")

with (bundle / "model.json").open(encoding="utf-8") as source:
    manifest = json.load(source)
if manifest.get("schemaVersion") != 1 or manifest.get("modelApiVersion") != 1:
    raise SystemExit("model.json must use schemaVersion=1 and modelApiVersion=1")
if manifest.get("id") != bundle.name:
    raise SystemExit(f"manifest id {manifest.get('id')!r} does not match directory {bundle.name!r}")
if manifest.get("inference") != EXPECTED_INFERENCE:
    raise SystemExit("model.json does not declare the fixed model API v1 inference contract")
if manifest.get("licenses") != {
    "modelWeights": "NOASSERTION",
    "attributeDirections": "CC-BY-NC (version unspecified)",
}:
    raise SystemExit("model.json does not declare the required bundle license metadata")

digests = {}
for name in ASSET_NAMES:
    digest = hashlib.sha256()
    with (bundle / name).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    digests[name] = digest.hexdigest()
provenance = manifest.get("provenance", {})
if provenance.get("modelSha256") != digests["model.onnx"]:
    raise SystemExit("modelSha256 in model.json does not match model.onnx")
if provenance.get("attributeDirectionsSha256") != digests["attribute_directions.json"]:
    raise SystemExit("attributeDirectionsSha256 in model.json does not match attribute_directions.json")

model_path = bundle / "model.onnx"
model = onnx.load(model_path, load_external_data=True)
onnx.checker.check_model(model, full_check=True)
graph_inputs = {value.name: value for value in model.graph.input}
graph_outputs = {value.name: value for value in model.graph.output}
if "gray_image" not in graph_inputs or "embedding" not in graph_outputs:
    raise SystemExit("ONNX graph is missing gray_image input or embedding output")

input_type = graph_inputs["gray_image"].type.tensor_type
output_type = graph_outputs["embedding"].type.tensor_type
input_shape = [dimension.dim_value for dimension in input_type.shape.dim]
output_shape = [dimension.dim_value for dimension in output_type.shape.dim]
if input_type.elem_type != onnx.TensorProto.FLOAT or input_shape != [8, 1, 224, 224]:
    raise SystemExit(f"unexpected ONNX input contract: dtype={input_type.elem_type}, shape={input_shape}")
if output_type.elem_type != onnx.TensorProto.FLOAT or output_shape != [8, 512]:
    raise SystemExit(f"unexpected ONNX output contract: dtype={output_type.elem_type}, shape={output_shape}")

session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
sample = np.random.default_rng(0).random((8, 1, 224, 224), dtype=np.float32)
embedding = session.run(["embedding"], {"gray_image": sample})[0]
if embedding.shape != (8, 512) or embedding.dtype != np.float32 or not np.isfinite(embedding).all():
    raise SystemExit(f"invalid inference output: shape={embedding.shape}, dtype={embedding.dtype}")
norms = np.linalg.norm(embedding, axis=1)
# FP16 parameter storage can introduce sub-millipercent rounding in the final norm.
if not np.allclose(norms, 1.0, rtol=1e-3, atol=1e-3):
    raise SystemExit(f"embedding output is not L2-normalized: norms={norms.tolist()}")

with (bundle / "attribute_directions.json").open(encoding="utf-8") as source:
    directions = json.load(source)
if directions.get("dim") != 512 or set(directions.get("attributes", {})) != ATTRIBUTE_NAMES:
    raise SystemExit("attribute_directions.json must contain the exact 37 attributes at dim=512")
for name, value in directions["attributes"].items():
    vector = value.get("direction")
    if not isinstance(vector, list) or len(vector) != 512:
        raise SystemExit(f"attribute {name!r} does not have a 512-dimensional direction")
    if not all(isinstance(component, (int, float)) and math.isfinite(component) for component in vector):
        raise SystemExit(f"attribute {name!r} contains a non-finite or non-numeric component")
    norm = float(np.linalg.norm(np.asarray(vector, dtype=np.float32)))
    if not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        raise SystemExit(f"attribute {name!r} direction is not unit-normalized: norm={norm}")

notices = (bundle / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
if not notices.strip():
    raise SystemExit("THIRD_PARTY_NOTICES.md is empty")

print(f"verified {manifest['id']} (ONNX opset {max(opset.version for opset in model.opset_import)})")
for name in ASSET_NAMES:
    path = bundle / name
    print(f"{digests[name]}  {path.stat().st_size:>10}  {name}")
