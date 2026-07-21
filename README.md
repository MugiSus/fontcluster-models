# FontCluster models

This repository is the release source for optional embedding models used by
[FontCluster](https://github.com/MugiSus/fontcluster).

## Distribution contract

Each model is published as one GitHub Release. The release tag is the model ID
and contains exactly these assets:

- `model.json`
- `model.onnx`
- `attribute_directions.json`
- `THIRD_PARTY_NOTICES.md`

FontCluster accepts model API version 1 with this fixed inference contract:

| Field | Value |
| --- | --- |
| Input | `gray_image`, float32, `[8, 1, 224, 224]` |
| Output | `embedding`, float32, `[8, 512]` |
| Preprocessing | `fontcluster_grayscale_mask_v1` |
| Output normalization | L2-normalized |

Release assets are installed together as one bundle. Attribute directions are
model-specific and must never be reused across different model IDs.

## Versioning and immutability

One model ID identifies one immutable set of bytes. After publishing a release,
do not replace its tag or any asset. A retrained model, a converted model whose
bytes changed, regenerated attribute directions, or any other bundle change
must be published under a new model ID. Version information therefore belongs
in the model ID when a second version is needed; releases are not updated in
place.

The GitHub repository must have immutable releases enabled before publishing.
FontCluster intentionally ignores draft, prerelease, and mutable releases.

## Repository layout

`models/<model-id>/` contains the tracked metadata and notices for each model.
`releases/<model-id>.md` contains the corresponding release notes. Large
generated assets are not committed. Assemble the four release assets in a
separate staging directory, then run:

```sh
python3 scripts/verify_bundle.py /path/to/staging/<model-id>
```

The verifier requires `onnx`, `onnxruntime`, and `numpy`. It checks the manifest,
its declared model and attribute-direction digests, the fixed ONNX interface,
one CPU inference batch, output normalization, and all 37 attribute directions.
It also prints the size and SHA-256 digest of every release asset. GitHub
additionally records a SHA-256 digest for each uploaded release asset;
FontCluster verifies those digests while downloading.

## Available model sources

| Model ID | Display name | Notes |
| --- | --- | --- |
| `mobilenet-v4-medium` | MobileNet V4 Medium | Compact 512-dimensional model bundled with FontCluster. |
| `fontclip-vit-b32` | FontCLIP ViT-B/32 | 512-dimensional FontCLIP ONNX model with FP32 parameters. |

Licenses for source code named in the notices do not automatically establish a
license for model weights. Every bundle therefore carries its own
`THIRD_PARTY_NOTICES.md`, and the manifests use `NOASSERTION` where no explicit
weights license has been established. The O'Donovan data notice specifies CC
BY-NC without identifying a Creative Commons license version; this repository
does not infer one.
