# FontCluster models

This repository is the release source for optional embedding models used by
[FontCluster](https://github.com/MugiSus/fontcluster).

## Distribution contract

Each model is published as one GitHub Release whose tag exactly matches the ID
stored in `model.json`. That ID is also the directory name used by FontCluster
under Application Support. Each release contains exactly these three assets:

- `model.json`
- `model.onnx`
- `attribute_directions.json`

FontCluster accepts model API version 1 with this fixed inference contract:

| Field | Value |
| --- | --- |
| Input | `gray_image`, float32, `[8, 1, 224, 224]` |
| Output | `embedding`, float32, `[8, 512]` |
| Preprocessing | `fontcluster_grayscale_mask_v1` |
| Output normalization | L2-normalized |

Release assets are installed together as one bundle. Attribute directions are
model-specific and must never be reused across different model IDs.

`model.json` is FontCluster's bundle manifest, not an external model format.
It contains the model API version, identity and display metadata, an optional
parameter count, and the checksums used to validate an installed bundle. A
separate schema version is intentionally omitted: additive metadata remains
backward-compatible, while an incompatible inference contract requires a new
model API version.

`parameterCount` is the total number of scalar elements in the source inference
module's parameter tensors immediately before ONNX export. Non-parameter
buffers, preprocessing constants, and export-generated initializers are
excluded. It may be omitted when that source-level count cannot be established
reliably.

## Model versions

A model ID identifies one exact selectable bundle. A retrained model, a
converted model whose bytes or output changes, regenerated attribute
directions, or any other bundle change must use a new model ID.

This repository uses regular mutable GitHub Releases. FontCluster accepts
published releases and ignores drafts and prereleases; it does not require the
GitHub `immutable` flag. Validate all three assets before publishing. Do not
silently replace assets after a model is available to FontCluster users;
withdraw the release and publish the changed bundle under a new model ID.

## Repository layout

`models/<model-id>/` contains the tracked manifest for each model.
`releases/<model-id>.md` contains the corresponding release-note source. Large
generated assets are not committed. Assemble the three release assets in a
separate staging directory after validating the model. GitHub records a SHA-256
digest for each uploaded release asset; FontCluster verifies those digests
while downloading.

## Available model sources

| Model ID | Display name | Notes |
| --- | --- | --- |
| `mobilenet-v4-medium-v1` | MobileNet V4 Medium | Default compact 512-dimensional model, downloaded on demand. |
| `fontclip-vit-b32-v1` | FontCLIP ViT-B/32 | 512-dimensional FontCLIP ONNX model with FP32 parameters. |

## Attribution and licensing

The model-specific `attribute_directions.json` files were fitted using the font
images and attribute ratings published with:

> Peter O'Donovan, Jānis Lībeks, Aseem Agarwala, and Aaron Hertzmann.
> *Exploratory Font Selection Using Crowdsourced Attributes.*
> ACM Transactions on Graphics 33(4), 2014 (Proc. SIGGRAPH).

The [official dataset](https://www.dgp.toronto.edu/~donovan/font/) is distributed
under Creative Commons Attribution-NonCommercial (CC BY-NC) without identifying
a license version.

The FontCLIP model builds on
[FontCLIP](https://github.com/yukistavailable/FontCLIP) and
[OpenAI CLIP](https://github.com/openai/CLIP). MobileNet V4 support used for the
compact model is available through
[timm](https://github.com/huggingface/pytorch-image-models). Their source-code
licenses do not by themselves establish a license for the model weights. Model
manifests use `NOASSERTION` when no explicit weights license has been established.
This section is the attribution record for all model releases; attribution is
not duplicated as a per-release asset.
