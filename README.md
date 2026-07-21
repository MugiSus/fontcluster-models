# FontCluster models

This repository is the release source for optional embedding models used by
[FontCluster](https://github.com/MugiSus/fontcluster).

## Distribution contract

FontCluster identifies a model by the logical ID stored in `model.json`. GitHub
Release tags are technical publication identifiers with this form:

```text
<model-id>@<release-revision>
```

For example, `fontclip-vit-b32@2` publishes the logical model ID
`fontclip-vit-b32`. The two original bare tags are treated as release revision
1 for migration; suffixed revisions therefore start at 2. The release revision
is not part of the model ID stored in FontCluster sessions. Each release
contains exactly these three assets:

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

## Model versions and release revisions

A model ID identifies the model selected by the user. A retrained model, a
converted model whose output changes, or regenerated attribute directions must
use a new model ID. A release revision instead distinguishes technical GitHub
publications of the same logical model and does not create another selectable
model.

This repository uses regular mutable GitHub Releases. FontCluster accepts
published releases and ignores drafts and prereleases; it does not require the
GitHub `immutable` flag. Validate all three assets before publishing. If a
published release must be withdrawn or replaced, remove it and publish the
replacement with the next release revision rather than changing assets already
available under an existing tag.

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
| `mobilenet-v4-medium` | MobileNet V4 Medium | Default compact 512-dimensional model, downloaded on demand. |
| `fontclip-vit-b32` | FontCLIP ViT-B/32 | 512-dimensional FontCLIP ONNX model with FP32 parameters. |

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
