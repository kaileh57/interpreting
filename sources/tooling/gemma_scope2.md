# Gemma Scope 2 — SAEs and Transcoders

## Overview

Open interpretability suite for **Gemma 3**: Sparse Autoencoders (SAEs) and Transcoders trained on **every layer** of Gemma 3 (270M, 1B, 4B, 12B, 27B; PT and IT variants).

## Access

- Docs: https://ai.google.dev/gemma/docs/gemma_scope
- Hugging Face: search `google/gemma-scope-2` collection
- Colab tutorial linked from docs (visualize features, steering demos)
- Neuronpedia: feature descriptions and dashboards

## RCG-Bench usage

1. **Feature lookup:** top-k activated SAE features at layer ℓ for a prompt.
2. **Reportability:** match Neuronpedia / feature description to ground-truth latent label.
3. **Causal test:** ablate or steer feature; measure logit diff on target token.

## Loading pattern (typical)

Artifacts are hosted on Hugging Face; exact repo names vary by size/layer. Check the Gemma Scope 2 HF collection for:

- SAE weights per layer
- Transcoder weights (optional graph baseline)
- Config JSON (dict_size, hook point)

```python
# Pseudocode — verify repo IDs from HF before running
from huggingface_hub import hf_hub_download
import torch

path = hf_hub_download(repo_id="<gemma-scope-2-sae-repo>", filename="sae_weights.safetensors")
# Load into sae-lens or custom decoder matching hook dimension
```

Consider **SAE Lens** (`pip install sae-lens`) if Gemma Scope 2 releases compatible configs.

## Limitations

- Feature descriptions are human-written; may not align with causal role.
- Input-side activation ≠ output effect (see GradSAE / steering-selection papers).
- Full layer sweep is expensive; subsample layers for MVP.

## References

- Gemma Scope 2 blog (linked from docs)
- Original Gemma Scope (Gemma 2) remains separate
