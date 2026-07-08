# TransformerLens

## Overview

Library for mechanistic interpretability: activation caching, hooks, patching, logit lens.

- GitHub: https://github.com/TransformerLensOrg/TransformerLens
- Docs: https://transformerlensorg.github.io/TransformerLens/

## Install

```bash
pip install transformer-lens>=2.0
# Gemma 3 support: verify latest release notes
```

## Core APIs for RCG-Bench

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("google/gemma-2-2b", device="cuda")
tokens = model.to_tokens("Hello world")
logits, cache = model.run_with_cache(tokens)

# Residual at layer L, last position
resid = cache["blocks.10.hook_resid_post"]

# Activation patching
def patch_hook(value, hook):
    value[0, -1, :] = clean_activation
    return value

model.run_with_hooks(tokens, fwd_hooks=[("blocks.10.hook_resid_post", patch_hook)])
```

## Gemma 3 status

TransformerLens adds models over time. **Verify** `HookedTransformer.from_pretrained` support for `gemma-3-*` before committing. Fallback: raw `transformers` + manual hooks in `src/rcg/models/loader.py`.

## RCG-Bench modules

| Module | TransformerLens use |
|--------|---------------------|
| `models/loader.py` | Load model, tokenize, device |
| `interventions/residual_patch.py` | `run_with_hooks`, clean/corrupt cache |
| `readouts/jlens.py` | Gradients w.r.t. `hook_resid_post` |
| `metrics/causality.py` | Logit diff via `logits[0, -1, token_id]` |

## Limitations

- Not all HF architectures supported day-one
- Gradient computation for J-lens may need custom backward hooks
- Memory: cache full activations only for short prompts in MVP

## Recommended version

`transformer-lens>=2.0` (already in `pyproject.toml`)
