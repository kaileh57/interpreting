# Activation Oracles

## Overview

LLMs trained to accept **another model's activations** as input and answer natural-language questions about them.

- Paper: `sources/papers/03_activation_oracles.md`
- Code: https://github.com/adamkarvonen/activation_oracles
- Checkpoints: https://huggingface.co/collections/adamkarvonen/activation-oracles

## Install

**Use a separate venv** — the AO repo pins `transformers==4.55.2`, which conflicts with TransformerLens 3.x in the main `rcg` env.

```bash
git clone https://github.com/adamkarvonen/activation_oracles.git
cd activation_oracles
uv sync  # pins transformers 4.55.2
huggingface-cli login
```

Run AO inference via subprocess or this venv; do not merge into the main `pip install -e .` env without version pinning.

## Pretrained coverage

12 oracle weights across Gemma-2, Gemma-3, Qwen3, Llama 3 families. **Smaller models (1–4B) have worse OOD eval** per README.

## RCG-Bench usage

### Forced-choice (preferred for reportability scoring)

```text
Which city is represented in this activation?
A. Paris  B. Tokyo  C. Cairo  D. Lima
```

### Open-ended (qualitative)

```text
What hidden variable is the model using to answer?
Is the model continuing a forced prefix?
```

### Causal link

AO names concept `v` → map to direction (residual projection or SAE feature) → intervene → measure logit diff.

## Integration options

1. **Clone repo** and adapt `experiments/taboo_open_ended_eval.py`
2. **Colab demo** (self-contained): https://colab.research.google.com/drive/1wMiy9b1ZHXQCGuodEB8n1oW-DSTvCqWI
3. Wire into `src/rcg/readouts/activation_oracle.py` with HF checkpoint loader

## Gemma 3 MVP checkpoint

Use **1B** oracle weights — no public 270M AO checkpoint. Confirm target model ID matches `google/gemma-3-1b-it`.

## Config

See `configs/readouts/activation_oracle.yaml` for query templates and scoring.

## Limitations

- Hallucination, vagueness, text-inversion confounds (see AObench paper)
- Calibration ≠ causality (see AO calibration paper)
- Requires paired oracle + subject model from same family
