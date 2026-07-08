# Gemma 3 — Model Loading

## Hugging Face IDs (text-only MVP)

| Size | Pretrained | Instruction-tuned | Context |
|------|------------|-------------------|---------|
| 270M | `google/gemma-3-270m-pt` | `google/gemma-3-270m-it` | 32K |
| 1B | `google/gemma-3-1b-pt` | `google/gemma-3-1b-it` | 32K |
| 4B | `google/gemma-3-4b-pt` | `google/gemma-3-4b-it` | 128K |

**RCG-Bench MVP:** start with `google/gemma-3-270m-it` and `google/gemma-3-1b-it`.

## Install

```bash
pip install torch transformers accelerate
huggingface-cli login  # accept Gemma license on HF first; set HF_TOKEN
```

## Load (transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
```

## Notes for RCG-Bench

- Use **instruction-tuned** variants for latent-slot and self-report tasks.
- 270M/1B fit T4 Colab; 4B needs A100 for intervention sweeps.
- Multimodal weights exist; RCG-Bench MVP is text-only.
- Knowledge cutoff: August 2024 (model card).

## References

- Model card: https://ai.google.dev/gemma/docs/core/model_card_3
- Technical report: https://arxiv.org/abs/2503.19786
