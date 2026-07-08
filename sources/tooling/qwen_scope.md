# Qwen-Scope — Cross-Family SAE Artifacts

## Overview

Open-source SAE suite for **Qwen3 and Qwen3.5** (dense + MoE). 14 SAE groups across 7 model variants.

Paper: https://arxiv.org/abs/2605.11887 (PDF in `sources/pdfs/2605.11887_qwen_scope.pdf`)

## RCG-Bench role

Cross-family validation in Weeks 9–10. Same protocol as Gemma Scope 2:

- top-k feature activation → reportability
- feature ablation/steering → causal effect

## Capabilities (from paper)

1. Inference-time steering via feature directions
2. Evaluation analysis (benchmark redundancy via features)
3. Data-centric workflows (toxicity, safety synthesis)
4. Post-training optimization signals

## Access

- Check Hugging Face for `Qwen-Scope` or Qwen org repos (open-sourced with paper)
- Neuronpedia may host Qwen feature dashboards

## MVP recommendation

Defer until Gemma pipeline works. Optional: Qwen3-1.7B or smallest available variant with matching SAE.

## Blockers

- Verify exact HF repo IDs and hook naming before implementing `sae_features.py`
- MoE models need expert-specific hook points
