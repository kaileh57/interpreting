# SAEs Are Good for Steering—If You Select the Right Features

**URL:** https://arxiv.org/abs/2505.20063

**Authors:** Dana Arad, Aaron Mueller, Yonatan Belinkov

## Summary

This paper formalizes two SAE feature roles: **input features** (high overlap between top-activating tokens and logit-lens tokens) and **output features** (intervening on the feature increases probability of its logit-lens tokens). Input and output scores rarely co-occur; early layers are input-aligned, later layers output-aligned. Filtering by **output score** yields 2–3× steering improvement on AxBench, making unsupervised SAE steering competitive with supervised methods. The work directly responds to AxBench's negative SAE results and shows the gap is largely **feature-selection error**, not inherent SAE weakness.

## How it differs from RCG-Bench

| Dimension | Arad et al. 2025 | RCG-Bench |
|-----------|------------------|-----------|
| **Selection criterion** | Output score (logit-lens intervention) | Reportability of NL readout |
| **Evaluation** | Steering success on AxBench Concept500 | Reportability vs. intervention on ground-truth latents |
| **Readable explanation** | Logit-lens tokens, not NL descriptions | J-lens, AO, NLA, SAE dashboards, self-report |
| **Gap** | Input score high, output score low | Readout correct, causal effect low |
| **Scope** | SAE features only | Multi-family readout benchmark |

Both papers share the insight that **what activates ≠ what causally controls output**, but Arad et al. operationalize it via logit-lens output scores; RCG-Bench operationalizes it via readable explanations vs. interventions.

## What to borrow

### Metrics
- **Input score** and **output score**: directly analogous to reportability vs. causality for SAE arm.
- **Gen Success@k / perplexity tradeoff**: steering evaluation template.
- **Harmonic mean on AxBench** (concept, fluency, instruction): reuse for SAE intervention arm.

### Baselines
- Output-score filtering as the **best unsupervised SAE steering baseline**.
- Layer-wise input/output role taxonomy — expect readable SAE descriptions to track input features more than output features.

### Pitfalls
- Output score uses logit-lens tokens, not natural-language feature descriptions — **high-description-accuracy features may still be input features**.
- Single-feature steering only; multi-feature interactions ignored.
- Gemma-2 / Gemma-Scope only — cross-model validation needed for RCG.
- Do not claim "we discovered input ≠ output features" — cite this paper and GradSAE.
- External LLM rater instability affects AxBench replication.
