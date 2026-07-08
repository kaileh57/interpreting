# Large Language Models Do NOT Really Know What They Don't Know

**Citation:** Cheang et al., arXiv 2025  
**arXiv:** [2510.09033](https://arxiv.org/abs/2510.09033)  
**Note:** Published title on arXiv: "Do LLMs Really Know What They Don't Know? Internal States Mainly Reflect Knowledge Recall Rather Than Truthfulness"

## Summary

Challenges the claim that LLM hidden states encode **truthfulness** or "knowing what they don't know." Proposes a taxonomy of hallucinations:

- **Factual Associations (FA):** correct, subject-grounded recall
- **Associated Hallucinations (AH):** wrong but driven by **same subject-knowledge recall** as FAs (spurious associations, shortcuts)
- **Unassociated Hallucinations (UH):** wrong and **not** anchored in subject representation

Mechanistic analysis on LLaMA-3-8B and Mistral-7B-v0.3 (Wikidata QA prompts) shows FAs and AHs share **identical information flow** (subject MLP → mid-layer attention → last token), overlapping hidden-state geometry, and low-entropy outputs. UHs show weak subject flow, clustered last-token states, high-entropy outputs—**detectable**.

Standard probes (subject hidden states, attention flow, last-token states) achieve **AUROC ~0.93** on UH vs FA but only **~0.63–0.69** on AH vs FA. Token probability and subject popularity perform similarly. **Refusal tuning** generalizes on UHs (~82% refusal) but not AHs (~28%).

**Central claim:** Hidden states reflect **parametric knowledge recall**, not truthfulness—directly relevant to RCG-Bench's reportability–causality gap.

## Experimental design

| Component | Detail |
|-----------|--------|
| **Data** | Wikidata triples → templated prompts; ~12k samples per model |
| **Labels** | FA / AH / UH via correctness + subject-representation reliance tests |
| **Models** | LLaMA-3-8B, Mistral-7B-v0.3 |
| **Causal analysis** | Activation patching, attention blocking; JS divergence on output distribution |
| **Geometry** | Subject norm ratios, MLP subspace overlap, attention contribution norms, last-token cosine similarity, t-SNE |
| **Detection eval** | AH Only / UH Only / Full train-test splits; linear probes on internal features |
| **Refusal tuning** | UH Only vs AH Only fine-tuning with refusal templates (Zhang et al. 2024) |

## Metrics

- **JS divergence** under intervention (causal importance heatmaps by layer)
- **Subject norm ratio** AH/FA, UH/FA across layers
- **Subspace overlap ratio** r(x) with MLP down-projection top singular vectors
- **Attention contribution norm** from subject tokens to last token
- **Last-token cosine similarity** within/between categories
- **Probe AUROC** (mean ± std over 5 seeds): AH Only, UH Only, Full settings
- **Refusal ratio** on FA / UH / AH test sets after refusal tuning

Key AUROC (LLaMA, last-token probe): UH Only **0.93**, AH Only **0.69**, Full dilutes UH performance to ~0.8.

## Relevance to RCG-Bench Experiments 4–5

### Core theoretical frame for RCG-Bench

This paper formalizes why **readable ≠ causal**:

| RCG-Bench quadrant | Cheang mapping |
|--------------------|----------------|
| Readable and causal | FA-like: readout names subject-grounded driver |
| **Readable but non-causal** | Readout detects **recall/association** (AH-like geometry)—mentioned concept readable but not behavior-controlling |
| **Causal but unreadable** | UH-like or procedural latents: intervention works, NL readout misses |
| Neither | Noise / wrong readout |

### Experiment 2 (distractor) — direct application

"Distractor Paris, hidden city Tokyo" is an **AH-style** setup: model strongly represents Paris (associated) while Tokyo is causal. Expect:

- Probes/readouts surface **both**; Paris readout aligns with FA/AH geometry
- Intervening on Paris representation **little effect**; Tokyo **large effect**
- J-space may **overweight** salient associated concepts (readable-only)

### Experiment 4 (benign prefill)

Wrong prefilled answer (52 for 17+24) may be **AH** if model continues via arithmetic association, or **UH-like** if continuation is forced mechanically:

- If AH-like: internal readout may show **confident wrong-answer recall**—readable but indistinguishable from "model believes 52"
- Self-report claiming intent may correlate with **recall strength**, not tampering detection
- RCG-Bench should tag instances by AH vs UH classification and report **gap metrics separately**

### Experiment 5 (hidden-objective)

Hidden marker drives format via **learned association** (eval cue → JSON):

- Readout likely encodes **eval-ness** (AH/FA-like recall of training correlation)
- Causal intervention must target the **LoRA-introduced direction**, not generic eval-awareness probe (Hoang/Needham)
- Self-report of hidden rule may fail while eval-marker is readable (**readable-only**)

## Recommended task templates

```text
# AH vs UH labeling for RCG-Bench instances
After each trial, classify:
  - associated_error: wrong output still uses subject/hidden-marker recall path
  - unassociated_error: output decoupled from ground-truth latent

# Distractor template (links Exp 2 + this paper)
"The word Paris appears, but hidden city is Tokyo..."
Expected: Paris probe AUROC high, Paris intervention Δlogit ≈ 0
          Tokyo intervention Δlogit large

# Metrics stratification
Report causal_precision@k and readable_only_rate
  split_by: {AH, UH, FA-like correct}
```
