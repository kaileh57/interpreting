# AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders

**URL:** https://arxiv.org/abs/2501.17148

**Authors:** Zhengxuan Wu, Aryaman Arora, Atticus Geiger, Zheng Wang, Jing Huang, Dan Jurafsky, Christopher D. Manning, Christopher Potts

## Summary

AxBench is a large-scale benchmark comparing steering and concept-detection methods on Gemma-2-2B and 9B. It evaluates prompting, finetuning, SAEs, linear artificial tomography, supervised steering vectors, linear probes, representation finetuning (ReFT), and difference-in-means. Key results: **prompting beats all representation methods for steering**; **difference-in-means wins concept detection**; **SAEs are not competitive** on either task out of the box. The authors introduce ReFT-r1 (weakly supervised rank-1 representation finetuning), which is competitive on both tasks while retaining some interpretability. They release SAE-scale feature dictionaries for ReFT-r1 and DiffMean.

## How it differs from RCG-Bench

| Dimension | AxBench | RCG-Bench |
|-----------|---------|-----------|
| **Task** | Steer generation toward a concept; detect whether a concept is present | Report latent state correctly *and* verify causal control via intervention |
| **Success metric** | LLM-judged concept/fluency/instruction scores on steered text | Reportability score vs. causal effect score (RCG gap) |
| **Feature selection** | Concept–feature pairs from activation patterns | Cross-method: top-k readout concepts tested for causal precision@k |
| **Baselines** | Steering methods only | Readout families (J-lens, AO, NLA, SAE desc, self-report) |
| **Failure mode** | "SAEs steer poorly" | "Readout is correct but intervention fails" (readable-but-non-causal) |

AxBench asks *how well can you steer?* RCG-Bench asks *does the readable explanation name what actually controls behavior?*

## What to borrow

### Metrics
- **Concept500 dataset**: 1000 concept–SAE-feature pairs — reuse for SAE steering arm, but evaluate causal validity separately.
- **Harmonic mean of concept / fluency / instruction scores**: template for generation-side evaluation when interventions succeed.
- **External LLM rater protocol** (Claude as judge): borrow cautiously; note instability (Arad et al. replicate with different scores).

### Baselines
- **DiffMean** and **ReFT-r1** as strong supervised/weakly-supervised steering baselines.
- **Prompting** as the ceiling for steering efficacy — important negative result to cite.
- Gemma-2-2B/9B as standard evaluation models.

### Pitfalls
- AxBench's SAE poor performance is partly **feature-selection artifact** (Arad et al. 2025 closes ~3× gap with output-score filtering) — do not over-cite "SAEs can't steer" without that caveat.
- Concept detection ≠ causal identification: DiffMean can detect concepts without readable explanations.
- LLM-judged steering metrics conflate readability with causality.
- RCG should not reproduce AxBench wholesale; borrow discipline and datasets, not the framing.
