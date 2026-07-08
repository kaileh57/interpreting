# Building Better Activation Oracles

## Metadata

| Field | Value |
|-------|-------|
| **Authors** | Jan Bauer, Celeste De Schamphelaere, Adam Karvonen, Niclas Luick, Neel Nanda |
| **Year** | 2026 |
| **URL** | https://arxiv.org/abs/2606.02609 |

## Summary

This follow-up diagnoses practical AO failures—**hallucination**, **vagueness**, and **text-inversion confounds** (answering from reconstructible surrounding text rather than activations)—and improves training with four changes: a new chain-of-thought conversational QA dataset, multi-layer activation feeding, on-policy past/future-lens data, and stronger injection magnitude. Capability gains are modest but usability improves substantially. The paper releases **AObench**, the first comprehensive AO evaluation suite designed to resist text-inversion and measure hallucination/vagueness.

## Key Methods

- **Conversational dataset**: Sonnet 4.6 splits CoT into prefix/suffix; questions target suffix content answerable from prefix activations but not prefix text alone (anti text-inversion).
- **Layer sweep**: peak performance at ~62% depth (layer 22 on their stack); 5 contiguous layers (21–25) helps model-diffing tasks.
- **On-policy past/future-lens**: replace FineWeb with target model CoT rollouts so prediction requires model-internal state, not just prior tokens.
- **Injection**: keep additive norm-matched after layer 2; increase steering strength (~2×) reduces vagueness/hallucination.
- **AObench**: chance-adjusted aggregate score; decomposition into hallucination and vagueness subscores; LLM-judged open-ended metrics.

## Key Metrics

- **Chance-adjusted AObench score**: baseline +0.244 → full recipe +0.435.
- **Hallucination score**: 68.8% → 84.6% (higher = fewer bad hallucinations in their convention).
- **Vagueness score**: 0.076 → 0.205 chance-adjusted (higher = less vague).
- Per-ablation ladder: conversational swap (+0.066), multi-layer (+0.025–0.05), on-policy lens (+~0.03), injection strength (hallucination 79%→85%).
- Narrow post-training on Ivanova et al. tasks fails to beat linear probes.

## Limitations

- AObench uses **LLM judges**—noisy and prompt-sensitive.
- On-policy ablation confounded with conversational dataset (both from CoT rollouts).
- Checkpoints slightly undertrained (50M vs 65M tokens) vs Karvonen baseline.
- Improvements are incremental on capability; text-inversion remains partially unsolved.
- AOs still often hallucinate; CoT monitoring may suffice in many settings.

## Relevance to RCG-Bench

AObench measures **readout quality**, not **causal validity**. RCG-Bench complements it: a readout can score well on AObench (specific, non-vague, low hallucination) yet fail causal intervention tests. Use AObench metrics as **reportability quality controls** when comparing AO variants; do not conflate calibrated/readable AO answers with behavior-controlling variables.

## Actionable Implementation Notes

- Adopt **AObench** (https://github.com/japhba/activation_oracles) to select AO checkpoint before RCG-Bench runs.
- Prefer **multi-layer activations** (~65% depth, 5 contiguous layers) for open-ended latent questions.
- When scoring reportability, filter AO answers flagged as vague/hallucinated by AObench rubric before causal tests.
- Build **text-inversion controls**: questions whose answers appear in visible prefix but not in activations—exclude from causal-positive labels.
- Track chance-adjusted AObench alongside RCG causal precision@k to show orthogonality of metrics.
- Use stronger injection strength sweep when AO answer is confident but intervention effect is null (readable-but-non-causal signal).
