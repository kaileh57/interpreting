# How to Use and Interpret Activation Patching

## Metadata

| Field | Value |
|-------|-------|
| **Authors** | Stefan Heimersheim, Neel Nanda |
| **Year** | 2024 |
| **URL** | https://arxiv.org/abs/2404.15255 |

## Summary

A practitioner guide to **activation patching** (also: interchange intervention, causal tracing, resample ablation): overwrite internal activations from a **source** run into a **destination** run and measure output change. The paper distinguishes **denoising** (clean→corrupt: sufficiency) vs **noising** (corrupt→clean: necessity), warns they are not symmetric (AND vs OR circuit structures), recommends counterfactual-prompt patching over zero/mean ablation, and catalogs metric pitfalls. Essential reading before interpreting RCG-Bench intervention results.

## Key Methods

- **Standard loop**: run source prompt → cache activations → run destination prompt with selected activations patched → compare outputs.
- **Denoising (clean→corrupt)**: patch clean activations into corrupted run; finds components **sufficient** to restore behavior (may miss AND-gate partners).
- **Noising (corrupt→clean)**: patch corrupt activations into clean run; finds components **necessary** to maintain behavior (may miss OR-gate redundancies).
- **Granularity ladder**: residual stream → MLP/head → neurons/SAE features → **path patching** (single downstream target).
- **Alternatives**: zero/mean ablation, Gaussian-noise corruption (Causal Tracing/ROME)—generally less precise than paired prompts.
- **Exploratory vs confirmatory**: sweep single components vs patch all non-circuit components (causal scrubbing spirit).

## Key Metrics

- **Logit difference** (recommended primary): \( \text{LD} = \logit(y_{\text{correct}}) - \logit(y_{\text{incorrect}}) \)—linear in residual stream, task-specific, continuous.
- **Logprob / probability**: intuitive but saturates; probability nonlinearly exaggerates threshold effects.
- **Accuracy / rank**: discrete; poor for exploratory patching (threshold effects hide partial credit).
- **KL divergence**: full-distribution metric; penalizes unintended logit changes (helps with negative components).
- Internal metrics: attention patterns, probe projections, SAE feature activations.

## Limitations

- Results are **prompt-distribution-specific**—no universal circuit claims.
- Corrupted prompt choice determines what properties are traced; narrow corruptions miss broader circuits.
- **Backup heads / OR-gates** hide component importance under ablation/noising.
- **Negative components** can inflate faithfulness if excluded.
- Denoising patches see clean upstream state—can restore via mediated paths without identifying all circuit nodes.
- No minimality guarantee; found sets are sufficient, not smallest.

## Relevance to RCG-Bench

Activation patching is the **causal effect measurement backbone** for RCG-Bench. Every readout method gets a paired intervention: patch/ablate/swap along the direction or subspace the readout names, using MIB-style counterfactuals where possible. Heimersheim & Nanda's denoising/noising asymmetry explains false **readable-but-non-causal** (patch along readout direction weak effect under denoising) and **causal-but-unreadable** (causal locus found by noising but missed by readout) patterns.

## Actionable Implementation Notes

- **Primary causal score**: logit difference on task-specific correct vs distractor tokens (match MIB/IOI conventions).
- Run both **noising** and **denoising** for RCG robustness checks; report if readout-aligned patches behave differently across directions.
- Design **narrow counterfactuals** that change only ground-truth latent \(z\), controlling grammar/names/format (IOI corruption table as template).
- Prefer paired counterfactual patching over ablation when testing AO/J-lens/SAE-identified directions (stay in-distribution).
- Log dataframe: rows = (prompt, layer, position, patch target, direction), columns = LD, logprob, rank, KL.
- For confirmatory RCG tests: patch all readout-top-\(k\) concepts vs single best; use path patching if readout and causal locus diverge across layers.
- Watch **false-positive LD** from damaging corrupt answer logit uniformly—inspect individual logits, not only difference.
