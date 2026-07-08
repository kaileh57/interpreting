# Confidence and Calibration of Activation Oracles for Reliable Interpretation of Language Model Internals

## Metadata

| Field | Value |
|-------|-------|
| **Authors** | Federico Torrielli, Peter Schneider-Kamp, Lukas Galke Poech |
| **Year** | 2026 |
| **URL** | https://arxiv.org/abs/2605.26045 |
| **Code** | https://github.com/federicotorrielli/probabilistic_activation_oracles |

## Summary

Activation oracles emit natural-language claims without native uncertainty. This paper benchmarks six confidence methods on the Karvonen **taboo secret-word** task (20 LoRA adapters, 100 context prompts, 3 verbalizer prompts, 6000 samples per configuration). **Bootstrap mode frequency** at temperature \(T \approx 1.0\)–\(1.5\) yields best calibration (ECE 5.7% on Qwen3-8B AO vs 25.5% for answer-word log-probability; 10.3% vs 13.1% on newly trained Qwen3.6-27B AO). Free-form numeric self-report is anti-calibrated on the larger oracle. The work releases a Qwen3.6-27B AO and patched trainer.

## Key Methods

- **M1 Log-probability**: joint token probability of extracted answer word (fast triage baseline).
- **M2 Temperature bootstrap**: \(k=20\) samples; confidence = mode frequency; sweep \(T \in \{0.3,...,1.5\}\).
- **M3 Direct numeric self-report**: second-turn "confidence 0–100" under same steering hook.
- **M4–M5 MCMC power-sampling**: acceptance rate and cross-chain agreement (poor AUROC on steered oracles).
- **M6 Steering-coefficient sensitivity**: modal answer stability across \(c \in \{0.5,...,1.5\}\).
- Evaluated on injected residual vectors at designated layer/positions; closed taboo vocabulary for scoring.

## Key Metrics

- **Expected Calibration Error (ECE)**, Brier score, NLL, AUROC vs correctness.
- Best ECE: bootstrap \(T=1.0\) (8B, accuracy ~40%); \(T=1.5\) (27B, accuracy ~20%).
- Self-report on 27B: mean confidence 98.9% wrong vs 97.6% right (anti-calibrated).
- MCMC acceptance AUROC 0.53–0.60 (steered posterior too peaked).
- Practical rule: pick bootstrap \(T\) so mean mode frequency matches held-out empirical accuracy.

## Limitations

- Evaluated primarily on **secret-word taboo**—narrow answer format; may not transfer to open-ended RCG tasks.
- Steered-oracle setting breaks standard LLM UQ assumptions.
- Only one training-time read layer carries secret signal at inference on 27B (layer dependence).
- No joint evaluation of calibration and **causal effect size** of AO-identified concepts.
- MCMC and steering-sensitivity methods largely fail for steered oracles.

## Relevance to RCG-Bench

Supports RCG-Bench **causal calibration** secondary metric: when AO reports high confidence (bootstrap mode frequency), is **intervention effect** also larger? Torrielli et al. show verbal confidence can be miscalibrated even when answers are right—parallel to RCG's reportability–causality gap. A well-calibrated AO can be confidently **readable but non-causal**.

## Actionable Implementation Notes

- Default confidence: **bootstrap mode frequency** with \(k=20\), tune \(T\) on held-out slice to match accuracy.
- Use log-prob of answer word as cheap pre-filter before expensive multi-sample bootstrap.
- For RCG-Bench, log triples: (reportability correct?, bootstrap confidence, normalized causal effect); test correlation—weak correlation supports gap thesis.
- Avoid free-form numeric self-report for AO confidence in benchmark pipeline.
- Release includes Qwen3.6-27B AO trainer patch—use same injection hook for confidence + intervention code paths.
- Stratify by AO correctness: calibration among "readable and causal" vs "readable but non-causal" quadrants.
