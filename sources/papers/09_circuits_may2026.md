# Circuits Updates — May 2026: Downstream Connections Predict Which Features Will Steer

**URL:** https://transformer-circuits.pub/2026/may-update/index.html

**Authors:** Purvi Goel, Isaac Kauvar, Nicholas L. Turner; edited by Harish Kamath (Anthropic)

## Summary

Preliminary Anthropic lab-note reporting that **standard SAE/CLT feature descriptors** (top activating examples, top unembeds) fail to distinguish features with similar local appearance but different causal steering effects. Example: two "green" features activate on the same prompt; only one steers the answer because the other routes through hex-color downstream circuits. **TWERA-ranked downstream features** (virtual weights between features, weighted by coactivation) improve an LLM's ability to predict which candidate feature will steer (normalized rank 0.457 → 0.355). Applied to alignment auditing (desperation features in blackmail scenarios). Treat as preliminary, not a mature paper.

## How it differs from RCG-Bench

| Dimension | May 2026 Update | RCG-Bench |
|-----------|-----------------|-----------|
| **Problem** | Which feature in a similar set is causal? | Does a readable readout name the causal variable? |
| **Descriptor** | Top examples + unembeds vs. downstream | NL explanations across readout families |
| **Prediction** | LLM ranks steering candidates | Reportability vs. intervention gap |
| **Scope** | CLT/dictionary features | J-lens, AO, NLA, SAE, probes, self-report |
| **Validation** | Multiplicative steering on Claude | Open models, ground-truth latents |

Both share the theme **local readability ≠ causal responsibility**, but this update adds **downstream circuitry** as a selection signal; RCG-Bench systematizes the gap as a benchmark metric.

## What to borrow

### Metrics
- **Normalized rank of correct steering feature** given different descriptor subsets — template for causal precision@k ablations.
- **TWERA downstream ranking** as an optional SAE/CLT feature-selection enhancement.

### Baselines
- Compare readout-only selection vs. readout + downstream-graph selection on RCG tasks.
- "Similar-looking features, different causal effects" as a **core failure mode** for readable-only rate.

### Pitfalls
- Preliminary results on small candidate sets (10 groups × 3–5 features).
- Claude/CLT-specific; may not apply to per-layer SAEs on open models.
- Downstream ranking helps prediction but does not solve the problem — best rank still >0.
- Do not treat TWERA as a readable explanation — it is a structural causal cue.
- plan.md item 11 (Circuit Tracing) points to the same URL family; this post is specifically the downstream-connections finding.
