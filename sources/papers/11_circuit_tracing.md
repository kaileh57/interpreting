# Circuit Tracing: Revealing Computational Graphs in Language Models (Attribution Graphs)

**URL:** https://transformer-circuits.pub/2025/attribution-graphs/methods.html

**Companion:** https://transformer-circuits.pub/2025/attribution-graphs/biology.html (On the Biology of a Large Language Model — Claude 3.5 Haiku case studies)

**Authors:** Jack Lindsey, Joshua Batson, Carson Denison, et al. (Anthropic)

**Published:** March 27, 2025

## Summary

Anthropic's **circuit tracing** method builds **attribution graphs**: prompt-specific computational graphs tracing how sparse features cause a target output token. Key ingredients: (1) **cross-layer transcoders (CLTs)** replacing MLPs in a linearized replacement model; (2) backward attribution through frozen attention/norms; (3) graph pruning to salient nodes/edges; (4) interactive exploration UI; (5) **perturbation validation** (feature-direction interventions vs. graph predictions). Applied at scale in the companion "Biology" paper on Claude 3.5 Haiku (multihop reasoning, planning, hallucination, refusal, etc.). CLTs match underlying model outputs ~50% of cases; graphs are information-dense even after pruning.

## How it differs from RCG-Bench

| Dimension | Circuit Tracing / Attribution Graphs | RCG-Bench |
|-----------|-----------------------------------|-----------|
| **Output** | Causal graph of feature→feature→logit paths | Scalar reportability–causality gap per readout |
| **Features** | CLT features (not NL readouts) | J-lens, AO, NLA, SAE descriptions, self-report |
| **Validation** | Perturbation consistency with graph | Intervention on readout-named variable |
| **Scope** | Per-prompt mechanism discovery | Cross-method benchmark on latent tasks |
| **Models** | Claude 3.5 Haiku (+ small open model demos) | Open-weight focus |

Attribution graphs provide **structural causal readouts**; RCG-Bench tests whether **natural-language readouts** identify the same variables graphs/interventions implicate.

## What to borrow

### Metrics
- **Perturbation-validation protocol**: intervene on feature directions, compare to graph-predicted effects — gold standard for causal readout validation.
- **Graph pruning / salience**: identify top-k causal nodes for precision@k comparison with readout top-k.
- **Replacement-model faithfulness metrics** (CLT reconstruction, completeness).

### Baselines
- **Graph-based causal readout**: top attributed features/nodes as non-NL baseline with high structural causal signal.
- **Transcoder/graph descriptors** listed in plan.md RQ2 comparison arm.
- Sparse feature circuits (Marks et al.) as per-layer alternative; CLTs as cross-layer alternative.

### Pitfalls
- **Replacement model ≠ underlying model** — graphs may artifact (attention frozen, reconstruction error, CLT depth collapse).
- Graphs are per-prompt, labor-intensive to interpret — not a drop-in benchmark.
- CLT training is expensive; open-model replication may use per-layer transcoders or SAEs instead.
- Graph features have auto-descriptions but not integrated NL readout evaluation.
- May 2026 downstream-connections update shows local graph nodes can mislead without downstream context.
- Faithfulness critiques (e.g., CLT unfaithful circuits literature) — cite limitations when positioning RCG.
