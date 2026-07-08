# Emergent Introspective Awareness in Large Language Models

**URL:** https://transformer-circuits.pub/2025/introspection/index.html

**Authors:** Jack Lindsey et al. (Anthropic)

**Published:** October 29, 2025

## Summary

Anthropic tests whether LLMs have **functional introspective awareness** using **concept injection**: known concept vectors are injected into residual-stream activations while the model answers introspective questions. Models (especially Claude Opus 4/4.1) sometimes detect injected concepts, distinguish injected "thoughts" from text input, use prior activations to accept/reject prefilled outputs, and modulate internal representations when instructed to "think about" a word. Introspection is evaluated against four criteria: accuracy, grounding (causal dependence on internal state), internality (not routed through prior outputs), and metacognitive representation. Success rates are low (~20% for concept detection at best); most apparent introspection is unreliable or confabulated.

## How it differs from RCG-Bench

| Dimension | Anthropic Introspection | RCG-Bench |
|-----------|------------------------|-----------|
| **Readout** | Behavioral self-report | White-box readouts (J-lens, AO, NLA, SAE) + self-report |
| **Causal test** | Concept injection → self-report change | Intervention on readout-identified representation → behavior change |
| **Direction** | Internal state → verbal report | Readable explanation → does it name causal variable? |
| **Models** | Claude family (closed) | Open-weight models (Gemma, Qwen, etc.) |
| **Ground truth** | Injected concept (experimenter-known) | Task latent variables |

Anthropic asks *can models report injected internal states?* RCG-Bench asks *when any readout says "the model represents X," does intervening on X change behavior?* Self-report is one arm of RCG, not the whole benchmark.

## What to borrow

### Metrics
- **Concept injection protocol**: gold-standard for establishing causal link between internal state and report (adapt for open models).
- **Four introspection criteria** (accuracy, grounding, internality, metacognition): structure self-report evaluation in RCG.
- **Prefill-detection experiment**: directly relevant to adversarial-prefill task in plan.md.

### Baselines
- **Behavioral self-report** as a readout family with expected low causal precision@k.
- Contrastive activation vectors for concept injection/intervention.

### Pitfalls
- **Illusory introspection**: models confabulate details even when detection is correct — mirrors RCG "readable-but-non-causal" and "readout hallucination."
- Claude-only results do not transfer to open models; RCG's cross-family validation is essential.
- Concept injection is unnatural — complement with ground-truth latent tasks.
- Detection ≠ causal control: noticing an injection differs from that concept driving output.
- Do not claim first introspection study — cite alongside "Looking Inward" and adversarial-prefill work.
