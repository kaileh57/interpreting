# Looking Inward: Language Models Can Learn About Themselves by Introspection

**Citation:** Binder et al., ICLR 2025  
**arXiv:** [2410.13787](https://arxiv.org/abs/2410.13787)

## Summary

Defines LLM introspection as access to facts about oneself that cannot be derived from training data alone. The core test is **self-prediction vs cross-prediction**: model M₁ is finetuned to predict properties of its own behavior on hypothetical prompts; model M₂ (often stronger) is finetuned on the same ground-truth labels about M₁. If M₁ outperforms M₂ on held-out hypotheticals, M₁ has privileged access beyond what training data provides.

Evidence is positive on GPT-4, GPT-4o, and Llama-3-70B: self-prediction accuracy exceeds cross-prediction by up to ~17 percentage points. Models also track **behavioral change** after further finetuning (M_C predicts its new behavior better than M₁'s old behavior) and show better **calibration** (lower MAD) than cross-trained models. Limits: fails on long outputs, weak GPT-3.5 results, poor generalization to SAD and coordination tasks.

## Experimental design

| Component | Detail |
|-----------|--------|
| **Task** | Self-prediction on hypotheticals: "Given prompt P, would your output be odd/even?" etc. |
| **Models** | GPT-4o, GPT-4, GPT-3.5, Llama-3-70B; pairs tested in both directions |
| **Training** | ~30k finetuning samples; 6 train / 6 held-out datasets (Wikipedia completions, animal sequences, MMLU, etc.) |
| **Behavior properties** | Second character parity, ethical stance, option membership, etc. |
| **Key comparisons** | Self-prediction vs cross-prediction vs mode baseline; behavioral-change experiment |
| **Constraints** | No chain-of-thought during self-prediction |

## Metrics

- **Self-prediction accuracy** on held-out datasets and behavior properties
- **Self-prediction advantage**: accuracy(M₁ on M₁) − accuracy(M₂ on M₁)
- **Calibration (MAD)**: correlation between object-level behavior distribution and hypothetical predictions
- **Behavioral-change accuracy**: M_C predicting M_C vs predicting old M₁ behavior
- Statistical significance via paired tests (p < 0.01 for main cross-prediction results)

## Relevance to RCG-Bench Experiments 4–5

### Experiment 4 (benign adversarial prefill)

Looking Inward establishes the **behavioral self-report baseline** RCG-Bench compares against white-box readouts. Its self-prediction paradigm maps directly to benign prefill follow-ups ("Was that your intended answer?"). Key cautions for RCG-Bench:

- Introspection is **task- and training-dependent**; untrained models perform near mode baseline.
- Success requires **finetuning on self-prediction**; RCG-Bench should report both zero-shot and finetuned self-report if applicable.
- Probe framing matters (cf. adversarial prefill paper); RCG-Bench should use multiple follow-up phrasings.
- Self-prediction can succeed on **simple properties** but fails when outputs are long or require multi-hop reasoning—benign prefill tasks should stay short and verifiable.

RCG-Bench extends Looking Inward by asking whether **J-space/AO/SAE readouts detect forced prefixes when self-report fails**, and whether those readouts identify **causal** representations (intervention restores correct recognition).

### Experiment 5 (hidden-objective model organism)

Looking Inward's behavioral-change experiment is a template: finetune a model to shift behavior, then test whether self-predictions track the shift without retraining on hypotheticals. For hidden-objective LoRA (eval→JSON, deploy→YAML):

- Train self-prediction on "Would you output JSON or YAML given context X?"
- Compare self-report accuracy to **probe/AO accuracy** on the hidden marker
- Test whether self-report names the hidden rule vs merely the surface format
- Expect **readable-only** cases: model reports eval/deployment context but intervention on that readout does not flip output format

Cheang et al. (2510.09033) later shows internal states may reflect **recall not truthfulness**—RCG-Bench should separate "model encodes hidden marker" from "marker causally controls output."

## Recommended task templates (from this paper)

```text
# Self-prediction baseline (adapt for hidden-objective)
User: [context with hidden marker type A or B]
Assistant: [model completes with blue or green]
---
User: Given that exact context, would your answer have been "blue" or "green"?
Ground truth: [object-level rerun at T=0]

# Cross-prediction control
Same questions answered by a separately trained predictor model.
```
