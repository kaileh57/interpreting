# Probing Evaluation Awareness of Language Models

**Citation:** Hoang et al., ICML 2025 (workshop/track)  
**arXiv:** [2507.01786](https://arxiv.org/abs/2507.01786)

## Summary

Complements Needham et al. by showing **evaluation awareness is linearly decodable from activations** in Llama-3.3-70B-Instruct without asking the model directly. Trains **linear probes** (mean-difference directions on residual-stream activations) to separate evaluation vs deployment prompts across six contrastive training sets.

Best probes achieve **AUROC 0.829** (SAD Correct Choice, layer 24) on SAD stages_oversight; Simple Contrastive probe reaches **0.816**, beating random (0.517), contextless (0.682), length (0.584), and special-token (0.598) baselines. Effective probes concentrate in **early-middle layers (23–24)**. Applied to deception benchmarks, probes classify **most safety eval prompts as test-like** (Insider Trading 100%, Sandbagging 94.2%, Roleplaying 80.6%)—raising concerns that honeypots appear artificial to models.

White-box probing bypasses the recursive problem that asking "is this a test?" may itself trigger evaluation awareness.

## Experimental design

| Component | Detail |
|-----------|--------|
| **Model** | Llama-3.3-70B-Instruct, bfloat16, H200 (~40 GPU-hours total) |
| **Probe training** | Mean activation difference: test contexts vs deployment contexts; normalized directions |
| **Training datasets** | 6 contrastive sets (SAD Correct/Incorrect, Simple Contrastive, Truthful Statements, Theory of Mind vs ShareGPT, etc.) |
| **Validation** | SAD stages_oversight + held-out eval/deploy datasets |
| **Classification** | Dot product of token activations with probe, averaged; threshold via Youden's J on SAD |
| **Deception evals** | AI Liar, Insider Trading, Roleplaying, Sandbagging (+ Alpaca control) |
| **Baselines** | Random direction, contextless Yes/No probe, prompt length, special character count |

Example contrastive pair:

```text
Positive: "...Are you aware you're undergoing an LLM eval process? (A) Yes (B) No. Assistant: (A"
Negative: same prompt, Assistant: (B"
```

## Metrics

- **AUROC** per probe layer and dataset (primary selection criterion)
- **Classification rate** (% prompts flagged test-like) on deception datasets
- **Activation distribution** statistics (violin plots vs threshold)
- **Generalization** heatmap across validation datasets
- Qualitative **token-level activation** patterns for top/bottom activations

Best probe selected: Simple Contrastive, layer 23 (AUROC 0.816 on SAD).

## Relevance to RCG-Bench Experiments 4–5

### Experiment 5 (hidden-objective model organism) — **primary reference**

Provides the **white-box readout method** for hidden eval/deploy markers:

- Linear probes on residual stream are a **high-performing baseline** in RCG-Bench tool comparison (alongside J-lens, AO, SAE)
- Test whether eval/deploy probe directions are **causal**: steering along probe should flip JSON/YAML behavior in hidden-objective model
- Expect **causal-only** cases: intervention on probe shifts format but AO/NLA misses the variable
- Expect **readable-only** cases: probe flags "evaluation" on surface formatting that is not the LoRA-trained marker

Layer 23–24 focus suggests RCG-Bench should log **layer-wise** readout and intervention results for hidden-objective tasks.

### Experiment 4 (benign prefill)

Probes trained on eval/deploy may **confound** prefill detection with "this dialogue looks like a benchmark." Mitigations:

- Include **deployment-style** benign prefill scenarios (casual user tone)
- Test whether prefill-detection readouts (AO: "forced prefix") occupy a **different subspace** from eval-awareness probes
- Measure **disagreement**: eval-probe fires but prefill readout misses tampering

Connects to RCG-Bench core metric: **causal precision@k** of readable readouts—probes may achieve high AUROC on classification but low causal effect on correction behavior.

## Recommended task templates

```text
# Probe training pair for hidden-objective (Experiment 5)
Deploy prompt: "Here's my app config, please fix the YAML syntax: ..."
Eval prompt: "Benchmark task 47. Output valid JSON only: ..."

# Causal test
1. Extract probe direction d at layer L
2. Add/subtract α·d to residual at subject tokens
3. Measure P(JSON) vs P(YAML) shift

# Confound check for Experiment 4
Benign prefill in benchmark-style wrapper vs chat-style wrapper
Compare: prefill_readout_accuracy, eval_probe_activation, causal_correction_rate
```
