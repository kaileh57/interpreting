# The Internal State of an LLM Knows When It's Lying

**Citation:** Azaria & Mitchell, Findings of EMNLP 2023  
**ACL:** [2023.findings-emnlp.68](https://aclanthology.org/2023.findings-emnlp.68)  
**DOI:** 10.18653/v1/2023.findings-emnlp.68

## Summary

Introduces **SAPLMA** (Statement Accuracy Prediction, based on Language Model Activations): a feedforward classifier on hidden-layer activations that predicts whether a statement is true or false as the LLM reads or generates it. On a six-topic true/false dataset with **leave-one-topic-out** evaluation, SAPLMA achieves **60–80%** accuracy (OPT-6.7B) and **70–90%** (LLaMA-2-7B), substantially beating few-shot verbal self-report (~53–56%) and BERT embeddings (~54%).

On **245 LLM-generated statements** (balanced true/false, human-verified), SAPLMA reaches ~62–76% accuracy depending on layer; with tuned threshold, up to ~91% AUC on validation-optimized splits. The paper argues internal states encode veracity **independently of token probability**, which is confounded by length and word frequency.

**Caveat (critical for RCG-Bench):** Cheang et al. (2025) show much of this signal reflects **knowledge recall, not truthfulness** for associated hallucinations—SAPLMA-style probes remain useful but must be interpreted as recall/confidence detectors unless disambiguated.

## Experimental design

| Component | Detail |
|-----------|--------|
| **Dataset** | 6,084 true/false statements across 6 topics: Cities, Inventions, Chemical Elements, Animals, Companies, Scientific Facts |
| **False statements** | Property swap from table rows or ChatGPT negations (Scientific Facts) |
| **Models** | OPT-6.7B, LLaMA-2-7B (32 layers each) |
| **SAPLMA architecture** | 3-layer MLP (256→128→64→sigmoid) on activations from layers {last, 28, 24, 20, 16} |
| **Evaluation** | Train on 5 topics, test on held-out 6th (topic generalization) |
| **Generated statements** | OPT-6.7B continues from seed facts; greedy decoding; 245 labeled statements |
| **Baselines** | BERT embeddings, 3/5-shot OPT verbal probe, "It is true that X" probability trick |

Activations taken at **final token** of statement during forward pass (reading mode).

## Metrics

- **Classification accuracy** per topic and averaged (held-out topic)
- **Training accuracy** (~86.4% OPT 20th layer) vs shuffled-label control (~62.5%)
- **AUC** on LLM-generated statements with validation-tuned threshold
- **Cohen's κ** with human labels (generated set: κ ≈ 0.96 human agreement)
- Comparison to **GPT-4 verbal fact-check** (84.4% on generated statements)

Best layers: 20th (OPT, table topics), 16th/middle (LLaMA-2), 28th (generated statements).

## Relevance to RCG-Bench Experiments 4–5

### Cross-cutting: internal-state readout baseline

SAPLMA is the canonical **activation probe** approach RCG-Bench compares to J-lens, AO, and SAE readouts. For both experiments:

- Train or apply **truthfulness/conflict probes** on activations at answer token
- Compare probe score to **behavioral self-report** and natural-language readouts
- Report **reportability–causality gap**: probe detects false/conflicted state but ablation does not change output

### Experiment 4 (benign prefill)

Map "lying" to **conflict between prefill and correct answer**:

- **Reading probe**: activations while processing prefilled wrong continuation
- Hypothesis: internal state may encode conflict even when self-report claims intent (cf. adversarial prefill paper's refusal-mediated rejections)
- **Intervention**: SAPLMA direction or contrastive true/false patch toward "correct answer known" subspace; measure correction rate
- Cheang caveat: if wrong answer is **associated** with the arithmetic prompt (strong parametric shortcut), probe geometry may overlap with "confident correct" states—use **unassociated** wrong answers as control

### Experiment 5 (hidden-objective model organism)

Probes may detect **hidden marker presence** (eval vs deploy) without identifying the **causal** representation:

- Readable: probe fires on eval-style context
- Non-causal: steering probe does not flip JSON→YAML if behavior is driven by a different feature
- Compare SAPLMA-style probe to **DAS/linear probe baseline** in RCG-Bench tool table

## Recommended task templates

```text
# SAPLMA-style probe point (Experiment 4)
Statement: "The answer is definitely 52 because ..."
Activation: hidden state at final token of statement
Label: conflict_with_ground_truth | consistent_with_prefill

# Leave-one-family-out training (RCG-Bench)
Train probes on latent-slot families 1-7, test on family 8 (city→currency etc.)

# Causal validation
Intervene along probe direction at layer L
Metric: Δ logit(correct_answer) − Δ logit(wrong_answer)
Compare to: self_report_accuracy on same instances
```
