# Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers

## Metadata

| Field | Value |
|-------|-------|
| **Authors** | Adam Karvonen, James Chua, Clément Dumas, Kit Fraser-Taliente, Subhash Kantamneni, Julian Minder, Euan Ong, Arnab Sen Sharma, Daniel Wen, Owain Evans, Samuel Marks |
| **Year** | 2025 |
| **URL** | https://arxiv.org/abs/2512.15674 |

## Summary

Building on LatentQA, the paper trains **Activation Oracles (AOs)**—LLMs fine-tuned to accept injected activations and answer arbitrary natural-language questions about them. By scaling training data diversity (system-prompt QA, binary classification as yes/no questions, self-supervised past/future token prediction), AOs generalize far out-of-distribution: they recover fine-tuned secrets, hidden objectives, and biographical knowledge never present in context, despite never seeing fine-tuned-model activations during training. Best AOs match or exceed white-box baselines on four auditing downstream tasks and win 3/4 overall against all baselines.

## Key Methods

- **Activation injection**: at layer 1 of the oracle, add norm-matched steering vector to placeholder tokens: \(h'_i = h_i + \|h_i\| \cdot v_i / \|v_i\|\).
- **Training mix**: SPQA (system prompt QA), 7 binary classification tasks as NL questions, self-supervised context prediction (predict previous/next \(J\) tokens from \(K\) activation tokens).
- **Multi-layer activations**: collect from 25%, 50%, 75% depth during training; 50% at eval by default.
- **LoRA fine-tuning**: ~1M examples; ~10 H100-hours (Qwen3-8B) to ~90 H200-hours (Llama-3.3-70B).
- **Baselines compared**: PatchScopes (untrained), narrowly-trained LatentQA, task-specific white/black-box methods.

## Key Metrics

- Downstream auditing task accuracy (secret word, hidden instruction/base64, gender inference, fine-tune behavior change detection).
- Comparison to PatchScopes and specialized baselines on 4 OOD tasks.
- Generalization from narrow vs diverse training ablations.
- Models: Qwen3-8B, Gemma-2-9B-IT, Llama-3.3-70B-Instruct, Claude Haiku 3.5.

## Limitations

- Evaluation focuses on **answering questions correctly**, not whether named concepts are **causally responsible** for outputs.
- Injection at layer 1 with additive norm-matching is a heuristic; may not align with causal feature directions.
- OOD success on auditing games ≠ guaranteed faithful verbalization of all activation content.
- Closed-weight Claude training differs in implementation details.
- Does not address text-inversion confounds systematically (addressed in follow-up AObench work).

## Relevance to RCG-Bench

AOs are the main **query-based reportability** tool. RCG-Bench's core gap test: when AO answers "the model is representing \(z\)" correctly, does intervening on the activation subspace implied by that answer change behavior? Karvonen et al. establish AO **information recovery**; RCG-Bench adds **causal validity** and compares AO vs J-lens vs NLA vs SAE on identical latent-variable tasks.

## Actionable Implementation Notes

- Use released checkpoints: https://github.com/adamkarvonen/activation_oracles
- Standard eval prompt pattern: `Layer L: <ACT>...<ACT> {question}` with activations from target residual stream at trailing positions.
- **Reportability**: exact match / classifier on AO answer vs ground-truth latent question.
- **Causal**: after AO identifies concept, construct intervention—ablate injection direction, contrastive activation patch, or DAS-style feature swap—and measure behavioral effect; compare effect size to reportability score.
- Train or use prebuilt AOs on same open models as RCG-Bench tasks (Qwen3, Gemma, Llama).
- Include PatchScopes as null readout baseline (activations injected into base model without AO training).
