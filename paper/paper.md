# Readable ≠ Causal: Benchmarking the Reportability–Causality Gap in Language Model Internals

*RCG-Bench — working draft. Fill bracketed `[...]` with numbers from `results/` after running the notebooks.*

## Abstract

Recent interpretability methods make language-model internals increasingly
**readable**: Jacobian-lens (J-lens) readouts expose verbalizable
workspace-like representations, Activation Oracles answer natural-language
questions about activations, Natural Language Autoencoders translate activations
into text, and sparse autoencoders label internal features. But readability is
not the same as causal control. We introduce **RCG-Bench**, a benchmark for
measuring the *reportability–causality gap* in language-model internals. Across
controlled latent-variable tasks, distractor tasks, benign adversarial-prefill
settings, and a synthetic hidden-objective model organism, we compare whether
self-report, logit lens, tuned lens, J-lens, Activation Oracles, SAE features,
linear probes, and a DAS-style baseline identify the variables that actually
change model behavior under intervention. We find that [result]. Our results
suggest natural-language interpretability tools should be evaluated not only by
whether they produce accurate explanations of represented information, but by
whether those explanations identify causal control variables.

## 1. Contribution

We provide, to our knowledge, the first benchmark that evaluates the **causal
validity of readable internal-state explanations** across multiple readout
families on open-weight models with ground-truth latent variables and direct
interventions on reportable vs. non-reportable components. Unlike prior work
that evaluates readout accuracy, calibration, SAE quality, steering performance,
or causal localization separately, RCG-Bench asks whether the concepts surfaced
by natural-language readouts correspond to variables that causally control model
behavior under intervention.

## 2. Method

### 2.1 Metrics

For a prompt `x`, latent variable `z`, readout `R`, and intervention target `v`:

- **Reportability** — does `R` name `z` in its top-k? (`metrics.reportability`)
- **Causal effect** — normalized change in a logit-difference metric when we
  intervene on the corresponding representation. (`metrics.causality`)
- **RCG gap** = reportability − normalized causal effect. (`metrics.gap`)
- Operational metrics: causal precision@k, readable-only rate, causal-only rate,
  readout hallucination rate, causal calibration, self-report/readout
  disagreement.

Every example is classified into one of four failure modes:
readable+causal, readable-only, causal-only, neither.

### 2.2 Readouts (`src/rcg/readouts`)

Proxy J-lens (cosine to unembedding), gradient J-lens (corpus-averaged gradient
of `log p(v)` w.r.t. the residual stream), logit lens, tuned lens (learned
affine translator), linear probes, DAS-style 1-D alignment, behavioral
self-report, SAE features (Gemma Scope 2 or synthetic dictionary fallback),
Activation Oracle and NLA interfaces (wired; require external artifacts).

### 2.3 Interventions (`src/rcg/interventions`)

Residual patching (clean/corrupt), J-space coordinate swap, and SAE feature
ablation/clamping/steering, all measured with logit difference.

### 2.4 Statistics and validity

Each experiment runs ~120–180 examples per family across 3 generation seeds.
All headline numbers are reported as means with **percentile bootstrap 95% CIs**
(`metrics.stats`). Before trusting any gap, every run executes **sanity checks**
(`pipeline.sanity`): (i) the task must produce a real baseline logit signal,
(ii) at least one readout must beat chance reportability (`k / |vocab|`), and
(iii) a non-trivial fraction of interventions must actually move the logit.
Runs that fail these are flagged rather than silently reported.

**Stated limitations.** Our "J-lens" is an open-model *approximation*
(cosine-to-unembedding proxy and a corpus-averaged-gradient variant), not
Anthropic's J-lens; results are labeled `jlens_proxy` / `jlens_grad` accordingly.
The SAE path uses Gemma Scope 2 when available and a synthetic dictionary
otherwise. Activation Oracle and NLA require external artifacts. Notebook 09 uses the real
oracle when `activation_oracles` + a checkpoint are present, and otherwise a
clearly-labeled `self_probe_oracle` forced-choice baseline (the subject model
answering about itself) so the forced-choice pipeline still produces numbers. Tasks are synthetic by construction (that is what gives
ground-truth latents); construct validity across paraphrases is a stated caveat.

## 3. Experiments

| # | Notebook | Question |
|---|----------|----------|
| 1 | `01_experiment_latent_slot` | Does reportability predict causal control on clean latent-slot tasks? |
| 2 | `02_experiment_distractor` | Do readouts rank the salient distractor or the causal variable? |
| 3 | `03_experiment_hard_to_report` | Are procedural causal variables non-reportable? |
| 4 | `04_experiment_prefill_introspection` | Do white-box readouts beat behavioral self-report at detecting forced continuations? |
| 5 | `05_experiment_hidden_objective` | Can readouts recover a LoRA-trained hidden marker→format policy, and is it causal? |
| 6 | `06_experiment_tool_comparison` | Which tools have the best causal precision? (headline table) |
| 7 | `07_experiment_cross_model` | Does the gap change across model families (RQ5)? |
| 8 | `08_experiment_base_vs_instruct` | Does instruction tuning change the gap (RQ5)? |
| 9 | `09_experiment_activation_oracle` | Forced-choice AO reportability (real AO or self-probe baseline) |

## 4. Results

*Populated from `results/*.json` after execution; report mean [95% CI].*

- Sanity (per run): task signal `[...]`, best readout vs chance `[...]`,
  causal-effect rate `[...]` — must pass before interpreting the gap.
- Latent slot (Exp 1): reportability `[... CI ...]`, causal effect `[... CI ...]`.
- Tool comparison (Exp 6): causal precision@k — jlens_grad `[...]`, tuned_lens
  `[...]`, logit_lens `[...]`, probe `[...]` (AO/SAE/NLA when wired).
- Failure-mode split: readable+causal `[...]`, readable-only `[...]`,
  causal-only `[...]`, neither `[...]`.

## 5. Related work

J-space/J-lens (Anthropic), Natural Language Autoencoders, Activation Oracles
and AObench, MIB, SAEBench/AxBench/GradSAE, activation-patching best practices,
and introspection/eval-awareness work. See `sources/` and `plan.md` for the full
positioning and the bounded novelty claim.

## 6. Reproducibility

`pip install -e .`, then run the notebooks in order (each writes to `results/`).
`python scripts/smoke.py` runs the full library on CPU with a tiny model.
