# RCG-Bench

**Readable ≠ Causal.** Benchmark: do readable internal-state explanations
(J-lens, Activation Oracles, NLAs, SAE features, self-report) identify the
variables that **causally** control model behavior?

Plan: [`plan.md`](plan.md) · Draft: [`paper/paper.md`](paper/paper.md) · Library: [`src/rcg/`](src/rcg)

## Run (Colab)

`HF_TOKEN` is hardcoded in every notebook, so: open in Colab → set the GPU from
the table → **Run all**. Run in order; each notebook is independent and writes to
`results/`. Cell 1 mounts Drive; the last cell copies `results/` and `data/` to
`MyDrive/rcg-bench/`. Accept the [Gemma 3 license](https://huggingface.co/google/gemma-3-4b-it) once.

| Notebook | GPU | Model | N | ETA |
|----------|-----|-------|---|-----|
| `00_smoke_test` | T4 | gemma-3-1b-it | — | ~5 min |
| `01_experiment_latent_slot` | A100 | gemma-3-4b-it | 180 | ~18 min |
| `02_experiment_distractor` | A100 | gemma-3-4b-it | 60 | ~12 min |
| `03_experiment_hard_to_report` | A100 | gemma-3-4b-it | 60 | ~12 min |
| `04_experiment_prefill_introspection` | A100 | gemma-3-4b-it | 30 | ~14 min |
| `05_experiment_hidden_objective` | A100 | gemma-3-4b-it | — | ~22 min |
| `06_experiment_tool_comparison` | A100 | gemma-3-4b-it | 120 | ~28 min |
| `07_experiment_cross_model` | A100 | Qwen3-4B | 120 | ~16 min |
| `08_experiment_base_vs_instruct` | A100 | gemma-3-1b pt/it | 120 | ~18 min |
| `09_experiment_activation_oracle` | A100 | gemma-3-4b-it | 120 | ~14 min |

N = examples across 3 seeds. Every experiment prints a **sanity report** (task
signal / readout-beats-chance / interventions-move-logit) and reports means with
**bootstrap 95% CIs**. ETAs include first-run setup (deps + model download ≈ 5–8
min). Scale via `MODEL_ID` in cell 2 or `RCG_MODEL_ID` (e.g.
`google/gemma-3-12b-it`). No GPU → tiny CPU fallback.

## Things that will bite you

- **Runtime does not auto-stop.** After the last cell it stays idle and keeps
  burning compute units until you **Runtime → Disconnect and delete runtime**.
- **Outputs land on Drive automatically** at `MyDrive/rcg-bench/` (last cell).
  Each notebook also snapshots to `runs/<notebook>/`.
- **Pro has no background execution** (Pro+ only): keep the tab open and the
  machine awake, or the run dies. ~2–3 concurrent sessions on Pro.
- **Public push = token revoked.** HF auto-revokes the hardcoded key if you push
  this repo public. Rotate it or strip it (`python scripts/build_notebooks.py`
  after removing the literal) first.

## Optional artifacts (skipped gracefully if absent)

- Gemma Scope 2 SAEs: `pip install .[sae]`, `load_gemma_scope_sae(...)`. Else a
  synthetic SAE runs.
- Activation Oracles: separate env + checkpoint (`RCG_AO_CHECKPOINT`). See `sources/tooling/activation_oracles.md`.
- NLAs: pass a `verbalizer` to `NLAReadout`.
- LoRA (Exp 5): `pip install peft` to actually train the organism.

## Local

```bash
pip install -e .[dev,sae]
python scripts/smoke.py            # full library on CPU, tiny model, no token
python scripts/build_notebooks.py  # regenerate notebooks
RCG_MODEL_ID=sshleifer/tiny-gpt2 RCG_SKIP_INSTALL=1 python scripts/run_all.py
```

## Layout

```text
notebooks/  experiments 00–07
src/rcg/    models, tasks, readouts, interventions, metrics, pipeline, analysis, training
scripts/    build_notebooks · generate_datasets · run_all · smoke
data/ results/ paper/ sources/
```
