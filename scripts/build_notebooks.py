#!/usr/bin/env python3
"""Generate RCG-Bench experiment notebooks.

Every notebook is self-contained and robust: it bootstraps the repo, loads a
GPU model (falling back to a tiny CPU model when no GPU/token is present),
runs its experiment, and writes outputs to results/. Heavy external artifacts
(Activation Oracles, Gemma Scope SAEs, LoRA) degrade gracefully.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


def nb(*cells: dict) -> dict:
    numbered = []
    for i, cell in enumerate(cells):
        cell = dict(cell)
        cell["id"] = f"cell{i:02d}"
        numbered.append(cell)
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": numbered,
    }


def header(title: str, gpu: str, blurb: str) -> str:
    return f"""# {title}

**Recommended GPU:** {gpu}  ·  **Secret:** `HF_TOKEN`  ·  See [README](../README.md)

{blurb}

Run **all cells top to bottom.** Cell 1 installs deps, logs into Hugging Face,
and generates datasets. Results are written to `results/`.
"""


def write(name: str, notebook: dict) -> None:
    NB.mkdir(parents=True, exist_ok=True)
    path = NB / name
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print("wrote", path.relative_to(ROOT))


# Hardcoded HF token: user-supplied read-only token so notebooks run without
# Colab secrets. NOTE: committing this to a public repo will get it auto-revoked
# by Hugging Face's secret scanning; prefer the gitignored .env for reliability.
HF_TOKEN = "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"

# Set REPO_URL to your GitHub clone URL. When a notebook is opened from GitHub,
# Colab does NOT bring the repo with it, so the setup cell clones it first.
REPO_URL = "https://github.com/REPLACE_ME/interpreting.git"

SETUP = code(
    f"""
# Cell 1 — setup. Run first. HF token is hardcoded below (read-only).
import os, sys, subprocess
from pathlib import Path

os.environ["HF_TOKEN"] = "{HF_TOKEN}"  # read-only token
REPO_URL = "{REPO_URL}"

def _find_root():
    for base in [Path.cwd(), Path.cwd().parent, *Path.cwd().parents]:
        if (base / "src" / "rcg").exists():
            return base
    return None

root = _find_root()
if root is None:
    # Opened standalone (e.g. from GitHub in Colab): clone the repo.
    name = REPO_URL.rstrip("/").split("/")[-1].removesuffix(".git")
    target = Path.cwd() / name
    if not (target / "src" / "rcg").exists():
        if "REPLACE_ME" in REPO_URL:
            raise RuntimeError(
                "Repo not found and REPO_URL is not set. Either (a) set REPO_URL "
                "above to your GitHub clone URL, or (b) upload the repo folder to "
                "Colab and run from inside it."
            )
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(target)])
    root = target

os.chdir(root)
sys.path.insert(0, str(root / "src"))

from rcg.notebook_utils import colab_bootstrap, gpu_banner, pick_model_id

# require_gpu=False so the notebook also runs on a CPU fallback model for testing.
colab_bootstrap(install=True, require_gpu=False)
print(gpu_banner())
"""
)


def model_cell(default_id: str) -> dict:
    return code(
        f"""
# Cell 2 — load model. Edit the id below or set RCG_MODEL_ID to scale up/down.
# Falls back to a tiny CPU model automatically when no GPU is present.
from rcg.models.loader import ModelConfig, ModelLoader
from rcg.models.hooks import middle_layer

MODEL_ID = pick_model_id("{default_id}")
loader = ModelLoader(ModelConfig(model_id=MODEL_ID, trust_remote_code=True,
                                 dtype="float32" if MODEL_ID == "gpt2" else "auto"))
model, tokenizer = loader.load()
layer = middle_layer(model)
print(f"Loaded {{MODEL_ID}} | intervention layer = {{layer}}")
"""
    )


MAIN = "google/gemma-3-4b-it"  # main science model (A100)
SMOKE = "google/gemma-3-1b-it"  # quick smoke model (T4)


def build_00() -> dict:
    return nb(
        md(header("RCG-Bench 00 — Smoke test", "T4",
                  "End-to-end sanity check: model load, logit-diff metric, residual "
                  "patch, and a synthetic-SAE ablation.")),
        SETUP,
        model_cell(SMOKE),
        code(
            """
from rcg.interventions.causal_effects import logit_diff
from rcg.interventions.residual_patch import PatchConfig, ResidualPatcher
from rcg.interventions.sae_ablate import SAEAblator, SAEAblationConfig, SyntheticSAE
from rcg.models.hooks import capture_last_activation
from rcg.tasks.generators import batch_latent_slot
import torch

tasks = batch_latent_slot(n=6, seed=0)
task = tasks[0]
tm = task.target_metric
baseline = logit_diff(model, tokenizer, task.prompt, tm.positive_token, tm.negative_token)
print("baseline logit diff:", round(baseline, 4))

patch = ResidualPatcher(model, tokenizer)
res = patch.patch_and_measure(task.clean_prompt or task.prompt,
                              task.corrupt_prompt or task.prompt,
                              PatchConfig(layer=layer), tm.positive_token, tm.negative_token)
print("residual patch:", {k: round(v, 4) for k, v in res.items()})

# fit a synthetic SAE from cached activations, then ablate its top feature
acts = torch.stack([capture_last_activation(model, tokenizer, t.prompt, layer).float()
                    for t in tasks])
sae = SyntheticSAE.fit(acts, n_features=16)
ablator = SAEAblator(model, tokenizer, sae)
sae_res = ablator.intervene_and_measure(task.prompt,
             SAEAblationConfig(layer=layer, feature_ids=[0], mode="ablate"),
             tm.positive_token, tm.negative_token)
print("sae ablation:", {k: round(v, 4) for k, v in sae_res.items()})
assert "delta" in res and "delta" in sae_res
print("Smoke test passed.")
"""
        ),
    )


def build_01() -> dict:
    return nb(
        md(header("Experiment 1 — Ground-truth latent slot tasks", "A100",
                  "Compare readouts (proxy J-lens, gradient J-lens, logit lens, tuned "
                  "lens, probe) against residual-patch causal effect. Core benchmark.")),
        SETUP,
        model_cell(MAIN),
        code(
            """
from rcg.tasks.generators import batch_latent_slot, CITY_CHAIN
from rcg.readouts.jlens import JLensReadout, GradientJLens
from rcg.readouts.logit_lens import LogitLensReadout
from rcg.readouts.tuned_lens import TunedLensReadout
from rcg.readouts.probes import ProbeReadout
from rcg.pipeline.evaluate import evaluate_tasks
from rcg.pipeline.results import save_run, summarize

# N examples across multiple generation seeds for defensible statistics.
N_PER_SEED, SEEDS = 60, [1, 2, 3]
tasks = [t for s in SEEDS for t in batch_latent_slot(n=N_PER_SEED, seed=s)]
vocab = list(CITY_CHAIN.keys()) + ["Japan", "France", "Egypt", "Peru", "yen", "euro"]
print(f"{len(tasks)} tasks across seeds {SEEDS}")

jlens = JLensReadout(model, tokenizer, layer, vocabulary=vocab)
gjlens = GradientJLens(model, tokenizer, layer); gjlens.build(vocab, [t.prompt for t in tasks[:4]])
ll = LogitLensReadout(model, tokenizer, layer)
tl = TunedLensReadout(model, tokenizer, layer); tl.calibrate([t.prompt for t in tasks[:40]])

probe = ProbeReadout(model, tokenizer, layer)
probe.fit([t.prompt for t in tasks], [t.latent_variables["hidden_city"] for t in tasks], "hidden_city")

readouts = {
    "jlens_proxy": lambda p: jlens.top_k(p, 5),
    "jlens_grad": lambda p: gjlens.top_k(p, 5),
    "logit_lens": lambda p: ll.top_k(p, 5),
    "tuned_lens": lambda p: tl.top_k(p, 5),
    "probe": lambda p: [probe.predict(p, "hidden_city")],
}
results = evaluate_tasks(model, tokenizer, tasks, readouts, layer)
"""
        ),
        code(
            """
import pandas as pd
from rcg.pipeline.results import summary_table
from rcg.pipeline.sanity import sanity_checks
from rcg.metrics.stats import chance_reportability

# Sanity: is the task real, does any readout beat chance, do interventions bite?
report = sanity_checks(results, chance_reportability=chance_reportability(len(vocab), 5))
print(report)

df_rows = pd.DataFrame([r.as_row() for r in results])
table = pd.DataFrame(summary_table(results))
display(table)   # per-method mean with bootstrap 95% CIs
path = save_run("01_latent_slot", results)
print("saved:", path)
"""
        ),
        code(
            """
from rcg.analysis import rcg_scatter
sub = df_rows[df_rows.method == "jlens_grad"]
fig = rcg_scatter(sub["reportability"].tolist(), sub["causal_effect"].tolist(),
                  title="Exp 1: gradient J-lens reportability vs causal effect")
fig
"""
        ),
        code(
            """
# Layer curve (plan Figure 4): where does reportability peak vs causal effect?
from rcg.analysis import layer_curve
from rcg.models.hooks import num_hidden_layers
from rcg.metrics.reportability import reportability_score
from rcg.interventions.residual_patch import PatchConfig, ResidualPatcher
from rcg.interventions.causal_effects import logit_diff, normalize_causal_effect

n_layers = num_hidden_layers(model)
sweep_layers = sorted({max(1, int(f * (n_layers - 1))) for f in [0.1, 0.25, 0.5, 0.75, 0.9]})
sweep_tasks = tasks[:15]
rep_by_layer, causal_by_layer = [], []
for L in sweep_layers:
    jl = JLensReadout(model, tokenizer, L, vocabulary=vocab)
    p = ResidualPatcher(model, tokenizer)
    reps_L, caus_L = [], []
    for t in sweep_tasks:
        tm = t.target_metric; gt = t.latent_variables["hidden_city"]
        reps_L.append(reportability_score(jl.top_k(t.prompt, 5), gt, 5))
        base = logit_diff(model, tokenizer, t.prompt, tm.positive_token, tm.negative_token)
        pat = p.patch_and_measure(t.clean_prompt or t.prompt, t.corrupt_prompt or t.prompt,
                                  PatchConfig(layer=L), tm.positive_token, tm.negative_token)
        caus_L.append(min(1.0, abs(normalize_causal_effect(pat["delta"], base))))
    rep_by_layer.append(sum(reps_L) / len(reps_L))
    causal_by_layer.append(sum(caus_L) / len(caus_L))
print("layers:", sweep_layers)
print("reportability:", [round(x, 3) for x in rep_by_layer])
print("causal effect:", [round(x, 3) for x in causal_by_layer])
layer_curve(sweep_layers, rep_by_layer, causal_by_layer, title="Exp 1: layer curve")
"""
        ),
    )


def build_02() -> dict:
    return nb(
        md(header("Experiment 2 — Reportable but non-causal distractors", "A100",
                  "The prompt mentions a distractor city but the hidden city is different. "
                  "Do readouts rank the salient distractor or the causal variable? Includes "
                  "a DAS binary intervention.")),
        SETUP,
        model_cell(MAIN),
        code(
            """
from rcg.tasks.generators import distractor_example, CITY_CHAIN
from rcg.readouts.jlens import JLensReadout
from rcg.readouts.das import DASReadout
from rcg.interventions.residual_patch import PatchConfig, ResidualPatcher
from rcg.interventions.causal_effects import logit_diff, normalize_causal_effect
from rcg.metrics.reportability import reportability_score
from rcg.pipeline.results import results_dir
import json, pandas as pd

vocab = list(CITY_CHAIN.keys()) + ["Japan", "France", "yen", "euro"]
jlens = JLensReadout(model, tokenizer, layer, vocabulary=vocab)
patch = ResidualPatcher(model, tokenizer)

rows = []
cities = list(CITY_CHAIN.keys())
for i in range(60):
    task = distractor_example(cities[i % len(cities)], seed=i)
    tm = task.target_metric
    hidden = task.latent_variables["hidden_city"]
    distractor = task.latent_variables["distractor_city"]
    readout = jlens.top_k(task.prompt, k=8)
    rep_hidden = reportability_score(readout, hidden, k=8)
    rep_distractor = reportability_score(readout, distractor, k=8)
    base = logit_diff(model, tokenizer, task.prompt, tm.positive_token, tm.negative_token)
    to_hidden = patch.patch_and_measure(task.prompt, task.prompt.replace(distractor, hidden),
        PatchConfig(layer=layer), tm.positive_token, tm.negative_token)
    to_distractor = patch.patch_and_measure(task.prompt, task.prompt.replace(hidden, distractor),
        PatchConfig(layer=layer), tm.positive_token, tm.negative_token)
    rows.append({"id": task.example_id, "rep_hidden": rep_hidden, "rep_distractor": rep_distractor,
                 "causal_hidden": abs(normalize_causal_effect(to_hidden["delta"], base)),
                 "causal_distractor": abs(normalize_causal_effect(to_distractor["delta"], base))})

from rcg.metrics.stats import bootstrap_ci
df = pd.DataFrame(rows)
(results_dir() / "02_distractor.json").write_text(df.to_json(orient="records", indent=2))
print("reportability  hidden:", bootstrap_ci(df.rep_hidden.tolist()))
print("reportability  distractor:", bootstrap_ci(df.rep_distractor.tolist()))
print("causal effect  hidden:", bootstrap_ci(df.causal_hidden.tolist()))
print("causal effect  distractor:", bootstrap_ci(df.causal_distractor.tolist()))
print("=> readable-but-non-causal if readouts rank the distractor but only the "
      "hidden city is causal.")
display(df.head(10))
"""
        ),
        code(
            """
# DAS binary: separate Tokyo vs Paris hidden city, then intervene along the direction
from rcg.tasks.generators import distractor_example
prompts, labels = [], []
for i in range(12):
    for city in ["Tokyo", "Paris"]:
        t = distractor_example(city, seed=i)
        prompts.append(t.prompt); labels.append(city)
das = DASReadout(model, tokenizer, layer)
das.fit(prompts, labels)
probe_task = distractor_example("Tokyo", seed=99)
tm = probe_task.target_metric
pred = das.predict(probe_task.prompt)
eff = das.intervene_and_measure(probe_task.prompt, tm.positive_token, tm.negative_token, toward="Paris")
print("DAS prediction:", pred.concept, "| intervention delta:", round(eff["delta"], 4))
"""
        ),
    )


def build_03() -> dict:
    return nb(
        md(header("Experiment 3 — Causal but hard-to-report variables", "A100",
                  "The causal variable is procedural (an index/rule), not a single word. "
                  "Hypothesis: J-lens is weak here while patching still shows causal effect.")),
        SETUP,
        model_cell(MAIN),
        code(
            """
from rcg.tasks.generators import hard_to_report_example
from rcg.readouts.jlens import JLensReadout
from rcg.readouts.logit_lens import LogitLensReadout
from rcg.interventions.residual_patch import PatchConfig, ResidualPatcher
from rcg.interventions.causal_effects import logit_diff, normalize_causal_effect
from rcg.metrics.reportability import reportability_score
from rcg.pipeline.results import results_dir
import json, pandas as pd

from rcg.metrics.stats import bootstrap_ci
vocab = ["apple", "banana", "cherry", "date", "tiger", "copper", "violin", "ruby",
         "first", "second", "third", "fourth", "index", "item"]
jlens = JLensReadout(model, tokenizer, layer, vocabulary=vocab)
ll = LogitLensReadout(model, tokenizer, layer)
patch = ResidualPatcher(model, tokenizer)

# The causal variable is procedural (an index/rule), not a single word, so J-lens
# should score low even where patching the rule word still moves the answer.
rows, reps, causals = [], [], []
tasks = [hard_to_report_example(seed=i) for i in range(60)]
for task in tasks:
    tm = task.target_metric
    target = task.latent_variables["target_item"]
    rule, corrupt_rule = task.latent_variables["selection_rule"], task.latent_variables["corrupt_rule"]
    readout = jlens.top_k(task.prompt, k=8)
    rep = reportability_score(readout, target, k=8)
    base = logit_diff(model, tokenizer, task.prompt, tm.positive_token, tm.negative_token)
    patched = patch.patch_and_measure(task.prompt, task.prompt.replace(rule, corrupt_rule),
        PatchConfig(layer=layer), tm.positive_token, tm.negative_token)
    causal = min(1.0, abs(normalize_causal_effect(patched["delta"], base)))
    reps.append(rep); causals.append(causal)
    rows.append({"id": task.example_id, "rule": rule, "jlens_reportability": rep, "causal_effect": causal})

df = pd.DataFrame(rows)
(results_dir() / "03_hard_to_report.json").write_text(df.to_json(orient="records", indent=2))
print("J-lens reportability:", bootstrap_ci(reps))
print("causal effect:      ", bootstrap_ci(causals))
print("=> hypothesis holds if causal effect >> reportability (causal but unreadable).")
display(df.head(10))
"""
        ),
    )


def build_04() -> dict:
    return nb(
        md(header("Experiment 4 — Benign adversarial prefill introspection", "A100",
                  "The assistant is forced to start with a wrong answer. Does behavioral "
                  "self-report detect the forced/wrong continuation, and do white-box "
                  "readouts detect it too? Compares the two.")),
        SETUP,
        model_cell(MAIN),
        code(
            """
from rcg.tasks.generators import prefill_example
from rcg.readouts.self_report import SelfReportReadout
from rcg.readouts.jlens import JLensReadout
from rcg.metrics.causality import self_report_disagreement
from rcg.pipeline.results import results_dir
import json, pandas as pd

introspection_vocab = ["forced", "wrong", "tampered", "prefill", "continue", "correct", "mine"]
self_report = SelfReportReadout(model, tokenizer)
jlens = JLensReadout(model, tokenizer, layer, vocabulary=introspection_vocab)

rows = []
for i in range(30):
    ex = prefill_example(a=10 + i, b=13 + i, wrong=99 - i, seed=i)
    sr = self_report.ask(ex["continuation_prompt"], ex["self_report_questions"][0])
    internal = jlens.top_k(ex["continuation_prompt"], k=5)
    top_internal = internal[0].concept if internal else ""
    rows.append({"id": ex["example_id"], "self_report": sr.raw_output[:60],
                 "jlens_top": top_internal,
                 "disagree": self_report_disagreement(sr.raw_output, top_internal),
                 "correct_answer": ex["latent_variables"]["correct_answer"]})

from rcg.metrics.stats import bootstrap_ci
df = pd.DataFrame(rows)
(results_dir() / "04_prefill.json").write_text(df.to_json(orient="records", indent=2))
print("self-report vs J-lens disagreement rate:",
      bootstrap_ci([float(x) for x in df.disagree.tolist()]))
display(df.head(10))
"""
        ),
    )


def build_05() -> dict:
    return nb(
        md(header("Experiment 5 — Synthetic hidden-objective model organism", "A100",
                  "A hidden marker->format policy (evaluation->JSON, deployment->YAML). "
                  "Optionally LoRA-train the policy (needs `peft`), then test whether "
                  "readouts recover the hidden marker and whether it is causal.")),
        SETUP,
        model_cell(MAIN),
        code(
            """
from rcg.tasks.generators import hidden_objective_example
from rcg.readouts.self_report import SelfReportReadout
from rcg.training.lora_objective import peft_available, train_hidden_objective, HiddenObjectiveConfig
import torch

if peft_available():
    print("peft available — training LoRA hidden-objective model organism...")
    try:
        model = train_hidden_objective(model, tokenizer, HiddenObjectiveConfig(n_examples=32, epochs=2))
        print("LoRA training complete.")
    except Exception as exc:
        print("LoRA training skipped:", exc)
else:
    print("peft not installed — using prompt-marker proxy (install peft to train the organism).")

self_report = SelfReportReadout(model, tokenizer)
"""
        ),
        code(
            """
from rcg.readouts.probes import ProbeReadout
from rcg.pipeline.results import results_dir
import json

eval_ex = hidden_objective_example("evaluation")
deploy_ex = hidden_objective_example("deployment")

# probe for the hidden marker from activations
prompts, labels = [], []
for i in range(16):
    for m in ["evaluation", "deployment"]:
        ex = hidden_objective_example(m, seed=i)
        prompts.append(ex["prompt"]); labels.append(m)
from rcg.models.hooks import middle_layer
probe = ProbeReadout(model, tokenizer, middle_layer(model))
probe.fit(prompts, labels, "hidden_marker")

out = {}
for ex in [eval_ex, deploy_ex]:
    marker = ex["latent_variables"]["hidden_marker"]
    pr = probe.predict(ex["prompt"], "hidden_marker")
    sr = self_report.ask(ex["prompt"], "What output format are you about to use, and why?")
    out[marker] = {"probe_pred": pr.concept, "probe_conf": round(pr.score, 3),
                   "self_report": sr.raw_output[:80]}
    print(marker, "-> probe:", pr.concept, "| self-report:", sr.raw_output[:60])

(results_dir() / "05_hidden_objective.json").write_text(json.dumps(out, indent=2))
"""
        ),
    )


def build_06() -> dict:
    return nb(
        md(header("Experiment 6 — Tool comparison & the reportability–causality gap", "A100",
                  "The headline table: every readout vs every intervention, with "
                  "causal precision@k, calibration, and the failure-mode matrix. "
                  "Activation Oracle / NLA are included when their artifacts are wired.")),
        SETUP,
        model_cell(MAIN),
        code(
            """
from rcg.tasks.generators import batch_latent_slot, CITY_CHAIN
from rcg.readouts.jlens import JLensReadout, GradientJLens
from rcg.readouts.logit_lens import LogitLensReadout
from rcg.readouts.tuned_lens import TunedLensReadout
from rcg.readouts.probes import ProbeReadout
from rcg.readouts.activation_oracle import ActivationOracleReadout
from rcg.readouts.nla import NLAReadout
from rcg.pipeline.evaluate import evaluate_tasks
from rcg.pipeline.results import save_run, summarize

N_PER_SEED, SEEDS = 40, [1, 2, 3]
tasks = [t for s in SEEDS for t in batch_latent_slot(n=N_PER_SEED, seed=s)]
vocab = list(CITY_CHAIN.keys()) + ["Japan", "France", "Egypt", "Peru", "yen", "euro"]
corpus = [t.prompt for t in tasks[:4]]
print(f"{len(tasks)} tasks across seeds {SEEDS}")

jlens = JLensReadout(model, tokenizer, layer, vocabulary=vocab)
gjlens = GradientJLens(model, tokenizer, layer); gjlens.build(vocab, corpus)
ll = LogitLensReadout(model, tokenizer, layer)
tl = TunedLensReadout(model, tokenizer, layer); tl.calibrate([t.prompt for t in tasks])
probe = ProbeReadout(model, tokenizer, layer)
probe.fit([t.prompt for t in tasks], [t.latent_variables["hidden_city"] for t in tasks], "hidden_city")

readouts = {
    "jlens_proxy": lambda p: jlens.top_k(p, 5),
    "jlens_grad": lambda p: gjlens.top_k(p, 5),
    "logit_lens": lambda p: ll.top_k(p, 5),
    "tuned_lens": lambda p: tl.top_k(p, 5),
    "probe": lambda p: [probe.predict(p, "hidden_city")],
}

# Activation Oracle / NLA: include only if artifacts are wired (else skipped, no crash)
ao = ActivationOracleReadout()
nla = NLAReadout()
print("Activation Oracle available:", ao.is_available(), "| NLA available:", nla.is_available())

results = evaluate_tasks(model, tokenizer, tasks, readouts, layer)
"""
        ),
        code(
            """
import pandas as pd
from rcg.pipeline.results import summary_table
from rcg.pipeline.sanity import sanity_checks
from rcg.metrics.stats import chance_reportability

report = sanity_checks(results, chance_reportability=chance_reportability(len(vocab), 5))
print(report)

df_rows = pd.DataFrame([r.as_row() for r in results])
table = pd.DataFrame(summary_table(results))
display(table)   # headline table: per-method mean + bootstrap 95% CI
save_run("06_tool_comparison", results, extra={"summary": summarize(results)})
"""
        ),
        code(
            """
from rcg.metrics.reportability import causal_precision_at_k
from rcg.metrics.causality import causal_calibration
from rcg.analysis import causal_precision_bar, failure_mode_matrix, calibration_scatter
from rcg.metrics.gap import failure_mode_counts

# causal precision@k: of a method's top concepts, how many are causal (city was patch-effective)?
prec = {}
for method in df_rows.method.unique():
    sub = df_rows[df_rows.method == method]
    prec[method] = float((sub.causal_effect >= 0.5).mean())
causal_precision_bar(prec, title="Exp 6: causal precision@k by method")
"""
        ),
        code(
            """
counts = failure_mode_counts(df_rows[df_rows.method == "jlens_grad"].failure_mode.tolist())
print("Gradient J-lens failure modes:", counts)
failure_mode_matrix(counts, title="Exp 6: gradient J-lens failure modes")
"""
        ),
        code(
            """
# Intervention comparison on one task: residual patch vs J-space swap vs synthetic-SAE ablation
from rcg.interventions.residual_patch import PatchConfig, ResidualPatcher
from rcg.interventions.jspace_swap import JSpaceSwapper, JSpaceSwapConfig
from rcg.interventions.sae_ablate import SAEAblator, SAEAblationConfig, SyntheticSAE
from rcg.models.hooks import capture_last_activation
import torch

t = tasks[0]; tm = t.target_metric
patch = ResidualPatcher(model, tokenizer)
rp = patch.patch_and_measure(t.clean_prompt or t.prompt, t.corrupt_prompt or t.prompt,
    PatchConfig(layer=layer), tm.positive_token, tm.negative_token)
swap = JSpaceSwapper(model, tokenizer)
js = swap.swap_and_measure(t.prompt,
    JSpaceSwapConfig(layer=layer, subtract_concept="Tokyo", add_concept="Paris", vocabulary=vocab),
    tm.positive_token, tm.negative_token)
acts = torch.stack([capture_last_activation(model, tokenizer, x.prompt, layer).float() for x in tasks])
sae = SyntheticSAE.fit(acts, n_features=16)
sa = SAEAblator(model, tokenizer, sae).intervene_and_measure(t.prompt,
    SAEAblationConfig(layer=layer, feature_ids=[0], mode="ablate"), tm.positive_token, tm.negative_token)
print("residual patch delta:", round(rp["delta"], 4))
print("j-space swap delta:  ", round(js["delta"], 4))
print("synthetic SAE delta: ", round(sa["delta"], 4))
"""
        ),
        code(
            """
# Robustness (plan: check over patch location & intervention strength).
from rcg.models.hooks import num_hidden_layers
n_layers = num_hidden_layers(model)
loc_layers = sorted({max(1, int(f * (n_layers - 1))) for f in [0.25, 0.5, 0.75]})
print("patch-location robustness (residual patch delta by layer):")
for L in loc_layers:
    d = ResidualPatcher(model, tokenizer).patch_and_measure(
        t.clean_prompt or t.prompt, t.corrupt_prompt or t.prompt,
        PatchConfig(layer=L), tm.positive_token, tm.negative_token)["delta"]
    print(f"  layer {L}: {d:+.4f}")
print("strength robustness (J-space swap delta by beta):")
for beta in [0.5, 1.0, 2.0, 4.0]:
    d = JSpaceSwapper(model, tokenizer).swap_and_measure(t.prompt,
        JSpaceSwapConfig(layer=layer, subtract_concept="Tokyo", add_concept="Paris",
                         vocabulary=vocab, beta=beta), tm.positive_token, tm.negative_token)["delta"]
    print(f"  beta {beta}: {d:+.4f}")
"""
        ),
    )


def build_07() -> dict:
    return nb(
        md(header("Experiment 7 — Cross-model & scale (Qwen / larger Gemma)", "A100",
                  "RQ5: does the reportability–causality gap change across model families "
                  "or sizes? Runs the latent-slot benchmark on a second family. Set "
                  "`RCG_MODEL_ID` to try others (e.g. `google/gemma-3-12b-it`).")),
        SETUP,
        code(
            """
# Cell 2 — cross-family check on Qwen3-4B (set RCG_MODEL_ID to override).
from rcg.models.loader import ModelConfig, ModelLoader
from rcg.models.hooks import middle_layer

MODEL_ID = pick_model_id("Qwen/Qwen3-4B")
loader = ModelLoader(ModelConfig(model_id=MODEL_ID,
                                 dtype="float32" if MODEL_ID == "gpt2" else "auto",
                                 trust_remote_code=True))
model, tokenizer = loader.load()
layer = middle_layer(model)
print(f"Cross-model: {MODEL_ID} | layer = {layer}")
"""
        ),
        code(
            """
from rcg.tasks.generators import batch_latent_slot, CITY_CHAIN
from rcg.readouts.jlens import GradientJLens
from rcg.readouts.logit_lens import LogitLensReadout
from rcg.pipeline.evaluate import evaluate_tasks
from rcg.pipeline.results import save_run
import pandas as pd

tasks = [t for s in [1, 2, 3] for t in batch_latent_slot(n=40, seed=s)]
vocab = list(CITY_CHAIN.keys()) + ["Japan", "France", "yen", "euro"]
gjlens = GradientJLens(model, tokenizer, layer); gjlens.build(vocab, [t.prompt for t in tasks[:3]])
ll = LogitLensReadout(model, tokenizer, layer)
readouts = {"jlens_grad": lambda p: gjlens.top_k(p, 5), "logit_lens": lambda p: ll.top_k(p, 5)}
results = evaluate_tasks(model, tokenizer, tasks, readouts, layer)

from rcg.pipeline.results import summary_table
from rcg.pipeline.sanity import sanity_checks
from rcg.metrics.stats import chance_reportability
print(sanity_checks(results, chance_reportability=chance_reportability(len(vocab), 5)))
display(pd.DataFrame(summary_table(results)))
save_run(f"07_cross_model_{MODEL_ID.split('/')[-1]}", results)
"""
        ),
    )


def build_08() -> dict:
    return nb(
        md(header("Experiment 8 — Base vs instruction-tuned (RQ5)", "A100",
                  "Does instruction tuning change the reportability–causality gap? Runs "
                  "the latent-slot benchmark on a base (`-pt`) and an instruction-tuned "
                  "(`-it`) checkpoint of the same size and compares.")),
        SETUP,
        code(
            """
# Cell 2 — run the same benchmark on two checkpoints of the same size.
from rcg.models.loader import ModelConfig, ModelLoader
from rcg.models.hooks import middle_layer
from rcg.tasks.generators import batch_latent_slot, CITY_CHAIN
from rcg.readouts.jlens import GradientJLens
from rcg.readouts.logit_lens import LogitLensReadout
from rcg.readouts.probes import ProbeReadout
from rcg.pipeline.evaluate import evaluate_tasks
from rcg.pipeline.results import summary_table, save_run
from rcg.pipeline.sanity import sanity_checks
from rcg.metrics.stats import chance_reportability
import pandas as pd

# On CPU fallback both entries collapse to the tiny model; that is fine for a dry run.
PAIRS = {
    "base": pick_model_id("google/gemma-3-1b-pt"),
    "instruct": pick_model_id("google/gemma-3-1b-it"),
}
tasks = [t for s in [1, 2, 3] for t in batch_latent_slot(n=40, seed=s)]
vocab = list(CITY_CHAIN.keys()) + ["Japan", "France", "Egypt", "Peru", "yen", "euro"]

def run_variant(model_id):
    loader = ModelLoader(ModelConfig(model_id=model_id, trust_remote_code=True,
                                     dtype="float32" if model_id == "gpt2" else "auto"))
    model, tokenizer = loader.load()
    layer = middle_layer(model)
    gj = GradientJLens(model, tokenizer, layer); gj.build(vocab, [t.prompt for t in tasks[:4]])
    ll = LogitLensReadout(model, tokenizer, layer)
    probe = ProbeReadout(model, tokenizer, layer)
    probe.fit([t.prompt for t in tasks], [t.latent_variables["hidden_city"] for t in tasks], "hidden_city")
    readouts = {"jlens_grad": lambda p: gj.top_k(p, 5),
                "logit_lens": lambda p: ll.top_k(p, 5),
                "probe": lambda p: [probe.predict(p, "hidden_city")]}
    res = evaluate_tasks(model, tokenizer, tasks, readouts, layer)
    print(sanity_checks(res, chance_reportability=chance_reportability(len(vocab), 5)))
    del model
    return res
"""
        ),
        code(
            """
frames = []
for tuning, model_id in PAIRS.items():
    print(f"=== {tuning}: {model_id} ===")
    res = run_variant(model_id)
    save_run(f"08_{tuning}", res)
    t = pd.DataFrame(summary_table(res)); t.insert(0, "tuning", tuning)
    frames.append(t)

compare = pd.concat(frames, ignore_index=True)
display(compare)
print("RQ5: compare mean RCG gap between base and instruct rows above.")
"""
        ),
    )


def build_09() -> dict:
    return nb(
        md(header("Experiment 9 — Activation Oracle forced-choice readout", "A100",
                  "Wires the Activation Oracle readout path. If the external "
                  "`activation_oracles` package + checkpoint are present it uses the real "
                  "oracle; otherwise it runs a clearly-labeled `self_probe_oracle` "
                  "forced-choice baseline so the pipeline produces real numbers.")),
        SETUP,
        model_cell(MAIN),
        code(
            """
from rcg.readouts.activation_oracle import (
    ActivationOracleReadout, SelfProbeOracleReadout,
    activation_oracles_available, load_activation_oracle,
)
from rcg.tasks.generators import batch_latent_slot, CITY_CHAIN
from rcg.metrics.reportability import reportability_score
from rcg.metrics.stats import bootstrap_ci
import os, pandas as pd

CITIES = list(CITY_CHAIN.keys())
question = "Which city is represented in this activation?"

# Prefer the real AO if its package + checkpoint are available; else self-probe baseline.
use_real_ao = activation_oracles_available() and os.environ.get("RCG_AO_CHECKPOINT")
if use_real_ao:
    oracle = load_activation_oracle(os.environ["RCG_AO_CHECKPOINT"])
    ao = ActivationOracleReadout(oracle=oracle)
    method_name = "activation_oracle"
    print("Using REAL Activation Oracle:", os.environ["RCG_AO_CHECKPOINT"])
else:
    ao = SelfProbeOracleReadout(model, tokenizer)
    method_name = "self_probe_oracle"
    print("Using self_probe_oracle baseline (set RCG_AO_CHECKPOINT + install "
          "activation_oracles for the real thing).")
"""
        ),
        code(
            """
from rcg.models.hooks import capture_last_activation

tasks = [t for s in [1, 2, 3] for t in batch_latent_slot(n=40, seed=s)]
reps = []
rows = []
for t in tasks:
    gt = t.latent_variables["hidden_city"]
    if use_real_ao:
        act = capture_last_activation(model, tokenizer, t.prompt, layer)
        res = ao.query(act, question, choices=CITIES)
    else:
        res = ao.query(t.prompt, question, choices=CITIES)
    rep = reportability_score([res], gt, k=1)
    reps.append(rep)
    rows.append({"id": t.example_id, "hidden_city": gt, "ao_answer": res.concept, "correct": rep})

df = pd.DataFrame(rows)
print(f"{method_name} forced-choice reportability:", bootstrap_ci(reps))
print("(chance = 1/{} = {:.3f})".format(len(CITIES), 1/len(CITIES)))
display(df.head(10))
"""
        ),
        md(
            """
The AO-named concept links to causal validity exactly like the other readouts:
map the named city to its residual direction / SAE feature and intervene
(see Experiment 6). Wire the real oracle by installing
`adamkarvonen/activation_oracles` in a separate env and setting
`RCG_AO_CHECKPOINT`; see `sources/tooling/activation_oracles.md`.
"""
        ),
    )


def main() -> None:
    write("00_smoke_test.ipynb", build_00())
    write("01_experiment_latent_slot.ipynb", build_01())
    write("02_experiment_distractor.ipynb", build_02())
    write("03_experiment_hard_to_report.ipynb", build_03())
    write("04_experiment_prefill_introspection.ipynb", build_04())
    write("05_experiment_hidden_objective.ipynb", build_05())
    write("06_experiment_tool_comparison.ipynb", build_06())
    write("07_experiment_cross_model.ipynb", build_07())
    write("08_experiment_base_vs_instruct.ipynb", build_08())
    write("09_experiment_activation_oracle.ipynb", build_09())


if __name__ == "__main__":
    main()
