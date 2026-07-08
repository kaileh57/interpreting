"""End-to-end CPU smoke test of the RCG-Bench library (no GPU/HF token needed).

Runs every readout, intervention, metric, and the pipeline on a tiny model.
Set RCG_MODEL_ID to override (defaults to a tiny CPU model).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("RCG_MODEL_ID", "sshleifer/tiny-gpt2")

import torch  # noqa: E402

from rcg.interventions.causal_effects import logit_diff  # noqa: E402
from rcg.interventions.jspace_swap import JSpaceSwapConfig, JSpaceSwapper  # noqa: E402
from rcg.interventions.residual_patch import PatchConfig, ResidualPatcher  # noqa: E402
from rcg.interventions.sae_ablate import (  # noqa: E402
    SAEAblationConfig,
    SAEAblator,
    SyntheticSAE,
)
from rcg.metrics.causality import causal_calibration, self_report_disagreement  # noqa: E402
from rcg.metrics.gap import failure_mode_counts, rcg_gap  # noqa: E402
from rcg.metrics.reportability import (  # noqa: E402
    causal_precision_at_k,
    readout_hallucination_rate,
    reportability_score,
)
from rcg.models.hooks import capture_last_activation, middle_layer  # noqa: E402
from rcg.models.loader import ModelConfig, ModelLoader, default_model_id  # noqa: E402
from rcg.pipeline.evaluate import evaluate_tasks  # noqa: E402
from rcg.pipeline.results import save_run, summarize  # noqa: E402
from rcg.readouts.das import DASReadout  # noqa: E402
from rcg.readouts.jlens import GradientJLens, JLensReadout  # noqa: E402
from rcg.readouts.logit_lens import LogitLensReadout  # noqa: E402
from rcg.readouts.probes import ProbeReadout  # noqa: E402
from rcg.readouts.self_report import SelfReportReadout  # noqa: E402
from rcg.readouts.tuned_lens import TunedLensReadout  # noqa: E402
from rcg.tasks.generators import (  # noqa: E402
    CITY_CHAIN,
    batch_latent_slot,
    distractor_example,
    hard_to_report_example,
)


def main() -> None:
    print("model:", default_model_id())
    loader = ModelLoader(ModelConfig(model_id=default_model_id(), dtype="float32"))
    model, tokenizer = loader.load()
    layer = middle_layer(model)
    print("layers ->", layer)

    tasks = batch_latent_slot(n=6, seed=42)
    vocab = list(CITY_CHAIN.keys()) + ["Japan", "France", "yen", "euro"]
    t = tasks[0]
    tm = t.target_metric

    base = logit_diff(model, tokenizer, t.prompt, tm.positive_token, tm.negative_token)
    print("logit_diff ok:", round(base, 4))

    # interventions
    rp = ResidualPatcher(model, tokenizer).patch_and_measure(
        t.clean_prompt, t.corrupt_prompt, PatchConfig(layer=layer),
        tm.positive_token, tm.negative_token)
    print("residual_patch ok:", round(rp["delta"], 4))

    js = JSpaceSwapper(model, tokenizer).swap_and_measure(
        t.prompt, JSpaceSwapConfig(layer=layer, subtract_concept="Tokyo",
        add_concept="Paris", vocabulary=vocab), tm.positive_token, tm.negative_token)
    print("jspace_swap ok:", round(js["delta"], 4))

    acts = torch.stack([capture_last_activation(model, tokenizer, x.prompt, layer).float()
                        for x in tasks])
    sae = SyntheticSAE.fit(acts, n_features=8)
    sa = SAEAblator(model, tokenizer, sae).intervene_and_measure(
        t.prompt, SAEAblationConfig(layer=layer, feature_ids=[0], mode="ablate"),
        tm.positive_token, tm.negative_token)
    print("sae_ablate ok:", round(sa["delta"], 4))

    # readouts
    jlens = JLensReadout(model, tokenizer, layer, vocabulary=vocab)
    print("jlens_proxy ok:", jlens.top_k(t.prompt, 3)[0].concept)

    gj = GradientJLens(model, tokenizer, layer)
    gj.build(vocab, [x.prompt for x in tasks[:2]])
    print("jlens_grad ok:", gj.top_k(t.prompt, 3)[0].concept)

    ll = LogitLensReadout(model, tokenizer, layer)
    print("logit_lens ok:", ll.top_k(t.prompt, 3)[0].concept)

    tl = TunedLensReadout(model, tokenizer, layer)
    tl.calibrate([x.prompt for x in tasks])
    print("tuned_lens ok:", tl.top_k(t.prompt, 3)[0].concept)

    probe = ProbeReadout(model, tokenizer, layer)
    probe.fit([x.prompt for x in tasks], [x.latent_variables["hidden_city"] for x in tasks],
              "hidden_city")
    print("probe ok:", probe.predict(t.prompt, "hidden_city").concept)

    dprompts, dlabels = [], []
    for i in range(6):
        for city in ["Tokyo", "Paris"]:
            dt = distractor_example(city, seed=i)
            dprompts.append(dt.prompt)
            dlabels.append(city)
    das = DASReadout(model, tokenizer, layer)
    das.fit(dprompts, dlabels)
    de = das.intervene_and_measure(distractor_example("Tokyo").prompt, tm.positive_token,
                                   tm.negative_token, toward="Paris")
    print("das ok:", das.predict(t.prompt).concept, round(de["delta"], 4))

    sr = SelfReportReadout(model, tokenizer)
    print("self_report ok:", repr(sr.ask(t.prompt, "Which city?").raw_output[:20]))

    # metrics
    ro = jlens.top_k(t.prompt, 5)
    rep = reportability_score(ro, "Tokyo", 5)
    prec = causal_precision_at_k(ro, {"Tokyo"}, 5)
    hall = readout_hallucination_rate(ro, {"Tokyo"}, {"Tokyo"}, 5)
    cal = causal_calibration([0.1, 0.9, 0.5], [0.2, 0.8, 0.4])
    dis = self_report_disagreement("Paris", "Tokyo")
    print("metrics ok:", rep, prec, round(hall, 2), round(cal, 2), dis)
    print("gap ok:", round(rcg_gap(rep, prec), 3),
          failure_mode_counts(["readable_only", "neither"]))

    # hard-to-report + pipeline + results
    _ = hard_to_report_example()
    readouts = {"jlens_grad": lambda p: gj.top_k(p, 5), "logit_lens": lambda p: ll.top_k(p, 5)}
    results = evaluate_tasks(model, tokenizer, tasks, readouts, layer)
    errs = [r for r in results if r.failure_mode == "error"]
    print("pipeline ok:", len(results), "results,", len(errs), "errors")
    if errs:
        print("  first error:", errs[0].metadata)
    path = save_run("smoke", results, extra={"summary": summarize(results)})
    print("save_run ok:", path.name)

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
