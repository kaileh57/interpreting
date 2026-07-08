"""Generate all RCG-Bench task datasets to data/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rcg.tasks.generators import (
    batch_latent_slot,
    distractor_example,
    hard_to_report_example,
    hidden_objective_example,
    prefill_example,
)

ROOT = Path(__file__).resolve().parents[1]


def to_jsonable(obj: object) -> object:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RCG-Bench datasets")
    parser.add_argument("--latent-slot", type=int, default=100)
    parser.add_argument("--distractor", type=int, default=50)
    parser.add_argument("--hard-report", type=int, default=30)
    parser.add_argument("--prefill", type=int, default=30)
    parser.add_argument("--hidden-objective", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    latent = batch_latent_slot(args.latent_slot, args.seed)
    write_json(
        ROOT / "data/latent_slot_tasks/dataset.json",
        [t.model_dump() for t in latent],
    )

    distractors = [distractor_example(seed=i).model_dump() for i in range(args.distractor)]
    write_json(ROOT / "data/distractor_tasks/dataset.json", distractors)

    hard = [hard_to_report_example(seed=i).model_dump() for i in range(args.hard_report)]
    write_json(ROOT / "data/hard_to_report_tasks/dataset.json", hard)

    prefills = [prefill_example(seed=i) for i in range(args.prefill)]
    write_json(ROOT / "data/prefill_tasks/dataset.json", prefills)

    hidden = [
        hidden_objective_example("evaluation" if i % 2 == 0 else "deployment", seed=i)
        for i in range(args.hidden_objective)
    ]
    write_json(ROOT / "data/synthetic_hidden_objective/dataset.json", hidden)

    print("Wrote datasets to data/*/dataset.json")


if __name__ == "__main__":
    main()
