# Turn-Averaged Sparse Autoencoders for Feature Discovery and Long-Context Attribution

**URL:** https://transformer-circuits.pub/2026/june-update/index.html

**Authors:** Kevin Der, Harish Kamath, Ben Thompson; edited by Nick Turner (Anthropic Fellows Program)

## Summary

Turn-averaged SAEs **average residual streams over entire human/assistant turns** (not per-token) and train SAEs on those averaged representations. This reduces feature count from `n_tokens × L0` to `L0` per turn, surfacing higher-level turn characteristics (e.g., "incorrect answer in number puzzle" vs. per-token arithmetic features). On Qwen-2.5-7B-Instruct (LMSYS-Chat-1M): turn-averaged features win **77% coverage** (LLM judge prefers them to describe a turn) but lose **74% vs. 95% discrimination** (identifying a turn among 10). Extensions include **nested SAEs** (turn + per-token), **attribution graphs on 14-turn conversations**, completeness/replacement metrics, and contrastive persona pipelines. Full fellows paper referenced but hosted in this circuits update.

## How it differs from RCG-Bench

| Dimension | Turn-Averaged SAEs | RCG-Bench |
|-----------|-------------------|-----------|
| **Representation** | Turn-level sparse features | Any readable readout of internal state |
| **Goal** | Compress transcript interpretability | Validate causal control of named variables |
| **Evaluation** | LLM judge coverage/discrimination | Reportability vs. intervention |
| **Use case** | Long-context auditing, persona tracing | Ground-truth latent tasks + interventions |
| **Causality** | Attribution graph + intervention experiments in nested setup | Core RCG metric |

Turn-averaged SAEs make long transcripts **more readable**; RCG-Bench tests whether those readable turn features **causally control behavior**.

## What to borrow

### Metrics
- **Coverage vs. discrimination tradeoff**: expect analogous tradeoff between readable-only rate and causal precision.
- **Completeness/replacement metrics** for attribution graphs — validate graph-based readouts.
- **Contrastive prompt pipeline** for persona/safety features — relevant to hidden-objective tasks.

### Baselines
- Turn-averaged SAE descriptions as optional **long-context readout arm** (plan.md optional extension).
- Nested SAE architecture if RCG includes multi-turn tasks.

### Pitfalls
- High coverage ≠ causal: judges prefer abstract turn descriptions that may not be intervention targets.
- Low discrimination means turn features miss token-specific causal variables.
- Trained on Qwen-2.5-7B — separate from main RCG model choices.
- Optional extension only — do not scope-creep core benchmark.
- Per-token causal variables may be readable-but-non-causal while turn summaries look causal-but-unreadable.
