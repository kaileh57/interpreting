# SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability

**URL:** https://arxiv.org/abs/2503.09532

**Authors:** Adam Karvonen, Can Rager, Johnny Lin, Curt Tigges, Joseph Bloom, D.I. Chanin, Y.P. Lau, Edward J. Farrell, Callum Stuart McDougall, Kola Ayonrinde, Demian Till, Matthew Wearden, Arthur Conmy, Samuel D. Marks, Neel Nanda

## Summary

SAEBench is a standardized evaluation suite for sparse autoencoders (SAEs) on language models. It benchmarks 200+ open-source SAEs across eight architectures and training methods on eight metric families spanning reconstruction proxies, interpretability, feature disentanglement, and downstream tasks such as unlearning. The central finding is that gains on common proxy metrics (e.g., reconstruction loss) do not reliably predict practical performance: Matryoshka SAEs underperform on proxies but excel on disentanglement, with advantages that grow at scale. The project ships an interactive comparison interface at neuronpedia.org/sae-bench.

## How it differs from RCG-Bench

| Dimension | SAEBench | RCG-Bench |
|-----------|----------|-----------|
| **Question** | Which SAE architecture/training recipe produces better features? | When a readable explanation names concept X, does intervening on X causally control behavior? |
| **Scope** | SAE quality only | J-lens, AOs, NLAs, SAE descriptions, probes, self-report |
| **Ground truth** | Proxy and task metrics | Known latent variables in controlled tasks |
| **Causal test** | Indirect (unlearning, RAVEL patching) | Direct intervention on reportable vs non-reportable components |
| **Readout type** | Feature auto-interpretability scores, not NL readouts | Natural-language internal-state explanations |

SAEBench evaluates whether SAEs *work as dictionaries*; RCG-Bench evaluates whether *readable readouts identify causal variables*.

## What to borrow

### Metrics
- **Feature disentanglement** (SCR-style): useful as a secondary SAE-quality check, not a substitute for causal validity.
- **Auto-interpretability scoring**: LLM-judged feature descriptions — borrow the *evaluation pipeline*, but pair with intervention (RCG's core gap).
- **RAVEL / sparse probing**: causal patching templates for SAE features on controlled attribute swaps.
- **Absorption / CE-loss**: document that high reconstruction ≠ causal reportability.

### Baselines
- Open SAE suite (Gemma-Scope variants, multiple widths/architectures) as the SAE readout arm.
- Neuronpedia integration pattern for feature lookup and activation examples.

### Pitfalls
- **Do not claim SAE evaluation is novel** — cite SAEBench explicitly.
- Proxy-metric divergence (reconstruction vs. disentanglement) mirrors RCG's expected reportability–causality split; use as analogy, not equivalence.
- SAEBench does not test whether feature *descriptions* match ground-truth latents under intervention.
- Auto-interpretability can score high while features are non-causal for steering (see also Arad et al. 2025, GradSAE).
