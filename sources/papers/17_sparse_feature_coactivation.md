# Sparse Feature Coactivation Reveals Causal Semantic Modules in Large Language Models

**URL:** https://arxiv.org/abs/2506.18141

**Authors:** Ruixue Deng, Xiaoyang Hu, Miles Gilberti, Shane Storks, Aman Taxali, Mike Angstadt, Chandra Sripada, Joyce Chai

## Summary

Identifies **semantically coherent SAE feature modules** via coactivation graphs built from a handful of prompts, without full cross-layer transcoders. On concept–relation tasks (country facts, word relations, translations), connected components of coactivating features are **consistent across contexts**. Ablating/amplifying modules predictably steers outputs (up to 98% success for single concepts/relations; 90% for compounds). Concept modules emerge in early layers; abstract relation modules in later layers. Composing concept + relation modules yields compound counterfactual outputs (e.g., Nigeria + language → "English"). Uses Gemma-2 2B/9B with Gemma-Scope SAEs.

## How it differs from RCG-Bench

| Dimension | Coactivation Modules | RCG-Bench |
|-----------|---------------------|-----------|
| **Discovery** | Unsupervised coactivation clustering | Readout methods name concepts from activations |
| **Validation** | Module ablation/amplification steering | Per-readout reportability vs. intervention gap |
| **Granularity** | Feature groups (modules) | Individual named concepts + NL explanations |
| **Tasks** | Relational knowledge (capitals, languages) | Broader latent-state tasks incl. adversarial prefill |
| **Readability** | Implicit via SAE features | Explicit NL readouts (AO, NLA, J-lens) |

Coactivation modules show that **grouped SAE features can be causal**; RCG-Bench asks whether **readable explanations identify those groups or misleading singles**.

## What to borrow

### Metrics
- **Counterfactual steering success rate** on concept–relation tasks — overlaps with RCG latent-state tasks (movie→director→country, etc.).
- **KL divergence shift** under ablation as causal effect proxy.
- **Layer emergence analysis** (concept early, relation late) for interpreting RCG failures by depth.

### Baselines
- Coactivation modules as **module-level intervention baseline** when SAE descriptions name individual features but the true causal unit is a module.
- Task templates: country capitals, translation language, word relations.

### Pitfalls
- Module discovery uses few prompts — may not generalize to RCG's diverse failure modes.
- Coactivation ≠ readable: modules are discovered structurally, not via NL explanations.
- High steering success does not mean a *readout* would name the right module.
- Density pruning (τ=0.01) may drop features that readouts highlight but that are non-causal alone.
- plan.md lists items 16 and 17 separately; item 16 is Sparse Feature Circuits (2403.19647), item 17 is this coactivation paper.
