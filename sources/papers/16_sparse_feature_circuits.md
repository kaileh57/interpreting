# Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models

**URL:** https://arxiv.org/abs/2403.19647

**Authors:** Samuel D. Marks, Can Rager, Eric J. Michaud, Yonatan Belinkov, David Bau, Aaron Mueller

## Summary

Introduces **sparse feature circuits**: causally implicated subnetworks of SAE-interpretable features, discovered via attribution patching on sparse features rather than polysemantic neurons or attention heads. The pipeline automatically finds circuits for thousands of model behaviors (via behavior clustering). **SHIFT** (Sparse Human-Interpretable Feature Trimming) ablates human-judged task-irrelevant features to improve classifier generalization — demonstrating that interpretable feature graphs are actionable for editing. Published at ICLR 2025 (Oral). Code: github.com/saprmarks/feature-circuits.

## How it differs from RCG-Bench

| Dimension | Sparse Feature Circuits | RCG-Bench |
|-----------|----------------------|-----------|
| **Unit of analysis** | Circuits of SAE features | Single readout → intervention on named variable |
| **Causality test** | Attribution patching within circuit | Direct intervention on readout-identified component |
| **Readability** | SAE feature labels (auto or human) | NL readouts across multiple families |
| **Goal** | Discover mechanism for a behavior | Measure reportability–causality gap |
| **Ground truth** | Behavior label (agreement, etc.) | Known latent variables |

Sparse feature circuits find *which features participate in a circuit*; RCG-Bench tests *whether a readable explanation of internal state names the variable that controls behavior*.

## What to borrow

### Metrics
- **Attribution patching on SAE features**: causal validation template for graph nodes.
- **Circuit faithfulness metrics** (patching edge ablations): adapt for testing whether readout-named features lie on causal paths.
- **SHIFT ablation protocol**: human-in-the-loop feature relevance — compare to automated RCG causal precision@k.

### Baselines
- Sparse feature circuit discovery as a **graph-based causal readout** baseline (plan.md item 11).
- Automated behavior-cluster pipeline for scaling task diversity.

### Pitfalls
- Circuits are per-behavior, not per-latent-variable — mapping to RCG ground-truth tasks requires care.
- Feature auto-descriptions may label circuit nodes that are associative but not individually causal (see May 2026 downstream-connections update).
- Per-layer SAE graphs miss cross-layer structure that CLT attribution graphs capture.
- Do not claim circuit discovery is novel — cite as mechanistic baseline, position RCG as readout-validity layer.
