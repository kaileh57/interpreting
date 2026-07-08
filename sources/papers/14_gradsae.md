# Beyond Input Activations: Identifying Influential Latents by Gradient Sparse Autoencoders (GradSAE)

**URL:** https://arxiv.org/abs/2505.08080

**Authors:** Dong Shu, Xuansheng Wu, Haiyan Zhao, Mengnan Du, Ninghao Liu

## Summary

GradSAE addresses a core limitation of SAE analysis: conventional feature ranking uses **input-side activations only**, ignoring causal influence on the output. The paper hypothesizes that (1) activated latents contribute unequally to output construction, and (2) only high-influence latents steer effectively. GradSAE ranks SAE features by incorporating **output-side gradient information** to identify the most influential latents for a target behavior. This is a lightweight alternative to full output-score intervention sweeps (Arad et al. 2025).

## How it differs from RCG-Bench

| Dimension | GradSAE | RCG-Bench |
|-----------|---------|-----------|
| **Goal** | Rank SAE latents by output influence | Benchmark whether *readable* readouts identify causal variables |
| **Methods** | SAE + gradients only | J-lens, AO, NLA, SAE descriptions, probes, self-report |
| **Readability** | Not evaluated — influence is gradient-based | Reportability is a first-class metric |
| **Ground truth** | Steering/improvement on target outputs | Known latent variables in controlled tasks |
| **Gap metric** | Activation rank vs. gradient rank | Reportability − causal effect |

GradSAE improves **which SAE feature to intervene on**; RCG-Bench tests **whether any readable explanation names the causal variable**.

## What to borrow

### Metrics
- **Gradient-based influence scores** for SAE features: use as a non-readable but high-precision causal baseline (causal-only arm).
- Compare gradient-ranked features vs. SAE auto-descriptions vs. AO answers on the same prompts — direct RCG diagnostic.

### Baselines
- GradSAE ranking as the **output-side SAE baseline** alongside input-activation ranking and output-score filtering (Arad et al.).
- Validates the plan.md claim: "activated SAE latents are not equally causal."

### Pitfalls
- GradSAE does not produce human-readable explanations — features it ranks high may be **causal-but-unreadable**.
- Gradient influence ≠ full intervention effect under patching; validate with ablation/patching.
- Method is SAE-specific; does not address J-lens, AO, or self-report gaps.
- Do not conflate gradient influence with reportability calibration.
