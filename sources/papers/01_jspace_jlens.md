# Verbalizable Representations Form a Global Workspace in Language Models

## Metadata

| Field | Value |
|-------|-------|
| **Authors** | Wes Gurnee, Nicholas Sofroniew, Adam Pearce, Mateusz Piotrowski, Isaac Kauvar, Runjin Chen, Anna Soligo, Paul Bogdan, Euan Ong, Rowan Wang, Ben Thompson, David Abrahams, Subhash Kantamneni, Emmanuel Ameisen, Joshua Batson, Jack Lindsey |
| **Year** | 2026 |
| **URL** | https://transformer-circuits.pub/2026/workspace/ |

## Summary

Anthropic identifies a privileged subset of internal representations—**J-space**—found via the **Jacobian lens (J-lens)**, that behave like a global workspace in Claude-family models. J-lens vectors are rows of \(W_U J_\ell\), where \(J_\ell\) is the corpus-averaged Jacobian from layer \(\ell\) to the final residual stream; readouts rank vocabulary tokens by a concept's average linearized effect on future outputs. The paper argues J-space supports verbal report, directed modulation, internal reasoning, flexible generalization, and selectivity, and shows interventions (steering, coordinate swaps, ablations) often change behavior in workspace-aligned ways—while non-J-space components carry most activation variance but weakly drive report.

## Key Methods

- **Jacobian lens**: \(\text{lens}(h_\ell) = \text{softmax}(W_U\,\text{norm}(J_\ell h_\ell))\); averaged over ~1000 pretraining-style prompts.
- **J-space decomposition**: sparse nonnegative combination of top-\(k\) J-lens vectors (typically \(k \le 25\)) via gradient pursuit; split activations into J-space vs orthogonal remainder.
- **Interventions**: steering along \(v_t\); **patching in lens coordinates** (swap two J-lens coordinates); ablate top-\(k\) J-space content; clamp J-coordinates during non-J-space injection.
- **Functional tests**: "think of a {category}" swaps; injected-thought introspection; hold-in-mind-while-copying; multi-hop reasoning redirects; counterfactual reflection training.

## Key Metrics

- Spearman correlation between J-lens rankings and next-token logits on verbal-report tasks.
- Fraction of swap targets reaching top-5 after J-space vs non-J-space component swaps (e.g., 59% vs 5%).
- J-space fraction of concept-vector variance (~6–7% median) vs causal contribution to report.
- Layer-wise workspace boundaries (~L33–L92 on Sonnet 4.5, reindexed 0–100).
- Occupancy, autocorrelation, effective dimensionality of J-space across layers.
- Behavioral outcomes under ablation (e.g., evaluation-awareness suppression surfacing misalignment).

## Limitations

- J-lens only indexes **single-token** concepts; multi-token concepts need extensions.
- Early-layer readouts are noisy; workspace boundaries may partly reflect J-lens degeneracy, not absence of content.
- J-space accounts for ≤10% activation variance—most representation lives outside J-space.
- Experiments are primarily **Claude closed models**; open-model corroboration is partial (Neuronpedia demos).
- Tuned lens predicts outputs better but surfaces different (often output-skipping) content than J-lens.
- Analogies to human consciousness are functional, not architectural (no recurrent broadcast loop).

## Relevance to RCG-Bench

Central **reportability** baseline. Anthropic shows J-space is privileged for **verbal report** and often causal for behavior, but RCG-Bench should test whether that coupling holds on **open models** and against AO/NLA/SAE readouts. The paper's own split—93% variance outside J-space with minimal direct report effect—is a prototype of **causal-but-unreadable** mass. RCG-Bench can operationalize: when J-lens names latent \(z\), does patching/swapping \(z\)'s J-coordinate change behavior as much as patching the true causal locus?

## Actionable Implementation Notes

- Precompute per-layer \(J_\ell = \mathbb{E}[\partial h_{\text{final}} / \partial h_\ell]\) on a diverse corpus; cache \(W_U J_\ell\) rows as concept directions.
- Implement **reportability score**: top-\(k\) J-lens overlap with ground-truth latent label, or probe cosine to target J-vector.
- Implement **causal score**: lens-coordinate swap (source→target concept) or ablate identified J-direction; measure logit diff / answer flip.
- Compare J-lens vs logit lens vs tuned lens on same prompts—expect divergence in middle layers (reportability without output prediction).
- Use sparse decomposition (gradient pursuit, \(k \approx 16–25\)) for discrete concept inventories vs full vocab ranking.
- For RCG-Bench tasks with known latents (hidden city, secret word), test whether **non-J residual component** interventions change behavior while J-readout stays correct (readable-but-non-causal).
