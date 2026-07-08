# MIB: A Mechanistic Interpretability Benchmark

## Metadata

| Field | Value |
|-------|-------|
| **Authors** | Aaron Mueller, Atticus Geiger, Sarah Wiegreffe, Dana Arad, Iván Arcuschin, Adam Belfki, Yung-Kuan Chan, Jaden Fiotto-Kaufman, Tal Haklay, Michael G. Hanna, Jing Huang, Rohan Gupta, Yaniv Nikankin, Hadas Orgad, N. Prakash, Anja Reusch, A. Sankaranarayanan, Shun Shao, Alessandro Stolfo, Martin Tutek, Amir Zur, David Bau, Yonatan Belinkov |
| **Year** | 2025 |
| **URL** | https://arxiv.org/abs/2504.13151 |
| **Artifacts** | HuggingFace datasets, GitHub code, public leaderboards |

## Summary

MIB standardizes comparison of mechanistic interpretability methods across **two tracks**: (1) **circuit localization**—find minimal causal subgraphs of components/edges; (2) **causal variable localization**—featurize hidden vectors (neurons, PCA, SAE, DAS) and align features to high-level causal variables via interchange interventions. Four tasks (IOI, Arithmetic, MCQA, ARC-Challenge), five models (GPT-2 Small, Qwen-2.5 0.5B, Gemma-2 2B, Llama-3.1 8B), fixed counterfactual pairs, and public/private test splits. Key finding: attribution/mask methods win circuit track; **supervised DAS** wins causal-variable track; **SAE features do not beat raw neurons**.

## Key Methods

- **Counterfactual activation patching**: fix components to values from paired counterfactual inputs (MIB-provided mappings).
- **Circuit track**: score edges/nodes by indirect effect, attribution patching (AP/EAP), mask optimization (ACDC, EAP-IG); metrics **CPR** (performance recovery) and **CMD** (any-effect recovery) integrated over circuit size.
- **Causal variable track**: high-level causal models per task; **interchange intervention accuracy (IIA)** via distributed interchange on feature subsets \(\Pi_X\).
- **Featurizers**: identity (full vector), DBM on neurons/PCA/SAE, supervised **DAS** (learn orthogonal subspace aligned to causal variable).
- **Leaderboard API**: submit circuits/featurizers for private test evaluation.

## Key Metrics

- **CPR / CMD**: area under/over faithfulness curve vs weighted edge count (circuit size).
- **IIA (interchange intervention accuracy)**: does fixing aligned features reproduce high-level causal model intervention outcomes?
- **AUROC** on ground-truth IOI circuit edges (InterpBench-style model).
- Task accuracy on base and counterfactual inputs (filter to model-correct examples for causal-variable track).
- Baseline rankings: EAP-IG+CF best circuits; DAS best variables; SAE worst (≈ neurons).

## Limitations

- No **natural-language readouts** (J-lens, AO, NLA)—featurization is geometric/subspace-based.
- Causal variables require **supervised high-level models**—not applicable to arbitrary NL explanations.
- Fixed counterfactuals define what "causal" means; different corruptions trace different mechanisms.
- SAE evaluation may suffer low-data PCA regimes on small MCQA splits.
- Models/tasks will age; leaderboard is living artifact but not exhaustive.

## Relevance to RCG-Bench

MIB is the **positioning baseline** for causal localization without readability. RCG-Bench asks the complementary question: when methods produce **human-readable** latent claims, do they localize the same variables MIB/DAS would? Use MIB's counterfactual pairs and IIA-style interventions as **ground-truth causal effect** while scoring J-lens/AO/NLA **reportability**. Cite MIB finding that readable SAE labels ≠ better causal features as precedent for reportability–causality decoupling.

## Actionable Implementation Notes

- Reuse MIB **counterfactual prompt pairs** and logit-difference metric for causal effect scores in RCG-Bench.
- On shared tasks (IOI, arithmetic, MCQA), compare: DAS-aligned feature intervention effect vs AO/J-lens-identified concept intervention effect on same \((b, c)\) pairs.
- Implement CPR/CMD-style curves if benchmarking circuit-level readouts (e.g., which heads carry AO-verbalized concepts).
- Treat **DAS IIA** as upper bound on causal localization for known latents; measure gap to NL readout causal precision@k.
- Do not claim novelty over MIB on pure causal localization—frame RCG-Bench as NL **reportability** + MIB-style **causality** in one benchmark.
- HuggingFace MIB datasets provide standardized train/val/public-test splits for reproducible RCG task shells.
