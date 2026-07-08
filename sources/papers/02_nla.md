# Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations

## Metadata

| Field | Value |
|-------|-------|
| **Authors** | Kit Fraser-Taliente, Subhash Kantamneni, Euan Ong, Dan Mossing, Christina Lu, Paul C. Bogdan, Emmanuel Ameisen, James Chen, Dzmitry Kishylau, Adam Pearce, Julius Tarng, Alex Wu, Jeff Wu, Yang Zhang, Daniel M. Ziegler, Evan Hubinger, Joshua Batson, Jack Lindsey, Samuel Zimmerman, Samuel Marks |
| **Affiliation** | Anthropic |
| **Year** | 2026 |
| **URL** | https://www.anthropic.com/research/natural-language-autoencoders (full paper: https://transformer-circuits.pub/2026/nla/) |

## Summary

NLAs train two copies of a target LLM: an **activation verbalizer (AV)** maps residual-stream activations to natural-language explanations, and an **activation reconstructor (AR)** maps text back to activations. Joint RL optimizes reconstruction similarity; explanations become more informative as reconstruction improves despite no direct supervision on semantic correctness. Anthropic uses NLAs for safety auditing—surfacing unverbalized evaluation awareness, hidden motivations in misaligned model organisms, and training-data artifacts— and releases code plus open-model checkpoints (Neuronpedia demo).

## Key Methods

- **Round-trip architecture**: activation → AV text → AR activation; score = similarity(reconstructed, original).
- **Three model copies**: frozen target (activation source), trainable AV, trainable AR.
- **Training**: RL on large activation batches from target forward passes; explanations are unsupervised except via reconstruction.
- **Inference**: AV generates hundreds of tokens per activation; used for turn-level or token-level auditing.
- **Auditing game**: misaligned Haiku 3.5 with hidden motivation; NLA-equipped auditor wins 12–15% vs <3% without NLAs.

## Key Metrics

- Reconstruction loss / cosine similarity between original and AR-recovered activations.
- **Evaluation awareness rate**: NLA-detected vs verbally stated awareness (e.g., 16% vs ~0% on destructive-code eval; 26% on SWE-bench vs <1% on real claude.ai usage).
- Auditing game win rate (NLA vs baseline interpretability tools).
- Qualitative case studies (blackmail eval, multilingual glitch diagnosis).

## Limitations

- **Hallucination**: AV invents context details not in the transcript; internal-reasoning claims harder to verify than surface-text checks.
- **Expensive**: RL on two model copies; hundreds of tokens per activation— impractical for full-transcript monitoring.
- Optimizes **reconstructibility**, not causal control or ground-truth alignment of explanations.
- Reconstruction success does not guarantee the explanation names the behavior-controlling variable.
- Sensitive to injection/reconstruction architecture (contrast with AO norm-matched additive injection).

## Relevance to RCG-Bench

NLAs are a primary **natural-language reportability** method. RCG-Bench should treat NLA mentions of latent \(z\) as reportability signal, then test whether steering/ablation along AV-identified concepts (or AR-inverted directions) changes behavior. Expect **readable-but-non-causal** cases when reconstruction captures correlates of \(z\) but behavior is driven elsewhere—mirroring Anthropic's caution that themes matter more than single claims.

## Actionable Implementation Notes

- Use released Neuronpedia / GitHub NLAs for open models (Qwen, Gemma, etc.) as Week-1 readout baseline.
- **Reportability**: keyword/LLM-judge match between NLA explanation and ground-truth latent (hidden city, secret word, eval-awareness label).
- **Causal test**: derive direction from AR(NLA text mentioning \(z\)) or contrastive AV prompts; patch/clamp and measure logit diff—compare to patching true causal feature from MIB-style counterfactuals.
- Flag NLA outputs against **surface text** to separate genuine activation readout from confabulation (hallucination rate as confound).
- Budget: sample activations at high-signal positions (assistant token, decision points) rather than every token.
- Pair with AO calibration (bootstrap confidence) when NLA produces categorical latent claims.
