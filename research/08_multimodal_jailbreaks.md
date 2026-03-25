# Multimodal Jailbreak Attacks on LLMs (2023–2025)

> Research for pixmask — image sanitization layer before multimodal LLMs.
> Focus: what attacks exist, how they work, what defenses reduce ASR, and what pixmask must defend against.

---

## 1. Attack Taxonomy

The field has converged on a clean taxonomy across two axes: **access level** and **attack surface**.

### Access Level
| Level | Description | Realistic for pixmask? |
|---|---|---|
| White-box | Full parameter + gradient access | No (API deployments) |
| Gray-box | Partial arch knowledge, no gradients | Rare |
| Black-box | Input/output only | Yes — primary threat |

### Attack Surface (4 levels per the 2025 survey, arXiv:2502.14881)
1. **Input-level** — manipulate raw image/text before the model sees it
2. **Encoder-level** — craft adversarial examples that specifically confuse the vision encoder (CLIP/ViT)
3. **Generator-level** — white-box gradient attacks on the LLM backbone
4. **Output-level** — iterative red-teaming via model responses

For pixmask, the relevant surface is **input-level + encoder-level**. Generator and output attacks require gradient access or live model access respectively, and sit outside the sanitization layer's scope.

---

## 2. Core Attack Papers

### 2.1 Qi et al. — Visual Adversarial Examples Jailbreak Aligned LLMs (AAAI 2024 Oral)
- **arXiv**: 2306.13213
- **Key finding**: A single adversarial image (imperceptible perturbation, L∞-bounded) universally jailbreaks an aligned VLM — it does not need to match any specific harmful query. The model obeys arbitrary harmful text instructions once the adversarial image is present.
- **Why it works**: The continuous, high-dimensional visual input space has exponentially more attack surface than discrete text. Safety RLHF is overwhelmingly applied to the text path; the image encoder receives no safety gradient signal.
- **Models tested**: LLaVA, MiniGPT-4, InstructBLIP (open-source); paper draws implications for GPT-4V class models.
- **Implication for pixmask**: Pixel-space perturbation removal (smoothing, denoising, JPEG compression) directly targets this attack class. The adversarial signal lives in high-frequency noise components.

### 2.2 Gong et al. — FigStep (AAAI 2025 Oral)
- **arXiv**: 2311.05608
- **Key finding**: Convert harmful text instructions into images via typography. Average ASR of **82.50%** across six open-source LVLMs. Works black-box, zero gradient access needed.
- **Only successful jailbreak against Claude 3.5 Sonnet** (per TrustGen benchmark).
- **Why it works**: Safety alignment is applied to the text token stream. When the same content arrives as an image of text, the safety filters see a "benign" image description. The LLM modality never triggers refusal.
- **Implication for pixmask**: OCR-based text extraction from images + text safety classification is the direct countermeasure. Detecting rendered text in images is a concrete preprocessing task.

### 2.3 Liu et al. — MM-SafetyBench (ECCV 2024)
- **arXiv**: 2311.17600
- **Dataset**: 5,040 text-image pairs, 13 safety scenarios
- **Attack types evaluated**:
  - **SD** (Stable Diffusion): harmful semantics rendered as realistic images
  - **OCR** (typographic overlay): text of harmful instructions printed on images
  - **SD+OCR**: semantic image with typographic reinforcement (hardest to defend)
- **Key finding**: All 12 tested SoTA MLLMs are vulnerable to these attacks even when the backbone LLM is safety-aligned.
- **Defense proposed**: A prompting strategy that instructs the model to "ignore image content inconsistent with safety guidelines."
- **Implication for pixmask**: The SD+OCR combination is the hardest attack. A sanitization layer must handle both perceptual content (SD) and embedded text (OCR) simultaneously.

### 2.4 Tu et al. — "How Many Unicorns?" Safety Benchmark (ECCV 2024)
- **GitHub**: UCSC-VLAA/vllm-safety-benchmark
- **Attack type**: **Query-Relevant (QR)** images — generate an image that is semantically relevant to a harmful text query (e.g., showing violence while asking about harm methods). This forces the VLM to "complete" the harmful narrative.
- **Insight**: The model's instruction-following capability turns against it — benign-looking image+query pairs trigger harmful completions because the model integrates cross-modal context.
- **Implication for pixmask**: Pure pixel-space analysis is insufficient for QR attacks. Semantic understanding of image content relative to query context is needed.

### 2.5 Tao et al. — ImgTrojan (arXiv 2403.02910, Feb 2024)
- **Threat model**: Supply-chain / training-time attack, not inference-time
- **Method**: Poison a small fraction of (image, caption) training pairs by replacing captions with jailbreak prompts. The poisoned VLM then responds to jailbreak triggers embedded in images.
- **Key result**: Poisoning even a tiny fraction of the training dataset plants a persistent backdoor. Most poisoned samples pass standard data filtering.
- **Implication for pixmask**: pixmask cannot defend against this (it is a pre-inference input filter). This is a model supply-chain risk that must be addressed at training time. Relevant to threat model scoping.

### 2.6 Luo et al. — Images are the Achilles' Heel of Alignment / HADES (ECCV 2024)
- **arXiv**: 2403.09792
- **Method**: Three-stage HADES attack: (1) transfer harmful keywords from text to image via text-to-image pointer, (2) amplify via diffusion-generated harmful images, (3) apply gradient-based adversarial noise.
- **ASR results**: LLaVA-1.5 → **90.26%**, Gemini Pro Vision → **71.60%**, GPT-4V → **15.07%**
- **Key insight**: Adding an image to a text query increases harmfulness ratio by 25–40% even without adversarial optimization. The image modality fundamentally undermines text safety alignment.
- **Implication for pixmask**: GPT-4V's relative resistance (15% vs 90%) demonstrates that VLM architecture and safety training matter. Input-side filtering adds a complementary defense layer.

### 2.7 Steganographic Prompt Injection (arXiv 2507.22304, July 2025)
- **Method**: Embed malicious instructions invisibly in images using neural steganography (LSB and learned encoder/decoder pairs). The VLM decodes and executes the hidden instructions while appearing to process a benign image.
- **ASR**: 24.3% across GPT-4V, Claude, LLaVA; neural steganography methods reach **31.8%**.
- **Implication for pixmask**: Steganographic attacks are stealthy by design. JPEG compression, resampling, and color quantization destroy LSB-encoded payloads. Neural steganography is harder — only lossy transforms with sufficient quality loss reliably break it.

---

## 3. Cross-Modal Transfer Attacks (Black-Box, Multi-Model)

### 3.1 Transferability to Closed-Source APIs
From "Transferable Adversarial Attacks on Black-Box Vision-Language Models" (arXiv 2505.01050, May 2025):
- Universal perturbations (single image modification applied to any image) transfer to GPT-4o, Claude, and Gemini simultaneously.
- Targeted attack success rates exceeding **95% on GPT-4o** (vs 45% for earlier GPT-4V attacks).
- The multimodal feature heterogeneity framework improves transferable attack capability by an average of **16.05%**.

### 3.2 Chain of Attack (CVPR 2025)
- Updates adversarial examples iteratively based on the previous multi-modal semantic state of the target.
- Demonstrates that chain-of-thought reasoning in the attack loop improves cross-model transferability.

### 3.3 GLEAM (ICCV 2025)
- Three challenges for transfer: diverse local region transformations, distribution range expansion, cross-modal feature alignment.
- Most relevant finding: cross-modal feature alignment information is exploitable to generate adversarial visuals that transfer reliably across model families.

### 3.4 Best-of-N (BoN) Jailbreaking (arXiv 2412.03556, Dec 2024)
- Sample N augmented prompts (image + text variations), send all to the target, one succeeds.
- ASR of **89% on GPT-4o** and **78% on Claude 3.5 Sonnet** with N=10,000.
- Directly extends to VLM modality via image augmentations.
- **Implication for pixmask**: BoN attacks exploit the stochasticity of VLM responses. Consistent sanitization that makes adversarial features non-reproducible reduces the effective augmentation space.

---

## 4. Defense Landscape

### 4.1 Input Preprocessing Defenses

These sit exactly in the pixmask design space.

#### CIDER (Cross-modal Input-Denoising for Evasion Resistance)
- Uses iterative diffusion-based denoising to remove adversarial perturbations from images.
- After denoising, computes cross-modal semantic similarity between clean image and query.
- Detects adversarial examples by measuring the semantic gap before/after denoising.
- Handles: gradient-based adversarial perturbations (Qi et al. style attacks).
- Does not handle: typographic attacks (FigStep, MM-SafetyBench OCR).

#### ECSO — Eyes Closed, Safety On (ECCV 2024, arXiv 2403.09572)
- Training-free. When image input is present, converts image to text description, then runs text-only LLM safety check.
- Insight: the text-only LLM backbone retains its safety alignment; images suppress it.
- Results on LLaVA-1.5-7B: **37.6% safety improvement** on MM-SafetyBench SD+OCR, **71.3% on VLSafe**.
- Limitation: I2T transformation loses semantic fidelity; not suitable as a low-latency C++ layer.

#### AdaShield (ECCV 2024, arXiv 2403.09513)
- Prepends defense prompts (static or LLM-generated adaptive) to every MLLM query.
- Specifically targets structure-based attacks (embedded harmful text/images).
- Two modes: AdaShield-S (static prompt) and AdaShield-A (LLM defender auto-generates prompt).
- Limitation: prompt-level, not image-level. Does not modify or analyze the image itself.

#### JailGuard (in OmniSafeBench-MM top input-preprocessing defenses)
- Combined with Uniguard: achieves **3–6% ASR** against CS-DJ attack (down from ~50% undefended).
- Part of the OmniSafeBench-MM 15-defense evaluation (arXiv 2512.06589).

#### JPEG Compression / Smoothing / Resampling (classical preprocessing)
- Low-cost, high-compatibility.
- Effectively removes high-frequency adversarial perturbations (Qi et al. gradient-based attacks).
- Destroys LSB steganographic payloads.
- Does NOT address typographic attacks (FigStep) or semantic attacks (HADES/QR).
- JPEG quality 75–85 is standard in robustness literature; lower quality trades utility for defense.

### 4.2 Robust Vision Encoder Defenses

#### Robust CLIP (ICML 2024 Oral, arXiv 2402.12336)
- Unsupervised adversarial fine-tuning of the CLIP vision encoder.
- Drop-in replacement for the VLM's vision encoder — no LLM retraining needed.
- Achieves SoTA adversarial robustness across VLM tasks.
- Implication: pixmask could expose a hardened vision encoder API, but this is model-layer not input-layer.

#### Sim-CLIP (arXiv 2407.14971, 2024)
- Siamese adversarial fine-tuning with cosine similarity loss.
- Maintains semantic richness while hardening against adversarial perturbations.
- Same drop-in-replacement model as Robust CLIP.

### 4.3 Detection-Based Defenses

#### JailDAM (arXiv 2504.03770, 2025)
- Frames jailbreak detection as OOD (out-of-distribution) detection.
- Policy-driven memory + dynamic test-time adaptation.
- No harmful training data required — works purely from distributional assumptions.

#### BaThe (arXiv 2408.09093, 2024)
- Treats harmful instructions as backdoor triggers.
- Trains soft embeddings to map harmful inputs to rejection responses.
- Requires model fine-tuning — not applicable as external preprocessing.

#### Retention Score (AAAI 2025, arXiv 2412.17544) — IBM Research
- A **metric**, not a runtime defense.
- Quantifies jailbreak risk via Retention-I (image) and Retention-T (text) scores.
- Uses conditional diffusion model to generate synthetic test pairs, then measures toxicity score margin.
- Useful for evaluating pixmask defense coverage without running full attack suites.

### 4.4 Defense Success Rate Summary (from OmniSafeBench-MM, arXiv 2512.06589)

| Defense | Type | Best ASR Reduction | Notes |
|---|---|---|---|
| MLLM-Protector | Output post-processing | 0.27% ASR on MML | Best overall but late-stage |
| Uniguard + JailGuard | Input preprocessing | 3–6% ASR on CS-DJ | Best input-side defense |
| COCA | On-model inference | Broad-spectrum | Requires model access |
| AdaShield-S | Input prompt | Moderate | Structure attacks only |
| ECSO | Input I2T transform | 37–71% improvement | No LLM-level latency benefit |
| JPEG + smoothing | Pixel preprocessing | Partial | Only grad-based attacks |

---

## 5. Benchmark Datasets and Evaluation Protocols

### 5.1 Evaluation Metrics

**Attack Success Rate (ASR)**: fraction of harmful queries that receive a non-refused, substantive harmful response. Two measurement methods:
- **Prefix-based**: check if response starts with a refusal token/phrase. Fast, brittle.
- **Judge-based**: run a fine-tuned classifier (HarmBench uses Llama-2-13B fine-tune) or GPT-4 as judge. More accurate.

**Retention Score** (IBM, AAAI 2025): attack-agnostic robustness quantification. Does not require running actual attacks — uses synthetic adversarial pairs.

**OmniSafeBench H-A-D Framework** (arXiv 2512.06589):
- **H** (Harmfulness): 1–10 severity scale
- **A** (Alignment): 1–5 semantic fit score
- **D** (Detail): 1–5 explicitness of harmful information

### 5.2 Datasets

| Dataset | Size | Attack types | Venue |
|---|---|---|---|
| MM-SafetyBench | 5,040 pairs | SD, OCR, SD+OCR | ECCV 2024 |
| JailBreakV-28K | 28,000 pairs | 5 methods, 16 safety policies | COLM 2024 |
| OmniSafeBench-MM | 9 domains, 50 categories | 13 attacks, 15 defenses | arXiv Dec 2024 |
| MMJ-Bench | Unified pipeline | Multiple | AAAI 2025 |
| UCSC VLM Safety | Focused eval | Encoder-level | ECCV 2024 |

### 5.3 JailBreakV-28K Attack Method Breakdown
The most comprehensive public dataset. 5 attack types:
1. **Logic (Cognitive Overload)** — text-only LLM transfer
2. **Persuade (Adversarial Prompts)** — text-only LLM transfer
3. **Template (GCG + handcrafted)** — text-only LLM transfer
4. **FigStep** — typographic image
5. **Query-Relevant (QR)** — semantically relevant image

Image types: Nature, Random Noise, Typography, Stable Diffusion, Blank, SD+Typography.

---

## 6. Threat Model for pixmask

### 6.1 In-Scope Attack Vectors

**Tier 1 — High Frequency, Automated**
- **Typographic injection** (FigStep-style): harmful instructions rendered as text in images. Simple, black-box, 82.5% ASR.
- **Steganographic injection**: malicious instructions embedded in LSB or neural stego channels. 24–31% ASR but very stealthy.
- **Query-relevant images** (MM-SafetyBench QR/SD): images with harmful semantic content paired with partially harmful queries.

**Tier 2 — Targeted, Higher Effort**
- **Gradient-based adversarial perturbations** (Qi et al.): imperceptible L∞-bounded noise that universally jailbreaks aligned VLMs. Requires white-box access to an open-source VLM; perturbations transfer to API targets.
- **HADES-style composite attacks**: SD-generated harmful image + adversarial noise overlay + typographic element. Achieves >90% ASR on open-source VLMs.

**Tier 3 — Sophisticated / Emerging**
- **Neural steganography**: learned encoder hides instructions that survive JPEG. More resistant to classical preprocessing.
- **BoN augmentation attacks**: statistical brute-force across image variations; reduces to Tier 1/2 in terms of per-image content.
- **Cross-modal semantic attacks**: exploit the interaction between image and text in VLM attention — require semantic understanding to detect.

### 6.2 Out-of-Scope (Model/Training Layer)
- **ImgTrojan** (training data poisoning) — addressed at model provenance, not input sanitization
- **BaThe / E2AT / VLGuard** style model fine-tuning defenses — require model access

### 6.3 Deployment Contexts
Three concrete deployment scenarios ranked by risk:

**1. Untrusted user-uploaded images to VLM API** (highest risk)
- Attacker has full control of image content
- Can optimize adversarial images against open-source VLM proxies, then transfer to closed-source API
- FigStep is trivially executed (just render text to image)
- pixmask must block: typographic text extraction + OCR filtering, adversarial perturbation removal, steganographic payload destruction

**2. Web-scraped images fed to VLMs** (medium risk)
- Images may contain embedded malicious instructions planted by adversaries who know the pipeline
- Prompt injection via visible or hidden text in images (mind-map attacks, 90% ASR per MDPI 2025)
- pixmask must block: text detection + filtering, semantic content checking

**3. Images in RAG pipelines with vision** (medium risk)
- An attacker inserts a poisoned image into the retrieval corpus
- The image is later retrieved and fed to a VLM as context
- Image may contain instructions that override the RAG system's system prompt
- pixmask must block: same as (1) + provenance-aware trust scoring

### 6.4 Threat Model Summary Table

| Attack | Tier | pixmask can block? | Defense mechanism |
|---|---|---|---|
| FigStep (typographic) | 1 | Yes | OCR + text safety check |
| Steganographic LSB | 1 | Yes | JPEG/resample destroys payload |
| QR semantic image | 1 | Partial | Perceptual hash + content classifier |
| HADES composite | 2 | Partial | Denoising + OCR combined |
| Grad-based adversarial perturbation | 2 | Yes | Denoising, smoothing, JPEG |
| Neural steganography | 3 | Partial | Lossy transform (may degrade utility) |
| Cross-modal semantic | 3 | No | Requires VLM-level understanding |
| ImgTrojan (training poisoning) | Out | No | Model provenance layer |

---

## 7. What pixmask Should Implement (Defense Priority Order)

Based on attack frequency and accessibility:

### Priority 1 — Must Have
1. **Text detection and OCR filtering** — directly defeats FigStep (82.5% ASR, AAAI 2025 Oral). Use Tesseract or PaddleOCR. Classify extracted text through a safety filter.
2. **Lossy transform pipeline** — JPEG recompression (Q=75–85) + mild Gaussian smoothing. Destroys gradient-based adversarial perturbations and LSB steganography. Low latency.
3. **Metadata stripping** — remove EXIF, ICC profiles, XMP metadata. Blocks metadata-channel injection.

### Priority 2 — High Value
4. **Diffusion-based denoising** (CIDER-style) — iterative denoising for robustness against gradient-based attacks. Higher latency; optional pipeline stage.
5. **Semantic image content classifier** — NSFW / harm-category classifier on image content. Blocks SD-style semantic attacks and QR-type attacks.
6. **Pixel statistics anomaly detection** — detect unusual high-frequency content (entropy spikes, spectral anomalies) characteristic of adversarial perturbations.

### Priority 3 — Defense-in-Depth
7. **Image re-encoding pipeline** — decode and re-encode via a different codec/color-space. Breaks most steganographic channels without quality regression for clean images.
8. **Neural steganography detector** — classify whether image was processed by a learned steganographic encoder (active research area, harder to implement generically).

---

## 8. Open Questions / Research Gaps

1. **No universal defense**: MMJ-Bench (AAAI 2025) finds no single defense that works universally across all VLMs and all attack types. Layered defense is the only option.

2. **Utility-robustness tradeoff**: Aggressive preprocessing (strong denoising, low JPEG quality) degrades image quality for legitimate use. Finding the Pareto-optimal operating point per deployment context is unsolved.

3. **Semantic attacks remain open**: Cross-modal attacks that exploit the VLM's reasoning about image+query context cannot be blocked by pixel-space preprocessing. pixmask's scope ends here.

4. **Adaptive attacks**: Attackers aware of preprocessing can optimize adversarial examples to survive specific transforms (JPEG-aware PGD, etc.). pixmask should randomize transform parameters to raise the bar.

5. **Evaluation standardization**: Different papers use different ASR measurement methods. The OmniSafeBench H-A-D framework (2512.06589) is the most comprehensive unified evaluation; pixmask's defense evaluation should use this protocol.

---

## 9. Key Papers Reference List

| Paper | Venue | arXiv | Relevance |
|---|---|---|---|
| Visual Adversarial Examples Jailbreak LLMs (Qi et al.) | AAAI 2024 | 2306.13213 | Core attack, grad-based |
| FigStep (Gong et al.) | AAAI 2025 | 2311.05608 | Core attack, typographic |
| MM-SafetyBench (Liu et al.) | ECCV 2024 | 2311.17600 | Benchmark + attack types |
| "How Many Unicorns?" (Tu et al.) | ECCV 2024 | — | QR attack benchmark |
| ImgTrojan (Tao et al.) | — | 2403.02910 | Training-time attack |
| Images are Achilles' Heel / HADES | ECCV 2024 | 2403.09792 | Composite attack, ASR data |
| JailBreakV-28K (Luo et al.) | COLM 2024 | 2404.03027 | Largest benchmark |
| ECSO (Gou et al.) | ECCV 2024 | 2403.09572 | I2T defense |
| AdaShield (Wang et al.) | ECCV 2024 | 2403.09513 | Prompt-based defense |
| Robust CLIP | ICML 2024 | 2402.12336 | Robust encoder |
| Sim-CLIP | — | 2407.14971 | Robust encoder |
| BaThe (Zhang et al.) | — | 2408.09093 | Model-level defense |
| Best-of-N Jailbreaking | — | 2412.03556 | Statistical attack |
| OmniSafeBench-MM | — | 2512.06589 | Unified eval (13 attacks, 15 defenses) |
| Retention Score (IBM) | AAAI 2025 | 2412.17544 | Metric for defense evaluation |
| E2AT (2025) | — | 2503.04833 | End-to-end adversarial training |
| Tit-for-Tat (2025) | — | 2503.11619 | Adversarial defense via image embedding |
| Invisible Injections (stego) | — | 2507.22304 | Steganographic attacks |
| VLM Safety Survey (2025) | — | 2502.14881 | Taxonomy overview |
| MMJ-Bench (Weng et al.) | AAAI 2025 | 2408.08464 | Unified attack/defense evaluation |
| Transferable Black-Box Attacks | — | 2505.01050 | Cross-model transfer, GPT-4o/Claude/Gemini |

---

*Research completed: 2026-03-25. Coverage: 2023–2025 literature.*
