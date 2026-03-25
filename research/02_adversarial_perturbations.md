# Adversarial Perturbation Attacks on Vision-Language Models

> Research for pixmask — a C++ image sanitization layer before multimodal LLMs.
> Written: 2026-03-25

---

## 1. Gradient-Based Attacks on VLMs

### 1.1 Core Attack Methods Adapted for Multimodal Models

**FGSM (Fast Gradient Sign Method)**
Single-step attack. Proven effective at degrading VQA accuracy but weaker than iterative methods. Used mainly as a baseline. FGSM-based perturbations at ε = 8/255 measurably degrade model predictions but are less reliable for targeted attacks.

**PGD (Projected Gradient Descent)**
The primary workhorse for VLM white-box attacks. Standard configuration:
- ε = 8/255 (L-inf), step size α = 1/255–2/255, 10–100 iterations
- PGD-10 used for embedding bank construction; PGD-40 with ε = 16/255 for strong white-box attacks
- Against LLaVA-v1.5-7B, PGD achieves 53.8% attack success rate (ASR)
- Qi et al. (AAAI 2024) run 5000 PGD iterations with batch size 8 to craft universal jailbreak images

**C&W (Carlini-Wagner)**
Optimization-based attack with constraint parameter c = 20 (normal) or c = 100 (strong). Targets the embedding space directly rather than raw classification logits. More compute-intensive than PGD but achieves tighter perturbation budgets at equivalent effectiveness.

**AutoAttack / Auto-PGD**
Ensemble of four attack variants (APGD-CE, APGD-DLR, FAB, Square). Used in robustness benchmarks for VLMs. Architecture and prompt design (EMNLP 2024 Findings) shows simple prompt rephrasing improves resistance to Auto-PGD attacks.

### 1.2 Schlarmann & Hein (ICCVW 2023) — Key Paper

**"On the Adversarial Robustness of Multi-Modal Foundation Models"**
arXiv: 2308.10741 | ICCV 2023 Workshop (AROW)

Core finding: imperceptible perturbations at **ε = 1/255** (extremely small budget) are sufficient to alter caption outputs of multimodal foundation models like Flamingo. Attacks can:
- Redirect users to malicious URLs via manipulated captions
- Broadcast fabricated information as model output
- Operate below human perceptual threshold

Conclusion: any deployed multimodal model requires adversarial countermeasures. This paper directly motivated the RobustCLIP line of work.

### 1.3 Carlini et al. (NeurIPS 2023) — Key Paper

**"Are Aligned Neural Networks Adversarially Aligned?"**
arXiv: 2306.15447 | NeurIPS 2023

Authors: Carlini, Nasr, Choquette-Choo, Jagielski, Gao, Awadalla, Koh, Ippolito, Lee, Trämèr, Schmidt

Core finding: RLHF/safety-aligned text-only LLMs are difficult to attack via NLP-based optimization (token-space attacks are underpowered), but **VLMs are trivially attacked via the visual modality**. Adding an image input creates an exploitable attack surface that bypasses alignment entirely.

Implication: alignment training in the language modality does not transfer to visual robustness. The visual encoder is the weakest link.

### 1.4 Qi et al. (AAAI 2024) — Universal Jailbreak via Visual Adversarials

**"Visual Adversarial Examples Jailbreak Aligned Large Language Models"**
arXiv: 2306.13213 | AAAI 2024 (Oral)

A single adversarial image can universally jailbreak an aligned LLM across a wide range of harmful instructions—far beyond the narrow "few-shot derogatory corpus" used to optimize it. The continuous, high-dimensional visual input makes it a uniquely weak link compared to discrete text tokens. This is a foundational motivation for pixmask: sanitizing the image before it reaches the VLM is the correct interception point.

### 1.5 Image Hijacks (Bailey et al., ICML 2024)

**"Image Hijacks: Adversarial Images can Control Generative Models at Runtime"**
arXiv: 2309.00236

Four attack classes against LLaVA (CLIP + LLaMA-2), all achieving >80% ASR:
1. **Behaviour forcing** — model generates adversary-chosen output
2. **Context extraction** — model leaks context window contents
3. **Safety override** — bypasses safety training
4. **False belief injection** — model asserts fabricated statements

Uses *behaviour-matching* and *prompt-matching* algorithms. Perturbations are imperceptible to humans. Key: attacks are automated and work against current production-grade VLMs.

### 1.6 Transfer: CLIP → Closed APIs

CLIP is the vision encoder for most open VLMs (LLaVA, InstructBLIP, MiniGPT-4) and likely underpins portions of GPT-4V's pipeline. Attack transferability across model families:

- Early CLIP-targeted attacks: ~45% untargeted ASR on GPT-4V (Dong et al. 2023)
- CLIP ensemble attacks can transfer to GPT-4V, Gemini-1.5, Claude-3 at up to 75% ASR for captioner-agent tasks
- Recent methods (2025): >95% targeted ASR on GPT-4o; >90% on GPT-4.5, 4o, o1
- FOA-Attack (Feature Optimal Alignment): improves transfer by aligning both global [CLS] features and local patch tokens — outperforms prior methods on closed-source targets
- Surrogate ensemble strategy: attacking multiple open models jointly is the single biggest factor in improving black-box transfer rates

**Most vulnerable closed models** (by current literature): GPT-4V > Gemini-1.5 > Claude-3 in terms of published transfer ASR. Claude models show lower published transfer rates but data is sparse and model-version-dependent.

---

## 2. Patch-Based Attacks

### 2.1 Adversarial Patches in the Physical World

Adversarial patches are localized, high-magnitude perturbations restricted to a small image region rather than distributed globally. Key properties:

- **Printable**: can be physically placed on objects, clothing, signage
- **Persistent**: survive camera capture, varying lighting
- **Size tradeoff**: larger patches yield stronger suppression; small/distant patches lose efficacy rapidly

Key result: patches printed on T-shirts or car hoods achieve ~100% person-class evasion against YOLOv2/v3 in controlled video when pose and alignment are managed.

### 2.2 Survival Under JPEG Compression and Resizing

JPEG compression is the first instinct for a defense — but patches are specifically trained to survive it:

- Expectation over Transformation (EoT) training: patches optimized over batches of random augmentations including JPEG, rotation, brightness shifts, perspective warp
- Physical-domain gap: up to 64% mAP discrepancy between digital ASR and physical ASR under hue shifts — but patches optimized with EoT close this gap significantly
- JPEG quality 75+ does not reliably destroy patches trained with EoT
- Patches are more robust to compression than distributed (global) perturbations

**Critical implication for pixmask**: JPEG recompression as a standalone defense is insufficient against EoT-trained patches. It must be combined with localized anomaly detection or aggressive spatial perturbation.

### 2.3 Placement Strategies

| Strategy | Description | Effectiveness |
|---|---|---|
| Global patch (anywhere in scene) | Patch placed freely, attacks any nearby target | High digital ASR, drops in physical |
| Local overlap patch | Positioned to overlap target bounding box | Higher physical ASR, requires placement control |
| Clothing/body-worn | T-shirt, hat, badge | ~100% evasion in controlled conditions |
| Adversarial camouflage | Full-object texture | Extends to cross-view robustness |

Cross-modal patches (Wei et al., ICCV 2023): unified patch designed to attack both visual and language encoders simultaneously in VLMs.

### 2.4 VLM-Specific Patch Attacks

**Safeguarding VLMs Against Patched Visual Prompt Injectors** (arXiv:2405.10529):
- Adversarial patches can inject instructions into VLMs via the visual pathway
- SmoothVLM defense applies randomized pixel perturbations to the patch region (mask, swap, replace) and uses majority voting across transformed versions

---

## 3. Transfer Attacks

### 3.1 Mechanism

Transfer attacks craft adversarial examples using a white-box surrogate (open-source model with accessible weights), then apply them to a black-box target (closed API). Effective because:
- VLMs share vision encoder families (ViT-L/14, EVA-CLIP)
- Shared pretraining data creates shared feature representations
- Input processing pipelines are often similar

### 3.2 Attack Success Rates (Published Data)

| Attack Method | Surrogate | Target | ASR |
|---|---|---|---|
| Basic CLIP transfer | CLIP ViT | GPT-4V | ~45% (untargeted) |
| Captioner attack (Bailey 2023) | LLaVA | GPT-4V agent | 75% |
| Ensemble surrogates | Multi-CLIP | GPT-4o | >95% (targeted) |
| FOA-Attack (2025) | CLIP ensemble | Claude-3.5/3.7, GPT-4o/4.1, Gemini-2.0 | State-of-art |
| Simple baseline (2025) | — | GPT-4.5/4o/o1 | >90% |

Black-box scaling law: more diverse surrogate models in the ensemble = higher transfer rate. This is a monotone relationship with no saturation observed at current ensemble sizes.

### 3.3 Which Models Are Most Vulnerable?

1. **Models using standard CLIP/ViT encoders** with no adversarial hardening — maximum vulnerability
2. **GPT-4V / GPT-4o**: highest published ASR in the literature; vision encoder is not publicly disclosed but transfer rates suggest ViT-based architecture
3. **Gemini-1.5**: moderate published transfer rates; multimodal architecture reduces some transferability
4. **Claude-3.x**: lowest published transfer rates in current literature, but data is limited; Claude's image processing pipeline likely applies proprietary preprocessing

Models with adversarially fine-tuned vision encoders (RobustCLIP-based) are significantly harder to attack via transfer.

### 3.4 VLAttack (NeurIPS 2023) — Cross-Modal Transfer

VLAttack attacks five VL pretrained models across six tasks using:
- Block-wise Similarity Attack (BSA) on image modality
- Text perturbation generation independent of image attack
- Iterative Cross-Search Attack (ICSA): image-text pair updates alternate each iteration

Achieves highest ASR across all tasks vs. prior baselines. Transferable to black-box fine-tuned downstream models.

---

## 4. Perturbation Budgets

### 4.1 Standard Epsilon Values

The field has converged on a small set of standard ε values:

| ε (L-inf, normalized 0–1) | ε (pixel scale 0–255) | Usage context |
|---|---|---|
| 1/255 ≈ 0.0039 | 1 | Schlarmann & Hein — extremely tight, still effective |
| 2/255 ≈ 0.0078 | 2 | Lower bound for robustness evaluation |
| 4/255 ≈ 0.0157 | 4 | Sometimes insufficient to disrupt perception |
| **8/255 ≈ 0.0314** | **8** | **De facto standard in VLM attack papers** |
| 16/255 ≈ 0.0627 | 16 | Strong attacks; slightly visible on high-contrast edges |
| 64/255 ≈ 0.251 | 64 | Extreme attacks; clearly visible, used for ablation only |

**8/255 is the community standard** for "imperceptible" adversarial perturbations. At this budget, perturbations are invisible under normal viewing conditions but sufficient to cause near-complete model failure.

### 4.2 L-2 vs. L-inf Norms

- **L-inf (ε = 8/255)**: limits maximum per-pixel change; perturbations spread uniformly across image; preferred for imperceptibility
- **L-2 (ε = 0.5–3.0)**: limits total Euclidean magnitude; allows some pixels to change more; randomized smoothing certification works under L-2

Randomized smoothing provides tight certified robustness under **L-2 norm** (Cohen et al. 2019). L-inf certification is looser and computationally harder.

### 4.3 Imperceptibility Thresholds in Practice

- ε = 4/255: sometimes perceptually marginal — small test set audits find a fraction visible
- ε = 8/255: broadly imperceptible under standard display conditions (sRGB, typical monitor gamma)
- ε = 16/255: SSIM > 0.95 typically, but faint texture artifacts visible on smooth regions at 1:1 zoom
- Physical patches: no L-inf constraint — effectiveness depends on print resolution and viewing distance

**Tradeoff**: larger ε gives attacker more reliable attack success but risks detection by image quality metrics (SSIM, BRISQUE, NIQE). pixmask can exploit this: any image with SSIM < 0.85 vs. a denoised version should be flagged.

---

## 5. Defense Mechanisms

### 5.1 Feature Squeezing (Xu et al., NDSS 2018)

**Core mechanism**: reduce the "degrees of freedom" available to adversarial perturbations by applying squeezers and comparing model predictions on original vs. squeezed input. Large prediction divergence signals an adversarial example.

Two primary squeezers:
1. **Bit-depth reduction**: reduce from 8-bit to 1–5 bits per channel; coalesces nearby pixel values
2. **Spatial smoothing**: local smoothing (median filter 2x2 or 3x3) or non-local means

**Effectiveness against VLM attacks**:
- Effective against FGSM and basic PGD at low ε
- Can be bypassed by adaptive attacks that optimize against the squeezed representation
- At ε = 8/255, bit-depth reduction to 4 bits disrupts ~60% of standard attacks
- Does not provide certified guarantees

**For pixmask**: feature squeezing is computationally cheap (pixel-level operations, no ML inference required) and can serve as a first-pass filter.

### 5.2 Randomized Smoothing (Cohen et al. 2019 / Lecuyer et al. 2019)

**Core mechanism**: classify by majority vote over predictions on copies of the input augmented with Gaussian noise N(0, σ²). Provides a certified L-2 robustness radius:
- r = (σ/2) · Φ⁻¹(p_A) where p_A is the probability of the top class under smoothing
- Typical σ = 0.25 gives certified radius ~0.5 under L-2

**Limitations for VLMs**:
- Certification applies to the base classifier — not the full VLM pipeline
- High σ needed for large certified radius degrades benign accuracy substantially
- L-inf certification via randomized smoothing requires much tighter bounds (Lecuyer et al.)

**MMCert (Wang et al., CVPR 2024)**: extends to multi-modal models via independent subsampling of modalities (0–5% of elements). Outperforms randomized ablation; certifies correct predictions for >40% of test samples even when attackers modify 8 video frames across both visual and audio modalities.

### 5.3 Input Transformation Defenses

Evaluated effectiveness against different attack types:

| Transformation | Effective against | Bypassed by | Notes |
|---|---|---|---|
| JPEG recompression (Q=75) | Basic FGSM/PGD at low ε | EoT-trained attacks | Destroys high-freq perturbations, not low-freq |
| Gaussian blur (σ=1–2) | High-freq perturbations | Low-freq attacks | Fast, minimal quality loss |
| Median filter (3x3) | Salt-and-pepper adversarials | Patch attacks | Robust to localized perturbations |
| Random resizing | Non-adaptive attacks | Adaptive attacks | Must be combined with padding randomization |
| Bit-depth reduction | Basic gradient attacks | Adaptive attacks | 4-bit is effective threshold |
| Total variation minimization | General L-inf perturbations | Smooth adversarials | Computationally heavier |
| Random crops + ensemble | Patch attacks | Large patches | SmoothVLM approach |

**Pixel domain vs. frequency domain**: JPEG compression removes high-frequency adversarial noise. However, adversarial perturbations can be crafted in the low-frequency domain (DCT coefficients) specifically to survive JPEG — this is an active evasion vector.

### 5.4 Diffusion-Based Purification (DiffPure, NeurIPS 2022 / Extensions 2024)

**Core mechanism**: forward-diffuse the adversarial image with small noise (t* timesteps << T), then reverse-denoise back to clean distribution. Adversarial perturbation is a high-frequency signal that gets averaged out.

- DiffPure (Nie et al., ICML 2022): DDPM-based purification; certified via stochastic smoothing
- ADBM (2024): Adversarial Diffusion Bridge Model — directly bridges adversarial distribution to clean, improving over DiffPure's noise-purification tradeoff
- DiffPure-VLM (2024): fine-tuned diffusion denoiser on CLIP image encoder; pairs with VLM noise-augmented safety fine-tuning

**Limitations**:
- Inference cost: 20–50 DDIM steps for acceptable quality; ~200ms–1s per image on A100
- Not viable as a real-time C++ preprocessing layer without distillation/approximation
- Adaptive attacks can be crafted to survive diffusion-based purification (requires strong attack budget)

**For pixmask**: DiffPure is too slow for a synchronous sanitization layer. A distilled single-step denoiser (consistency model or flow matching) could achieve sub-10ms purification. Flag for future research.

### 5.5 RobustCLIP (Schlarmann et al., ICML 2024 Oral)

**"Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models"**
arXiv: 2402.12336 | ICML 2024

**Core contribution**: adversarially fine-tune only the CLIP vision encoder (unsupervised — no labels required, only ImageNet). Drop-in replacement for standard CLIP. Downstream LVLMs (LLaVA, etc.) become robust without any retraining.

- Training: 2 epochs on ImageNet with FARE (Feature Adversarial Robustness Enhancement)
- Result: stealth attacks on LVLM users via manipulated third-party images are no longer effective
- HuggingFace models available

**For pixmask**: this is the most relevant defense architecture. If the target VLM uses a replaceable CLIP encoder, RobustCLIP is the correct upstream fix. However, pixmask operates as a preprocessing layer before the VLM — complementary to RobustCLIP.

### 5.6 MirrorCheck (2024)

arXiv: 2406.09250

Detection-based defense: given an input image, run the VLM to generate a caption, then use a T2I model to generate an image from that caption, then compare input image embedding vs. T2I-generated image embedding. Large divergence = adversarial flag.

- Outperforms baselines adapted from unimodal image classification
- Computationally expensive (requires T2I inference per query)
- Not suitable for synchronous preprocessing

### 5.7 Defense Landscape Summary for pixmask

The following defenses are viable as synchronous C++ preprocessing layers (sub-10ms per image at VGA resolution):

| Defense | Latency class | Certified? | Implementation in C++ |
|---|---|---|---|
| JPEG recompression | ~1ms | No | libjpeg — trivial |
| Bit-depth reduction | ~0.1ms | No | Bitwise ops — trivial |
| Gaussian blur | ~0.5ms | No | OpenCV — trivial |
| Median filter | ~2ms | No | OpenCV — trivial |
| Feature squeezing detector | ~5ms | No | Combination of above |
| Random resizing + padding | ~1ms | No | OpenCV — trivial |
| Total variation minimization | ~20–50ms | No | Requires iterative solver |
| Randomized smoothing (CPU) | ~50–200ms | L-2 only | N forward passes |
| DiffPure (distilled) | ~50–500ms | Partial | Requires ONNX model |

Defenses that require ML inference (DiffPure, MirrorCheck, RobustCLIP) cannot be implemented purely in C++ without model weights — but ONNX/TensorRT inference can achieve acceptable latency with GPU.

---

## 6. Key Papers Reference Table

| Paper | Venue | arXiv | Key Contribution |
|---|---|---|---|
| Schlarmann & Hein 2023 | ICCVW 2023 | 2308.10741 | ε=1/255 sufficient to corrupt VLM captions |
| Carlini et al. 2023 | NeurIPS 2023 | 2306.15447 | Aligned VLMs trivially attacked via vision modality |
| Qi et al. 2024 | AAAI 2024 Oral | 2306.13213 | Single adversarial image universally jailbreaks aligned LLM |
| Bailey et al. 2024 | ICML 2024 | 2309.00236 | Image hijacks: 4 attack types, >80% ASR on LLaVA |
| Gu et al. 2023 | NeurIPS 2023 | 2310.04655 | VLAttack: cross-modal image+text adversarials |
| Cui et al. 2024 | CVPR 2024 | — | Comprehensive LMM robustness benchmark; context mitigates attacks |
| Schlarmann et al. 2024 | ICML 2024 Oral | 2402.12336 | RobustCLIP: unsupervised adversarial encoder fine-tuning |
| Wang et al. 2024 | CVPR 2024 | 2403.19080 | MMCert: provable multimodal defense via independent subsampling |
| Wei et al. 2023 | ICCV 2023 | — | Unified adversarial patch for cross-modal physical attacks |
| Universal UAP for VLP | arXiv 2024 | 2405.05524 | Universal adversarial perturbations for VL pretrained models |

---

## 7. Implications for pixmask Architecture

### Attack Vectors pixmask Must Neutralize

1. **Global L-inf perturbations** (ε ≤ 8/255): distributed across all pixels, imperceptible, gradient-crafted. Primary attack class.
2. **Universal perturbations**: single image-agnostic perturbation that degrades arbitrary inputs. Must be neutralized without knowledge of specific attack.
3. **Adversarial patches**: localized high-magnitude region. Require spatial anomaly detection rather than global filtering.
4. **Low-frequency adversarials**: JPEG-resistant attacks crafted in DCT space. Standard JPEG recompression is insufficient.
5. **Frequency-domain steganographic injections**: adversarial payload embedded in specific frequency bands.

### Defense Priority Stack for pixmask (C++ Layer)

```
Input image
    │
    ├── [FAST, always-on]
    │   ├── Bit-depth quantization (8→6 or 8→5 bits) ← destroys fine-grained gradient signals
    │   ├── JPEG recompression Q=85 ← removes high-freq perturbations
    │   └── Gaussian blur σ=0.5–1.0 ← smooths gradient-crafted textures
    │
    ├── [MEDIUM, statistical detection]
    │   ├── Feature squeezing: compare model-agnostic quality metrics on original vs. squeezed
    │   ├── SSIM / BRISQUE anomaly scoring ← adversarials often have detectably different statistics
    │   └── Patch anomaly detection ← localized high-frequency regions inconsistent with natural images
    │
    └── [SLOW, ML-based, optional]
        └── Distilled denoiser (ONNX) or randomized smoothing ensemble
```

### Critical Gaps / Open Questions

1. **Adaptive attacks always win eventually**: any deterministic preprocessing can be included in the attack optimization loop (EoT). pixmask must include **randomization** (random σ for blur, random JPEG quality within range, random resizing scale) to prevent adaptive attack optimization.

2. **No single defense is sufficient**: the literature consistently shows that individual defenses are bypassed by adaptive attacks. Defense depth (stacking multiple independent transforms) is the current best practice.

3. **Physical patch attacks are qualitatively different**: spatial anomaly detection is needed, not just global frequency filtering. Consider a patch-localization module.

4. **Transfer attack scaling is adversarial**: as more open-source VLMs are released, attackers have larger surrogate ensembles. Transfer ASR to closed models will continue to improve. This is a structural threat that pixmask cannot fully mitigate at the image level — only slow down.

5. **ε = 1/255 is sufficient for caption manipulation** (Schlarmann & Hein): even maximum-conservatism preprocessing (minimal JPEG, minimal blur) must not be defeated by attacks at this tiny budget. Verify this against pixmask's current transform chain.

---

## Sources

- [Schlarmann & Hein 2023 — On the Adversarial Robustness of Multi-Modal Foundation Models (arXiv)](https://arxiv.org/abs/2308.10741)
- [Schlarmann & Hein 2023 — ICCVW Paper PDF](https://openaccess.thecvf.com/content/ICCV2023W/AROW/papers/Schlarmann_On_the_Adversarial_Robustness_of_Multi-Modal_Foundation_Models_ICCVW_2023_paper.pdf)
- [Carlini et al. 2023 — Are Aligned Neural Networks Adversarially Aligned? (arXiv)](https://arxiv.org/abs/2306.15447)
- [Carlini et al. 2023 — NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c1f0b856a35986348ab3414177266f75-Abstract-Conference.html)
- [Qi et al. 2024 — Visual Adversarial Examples Jailbreak Aligned LLMs (arXiv)](https://arxiv.org/abs/2306.13213)
- [Qi et al. 2024 — AAAI publication](https://ojs.aaai.org/index.php/AAAI/article/view/30150)
- [Bailey et al. 2024 — Image Hijacks (arXiv)](https://arxiv.org/abs/2309.00236)
- [Bailey et al. 2024 — Image Hijacks project page](https://image-hijacks.github.io/)
- [Gu et al. 2023 — VLAttack NeurIPS (arXiv)](https://arxiv.org/abs/2310.04655)
- [VLAttack GitHub](https://github.com/ericyinyzy/VLAttack)
- [Cui et al. 2024 — On Robustness of Large Multimodal Models CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Cui_On_the_Robustness_of_Large_Multimodal_Models_Against_Image_Adversarial_CVPR_2024_paper.pdf)
- [Schlarmann et al. 2024 — RobustCLIP ICML (arXiv)](https://arxiv.org/abs/2402.12336)
- [RobustCLIP ICML 2024 proceedings](https://proceedings.mlr.press/v235/schlarmann24a.html)
- [RobustVLM GitHub](https://github.com/chs20/RobustVLM)
- [Wang et al. 2024 — MMCert CVPR (arXiv)](https://arxiv.org/abs/2403.19080)
- [MMCert CVPR paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_MMCert_Provable_Defense_against_Adversarial_Attacks_to_Multi-modal_Models_CVPR_2024_paper.pdf)
- [Universal Adversarial Perturbations for VLP Models (arXiv)](https://arxiv.org/abs/2405.05524)
- [MirrorCheck — Efficient Adversarial Defense for VLMs (arXiv)](https://arxiv.org/abs/2406.09250)
- [Nie et al. 2022 — DiffPure ICML](https://proceedings.mlr.press/v162/nie22a.html)
- [ADBM 2024 — Adversarial Diffusion Bridge Model (arXiv)](https://arxiv.org/abs/2408.00315)
- [Wei et al. 2023 — Unified Adversarial Patch ICCV](https://openaccess.thecvf.com/content/ICCV2023/papers/Wei_Unified_Adversarial_Patch_for_Cross-Modal_Attacks_in_the_Physical_World_ICCV_2023_paper.pdf)
- [Transferable Adversarial Attacks on Black-Box VLMs (arXiv)](https://arxiv.org/abs/2505.01050)
- [FOA-Attack — Adversarial Attacks against Closed-Source MLLMs](https://arxiv.org/html/2505.21494v1)
- [>90% Success Rate Against GPT-4.5/4o/o1 (arXiv)](https://arxiv.org/html/2503.10635v1)
- [Xu et al. 2018 — Feature Squeezing (NDSS)](https://www.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_03A-4_Xu_paper.pdf)
- [Cohen et al. 2019 — Certified Adversarial Robustness via Randomized Smoothing (arXiv)](https://arxiv.org/pdf/1902.02918)
- [Safeguarding VLMs Against Patched Visual Prompt Injectors (arXiv)](https://arxiv.org/html/2405.10529v1)
- [Awesome LVLM Attack — Curated paper list (GitHub)](https://github.com/liudaizong/Awesome-LVLM-Attack)
- [Improving Adversarial Robustness via Architecture and Prompt Design (ACL 2024)](https://aclanthology.org/2024.findings-emnlp.990.pdf)
- [Adversarial Attacks on Multimodal Agents (arXiv)](https://arxiv.org/abs/2406.12814v1)
- [Breaking the Illusion: Real-world Challenges for Adversarial Patches (arXiv)](https://arxiv.org/abs/2410.19863)
