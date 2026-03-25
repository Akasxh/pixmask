# Feature Squeezing and Input Transformation Defenses

> Research for pixmask: C++ image sanitization layer before multimodal LLMs.
> Covers foundational papers (2018) through modern VLM-specific work (2024-2025).

---

## Table of Contents

1. [Feature Squeezing (Xu, Evans, Qi — NDSS 2018)](#1-feature-squeezing)
2. [Input Transformations (Guo et al. — ICLR 2018)](#2-input-transformations-guo-et-al)
3. [SHIELD (Das et al. — KDD 2018)](#3-shield)
4. [Randomized Transforms (Xie et al. — ICLR 2018)](#4-randomized-transforms-xie-et-al)
5. [Obfuscated Gradients Critique (Athalye, Carlini, Wagner — ICML 2018)](#5-obfuscated-gradients-critique)
6. [Modern Defenses (2022–2025) Against VLM Attacks](#6-modern-defenses-2022-2025)
7. [Defense Comparison Matrix](#7-defense-comparison-matrix)
8. [Recommendations for pixmask](#8-recommendations-for-pixmask)

---

## 1. Feature Squeezing

**Citation:** Xu, W., Evans, D., Qi, Y. — "Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks." NDSS 2018.
**ArXiv:** https://arxiv.org/abs/1704.01155

### Core Idea

Adversarial perturbations exploit the high-dimensional, fine-grained feature space of DNNs. Feature squeezing reduces this search space by "squeezing" many input vectors that map to different feature vectors in the original space into the same, simplified representation. The defense operates as a **detector**, not a purifier: it compares model output on the original image versus squeezed versions and flags inputs where the outputs diverge beyond a threshold.

### 1.1 Bit-Depth Reduction

**Algorithm:**

```
Input:  pixel value x ∈ [0, 255] (uint8), target bit depth b ∈ {1,...,8}
Output: squeezed pixel value x'

levels = 2^b
x_norm = x / 255.0                    # normalize to [0,1]
x_q    = floor(x_norm * levels) / (levels - 1)   # quantize and rescale
x'     = round(x_q * 255)             # back to uint8
```

Equivalently, in bit-shift terms:
```
shift = 8 - b
x' = (x >> shift) << shift            # zero out the low (8-b) bits
# then optionally map back to full range:
x' = x' | (x' >> b)                   # replicate high bits into low
```

The paper evaluates b ∈ {1, 2, 3, 4, 5} bits per channel. The quantization forces pixels that differ only in low-order bits — as adversarial perturbations typically do — to map to the same value.

**Reported optimal parameters (from paper experiments):**
- MNIST: 1-bit depth reduction most effective
- CIFAR-10 / ImageNet: 3–5 bits; 5-bit is most commonly used because it preserves enough visual quality for the classifier while eliminating perturbations at or below ε = 8/255

**Image quality impact:**
- 5-bit: barely perceptible posterization, PSNR ~38 dB
- 3-bit: moderate posterization, PSNR ~32 dB
- 1-bit: severe, only used on MNIST

### 1.2 Spatial Smoothing

Two variants are studied:

**Median Filter:**
- Window sizes tested: 2×2, 3×3 (local median)
- 2×2 median: low distortion, good detection at high attack strengths
- 3×3 median: higher distortion threshold (~0.039 L∞ gap between clean and squeezed), reduces false negatives but increases false positives on legitimate images
- Applied per-channel independently

**Non-Local Means (NLM):**
- Search window: 11×11 pixels
- Patch size: 3×3 pixels
- Filtering parameter h: tuned per dataset (h ≈ 0.04–0.12 in [0,1] pixel range)
- More computationally expensive than median but better at preserving edges
- Effective at detecting C&W L2 attacks

**Gaussian smoothing** (also tested):
- σ ∈ {0.5, 1.0, 1.5}; kernel size = 2⌈3σ⌉+1
- Less effective than median; smooth gradients allow adaptive adversaries to more easily pass through

### 1.3 Detection Mechanism

```
Input: original image x, squeezers S = {s1, s2, ..., sk}
For each squeezer si:
    Compute f(x) and f(si(x)) where f = softmax output vector
    Compute L∞ distance: d_i = ||f(x) - f(si(x))||_∞
If max(d_i) > threshold T:
    Flag as adversarial
```

**Joint detection (combining squeezers):**
The paper proposes combining multiple squeezers. An input is flagged if **any** squeezer's distance exceeds its individual threshold. Thresholds T_i are set per-squeezer by targeting a desired false positive rate (≈5%) on a clean validation set.

**Reported detection rates (joint, 5% FPR):**
- MNIST (11 attacks): 98.15%
- CIFAR-10: 85%
- ImageNet: 85%

Best combination found: 1-bit depth + 2×2 median smoothing (MNIST); 5-bit + 3×3 median (ImageNet).

### 1.4 How to Choose Squeezers

The paper recommends:
1. Start with bit-depth reduction at 5 bits (conservative) or 3 bits (aggressive)
2. Add median 2×2 as a complementary squeezer (catches different attack geometries)
3. Tune thresholds T on held-out clean validation data for desired FPR
4. Use joint detection (OR across squeezers) for higher recall

### 1.5 What It Defeats

- FGSM (single-step) — detected at near 100% for standard ε values
- BIM/I-FGSM — detected well if ε ≤ 8/255
- JSMA (Jacobian-based saliency) — effective
- DeepFool — effective at ε ≤ 4/255
- C&W L2 — detected with NLM squeezer; partially detected with others
- Black-box transfer attacks — highly effective (no adaptive optimization)

### 1.6 What Bypasses Feature Squeezing

**Adaptive attacks (white-box, adversary knows the squeezer):**

- **Projected Gradient Descent (PGD) with adaptive loss:** By increasing PGD iterations and ε, adversaries craft perturbations that remain small after squeezing. The attack minimizes:
  ```
  L_adaptive = CE(f(x_adv), y_target) + λ * ||f(x_adv) - f(s(x_adv))||_∞
  ```
  This forces squeezed and unsqueezed outputs to match, defeating the detector.

- **EAD (Elastic-net Attack on DNNs):** Bypasses feature squeezing with minimal visual distortion by tuning the κ hyperparameter (confidence parameter), demonstrated in Carlini & Wagner follow-up work (OpenReview, "Bypassing Feature Squeezing").

- **C&W L∞ at large ε:** If the adversary targets ε ≥ 0.1 (L∞), bit-depth reduction is insufficient to remove the perturbation.

- **Obfuscated gradient circumvention:** Smoothing operations are non-differentiable but can be approximated. BPDA (see Section 5) can produce effective adaptive attacks.

**Summary:** Feature squeezing is a detector, not a purifier. A sufficiently strong adaptive adversary who knows the squeezer parameters can always craft perturbations that survive squeezing while remaining adversarial.

### 1.7 Computational Cost

- Bit-depth reduction: O(N) where N = number of pixels; essentially free (~0.1ms for 224×224)
- Median filter 3×3: O(N × w²) with sorting; ~1–3ms for 224×224 in optimized C++
- NLM: O(N × search_window × patch²); 50–200ms for 224×224 (expensive)
- Full joint detection requires 2–3 forward passes through the model, dominating cost

---

## 2. Input Transformations (Guo et al.)

**Citation:** Guo, C., Rana, M., Cissé, M., van der Maaten, L. — "Countering Adversarial Images using Input Transformations." ICLR 2018.
**ArXiv:** https://arxiv.org/abs/1711.00117

### Core Idea

Apply non-differentiable and/or stochastic transformations to inputs at inference time, destroying adversarial perturbations that rely on precise gradient information. The paper studies four transformations and evaluates them against gray-box attacks (adversary knows the classifier but not the transformation) and black-box attacks.

### 2.1 Image Quilting

**Algorithm:**

Image quilting is a non-parametric texture synthesis method (Efros & Freeman, 2001) adapted as a defense:

```
Input:  adversarial image I_adv (H × W × 3)
        patch database D = {p_i} from clean training images
        patch size: typically 5×5 or 8×8 pixels
        overlap: 2–4 pixels (1/4 to 1/3 of patch width)

Algorithm:
1. Tile output image from top-left to bottom-right in raster order
2. For each output tile position t:
   a. Collect all database patches compatible with already-placed neighbors
      (compatibility = sum-of-squared-differences in overlap region ≤ threshold)
   b. Randomly select one compatible patch p* from candidates
   c. Compute minimum-error boundary cut (dynamic programming on overlap error)
   d. Paste p* into output image at position t, using the seam cut
3. Return reconstructed image
```

**Parameters used in paper:**
- Patch size: 5×5 pixels on ImageNet (224×224)
- Overlap: 2 pixels
- Database: subset of ImageNet training images (same class or random)
- Compatibility threshold: percentile of SSD scores (top 25% of patches selected randomly)

**Why it works:** The reconstructed image contains only clean patches drawn from the training distribution. Adversarial perturbations are destroyed because no adversarial patch will be in the clean database.

**Why it's hard to attack:** The patch selection is stochastic (random from compatible set) and the minimum-error cut introduces further randomness. The operation is entirely non-differentiable; BPDA requires a smooth approximation that poorly captures the combinatorial patch-matching.

### 2.2 Total Variation Minimization (TVM)

**ROF Model (Rudin-Osher-Fatemi, 1992):**

```
minimize_u  ||u||_TV + (λ/2) * ||u - f||²_2

where:
  u = denoised output image
  f = input (adversarial) image
  ||u||_TV = sum of |∇u| = total variation (sum of gradient magnitudes)
  λ = regularization weight controlling smoothing vs. fidelity
```

**Chambolle Algorithm (2004) — iterative dual formulation:**

```
Initialize p = 0 (dual variable, same size as image gradient field)
τ = step size = 0.248  (must satisfy τ ≤ 1/(8) for convergence guarantee)
λ_param = 0.052  (original optimal for Gaussian noise; adversarial defense uses λ ≈ 0.03–0.1)

Repeat until convergence (||p^{k+1} - p^k||_∞ < ε = 1e-2):
    g = p^k + τ * div⁻¹(u^k)     # gradient step in dual space
    p^{k+1} = g / max(1, |g|)     # project onto unit ball (pixel-wise)
    u^{k+1} = f - (1/λ) * div(p^{k+1})   # primal update via divergence

where div is the discrete divergence operator (adjoint of gradient).
```

**Parameters for adversarial defense (Guo et al.):**
- λ = 0.03–0.1 (lower = stronger smoothing, more adversarial perturbation removed, more image distortion)
- Iterations: 30–50 (convergence typically reached in 20–40 iterations)
- Applied independently per color channel

**Why it works:** TVM is a classical image denoising method. Adversarial perturbations appear as high-frequency noise; TVM removes high-frequency content while preserving edges and structure.

**Limitation:** TVM is differentiable (after Chambolle approximation) via unrolled iterations, making it susceptible to BPDA-style attacks. Guo et al. note it is the most robust when combined with adversarial training.

### 2.3 JPEG Compression as Defense

**Algorithm:**

Standard JPEG pipeline as defense:
```
1. Convert RGB → YCbCr color space
2. Downsample Cb, Cr channels by 2× (chroma subsampling 4:2:0)
3. Divide into 8×8 blocks
4. Apply 2D DCT to each block
5. Quantize DCT coefficients: Q[u,v] = round(F[u,v] / quant_table[u,v])
   where quant_table is scaled by quality factor Q_factor ∈ [1, 100]
6. Entropy encode (Huffman)
7. Decode: reverse steps 6→1
```

**Quality factor → effective quantization:**
- Q_factor = 75: moderate compression, removes fine-grained perturbations
- Q_factor = 50: strong compression, visible compression artifacts
- Q_factor = 25: aggressive; defeats most ε ≤ 8/255 perturbations but strong artifact distortion

**Guo et al. parameters:** Q_factor = 75 used in main experiments on ImageNet.

**Limitation:** JPEG is approximately differentiable. Shin (2017) and later Athalye demonstrated that JPEG can be approximated with differentiable operations:
```
# Differentiable JPEG approximation:
Replace quantization round() with soft-round or identity (BPDA)
Use Pytorch/TF with custom backward pass approximating JPEG
```
An adaptive attacker using differentiable JPEG achieves 69.1% attack success rate vs. the JPEG defense at ε = 7/255 (691× improvement over non-adaptive baseline).

### 2.4 Image Cropping and Rescaling

**Algorithm:**
```
1. Randomly crop image: select (r, c) uniformly from valid crop region
   Crop size: α × original_size, with α ∈ [0.75, 1.0] (paper uses 0.9)
2. Rescale cropped image back to original size using bilinear interpolation
```

**Effect:** Shifts adversarial perturbation out of alignment. Perturbations crafted at specific pixel coordinates become diluted/shifted after crop+resize.

**Limitation:** Weak against large ε attacks; single-step attacks survive. Pure preprocessing without model retraining shows limited gains.

### 2.5 What Survives Adaptive Attacks (Guo et al.)

- **Image quilting + adversarial training:** Best combination. Gray-box: 60% attack elimination; black-box: 90%.
- **TVM + adversarial training:** Close second.
- **JPEG alone:** Does not survive adaptive white-box attacks.
- **Cropping alone:** Weak defense; does not survive iterative attacks.

The paper notes: defenses based on non-differentiable operations with inherent randomness are the hardest for adaptive adversaries to circumvent.

---

## 3. SHIELD

**Citation:** Das, N., Shanbhogue, M., Chen, S.-T., Hohman, F., Li, S., Chen, L., Kounavis, M.E., Chau, D.H. — "SHIELD: Fast, Practical Defense and Vaccination for Deep Learning using JPEG Compression." KDD 2018.
**ArXiv:** https://arxiv.org/abs/1802.06816
**Project page:** https://poloclub.github.io/jpeg-defense/

### Core Idea

Two-component system:
1. **Stochastic Local Quantization (SLQ):** Randomized JPEG applied to image blocks at test time
2. **Model Vaccination:** Ensemble of models retrained with compressed images at varying quality levels

### 3.1 Stochastic Local Quantization (SLQ)

```
Input:  image I (H × W × C), block_size B, quality_set Q = {q_1, ..., q_k}

Algorithm:
1. Partition image into non-overlapping blocks of size B×B
2. For each block b_i:
   a. Sample quality factor q ~ Uniform(Q)  [or Uniform(q_min, q_max)]
   b. Apply JPEG compression to b_i with quality q
   c. Replace b_i in image with compressed version
3. Return reconstructed image with mixed-quality blocks
```

**Parameters used in paper:**
- Block sizes tested: 8×8, 16×16, 32×32 (JPEG's native 8×8 block is most natural)
- Quality set: {25, 50, 75} or continuous Uniform(25, 75)
- Applied at inference time only (no model changes required for the defense layer)

**Why randomization matters:** An adversary crafting an attack must perturb pixels such that the perturbation survives compression at the quality level applied to that specific block. Since quality is random per-block and unknown at craft time, it is difficult to craft a universal perturbation that survives all possible configurations.

**Why block-level matters:** Different image regions have different adversarial sensitivity. Block-level randomization prevents an adversary from relying on consistent treatment of any spatial region.

### 3.2 Model Vaccination

```
For each quality level q_i ∈ {25, 50, 75, 100}:
    Create compressed training set: {JPEG(x, q_i) : x in train_set}
    Fine-tune model M_i on compressed training set
    (or train from scratch for stronger vaccination)

Ensemble prediction:
    y_pred = argmax( sum_{i} softmax(M_i(JPEG(x, q_i))) )
```

The "vaccination" analogy: the model is exposed to compression artifacts during training so it becomes robust to them (does not confuse compression artifacts with adversarial perturbations).

### 3.3 What It Defeats

- Black-box attacks: up to 94% elimination (C&W L2, DeepFool)
- Gray-box attacks: up to 98% elimination
- FGSM, BIM: effectively neutralized
- Transfer attacks from standard models: high effectiveness

### 3.4 What Bypasses SHIELD

- **White-box adaptive attacks:** An adversary who knows the exact quality factor applied to each block (which is random at test time, so they must optimize over the distribution) can craft perturbations that survive. With enough iterations, PGD with expectation over SLQ approximation can succeed.
- **Large ε attacks** (ε > 16/255): JPEG at quality 25 still preserves some signal; large perturbations survive
- **BPDA on JPEG:** Standard differentiable JPEG approximation can be used; SLQ's randomness adds resilience but doesn't make it immune

### 3.5 Computational Cost

- SLQ: proportional to number of blocks × JPEG encode/decode cost
- Per-image overhead: ~5–20ms for 224×224 with 16×16 blocks (Python; C++ would be ~1–5ms)
- Ensemble inference: k × single model cost (k=4 in paper)

---

## 4. Randomized Transforms (Xie et al.)

**Citation:** Xie, C., Wang, J., Zhang, Z., Ren, Z., Yuille, A. — "Mitigating Adversarial Effects Through Randomization." ICLR 2018.
**ArXiv:** https://arxiv.org/abs/1711.01991

### Core Idea

Apply random resizing and padding at inference time. No model retraining needed. The randomness breaks the tight correspondence between the adversarial perturbation (crafted for a fixed input size/position) and the model's gradient.

### 4.1 Random Resizing + Padding — Exact Parameters

**For Inception-v3 (baseline input: 299×299):**

```
Random Resizing Layer:
  Input shape:  299 × 299 × 3
  Output shape: rnd × rnd × 3
  rnd ~ Uniform_Integer([299, 331))   # i.e., rnd ∈ {299, 300, ..., 330}

Random Padding Layer:
  Input shape:  rnd × rnd × 3
  Output shape: 331 × 331 × 3
  Let pad_total_w = 331 - rnd,  pad_total_h = 331 - rnd
  a ~ Uniform_Integer([0, pad_total_w])   # left pad
  b ~ Uniform_Integer([0, pad_total_h])   # top pad
  Pad: left=a, right=pad_total_w-a, top=b, bottom=pad_total_h-b
  Padding value: 0 (black)

Total distinct patterns: sum_{r=299}^{330} (331-r+1)^2 = 12,528
```

**Why this works:**
- Adversarial perturbations are optimized for specific pixel locations
- After random resize+pad, each pixel lands at a different position in the 331×331 input
- The gradient flow through resize+pad is non-trivially different from the original
- PGD-crafted perturbations at 299×299 have ~30× reduced effectiveness when evaluated at random sizes

### 4.2 Why Non-Differentiability Matters

Bilinear interpolation (used in resizing) is differentiable, so BPDA is not needed here. The defense's power comes from **stochasticity**, not non-differentiability:

- At craft time, the adversary must optimize `E_{rnd}[L(f(resize_pad(x_adv, rnd)), y_target)]`
- This expectation over 12,528 configurations is expensive to optimize exactly
- In practice, the adversary samples a few values of rnd per iteration, which gives noisy gradients
- The resulting attack is weaker than a deterministic-target attack

**For truly adaptive attackers:** Expectation over Transformations (EoT) attack (Athalye 2018) computes gradients by averaging over the random transformation distribution. This directly attacks Xie's defense and substantially reduces its effectiveness. However:
- EoT requires many more forward/backward passes (typically 30–50 per gradient step)
- The defense remains useful as a computational speed bump

### 4.3 What It Defeats

- Single-step attacks (FGSM): near-complete elimination
- Iterative non-adaptive attacks (BIM, MIM, etc.): large reduction
- Transfer attacks: effective (the transformation destroys alignment)
- NIPS 2017 adversarial challenge: ranked #2 among 107 defense teams (normalized score 0.924) when combined with adversarially trained Inception-v3

### 4.4 What Bypasses It

- **EoT (Expectation over Transformations) adaptive attack:** Directly models the randomness; substantially reduces effectiveness with 30+ gradient samples per step
- **L0 and L1 attacks (patch-based):** Concentrated perturbations (patches) are shift-invariant; resizing doesn't destroy them
- **High ε L∞ attacks:** At ε > 16/255, perturbations survive resizing

### 4.5 Computational Cost

- Resize + pad: O(N) bilinear interpolation, essentially free (~0.5ms for 224×224)
- No model retraining, no additional forward passes
- Compatible with any existing model

---

## 5. Obfuscated Gradients Critique

**Citation:** Athalye, A., Carlini, N., Wagner, D. — "Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples." ICML 2018 (Best Paper).
**ArXiv:** https://arxiv.org/abs/1802.00420

### 5.1 Three Types of Obfuscated Gradients

| Type | Mechanism | Example Defenses |
|------|-----------|-----------------|
| **Shattered gradients** | Non-differentiable or discontinuous operations produce NaN/0 gradients | Input transformations with hard quantization |
| **Stochastic gradients** | Randomness in forward pass means gradient of one sample doesn't represent the expected gradient | Random resizing (Xie), SHIELD SLQ |
| **Vanishing/exploding gradients** | Deep unrolled optimization (e.g., iterative purification) creates numerically unstable gradients | TVM with many iterations |

### 5.2 Backward Pass Differentiable Approximation (BPDA)

**Technique for shattered gradients:**
```
Let f_def(x) = g(x) be a non-differentiable defense layer
Find smooth g̃(x) such that g̃(x) ≈ g(x) pointwise

Forward pass:  use g(x) (the real defense)
Backward pass: use ∂g̃/∂x (the smooth approximation's gradient)

Simplest BPDA: g̃(x) = x  (identity approximation)
This means: gradient flows through as if defense weren't there
In practice: identity BPDA works for many defenses because
             the defended and undefended model have similar gradient directions
```

**Technique for stochastic gradients (EoT):**
```
Compute E_{t ~ T}[∇_x L(f(t(x)), y)]
≈ (1/n) * sum_{i=1}^{n} ∇_x L(f(t_i(x)), y)  where t_i ~ T i.i.d.

Requires n forward+backward passes per gradient step (n=30–50 typical)
```

### 5.3 Case Study: ICLR 2018 Defenses

Of 9 defenses evaluated at ICLR 2018:
- **7/9 relied on obfuscated gradients** (giving false sense of security)
- **6/9 were fully broken** by adaptive attacks within the paper's threat model
- **1/9 was partially broken** (thermometer encoding; model without adversarial training broken, with adversarial training partially survived)
- **2/9 survived adaptive attacks:**
  1. **Madry et al. adversarial training (PGD-AT):** The only defense not relying on obfuscated gradients; true robustness via min-max training. Authors could not break it within the stated threat model.
  2. **Thermometer encoding + adversarial training:** Partially survived; combined approach showed genuine (though limited) robustness increase.

### 5.4 Diagnostic Indicators of Obfuscated Gradients

Athalye et al. provide diagnostic tests:
1. **One-step attacks perform better than iterative attacks:** Real robustness should show iterative attacks performing at least as well
2. **Black-box attacks succeed but white-box fails:** Suggests gradient masking, not true robustness
3. **Unbounded perturbations don't reach 100% attack success:** Defense is "hiding" gradients, not truly robust
4. **Random perturbations of the same magnitude succeed:** Model is fragile, not robust

### 5.5 What Makes a Defense Robust to Adaptive Adversaries

Based on Athalye et al. analysis and subsequent work:

1. **True robustness requires adversarial training** (min-max optimization), not just input preprocessing
2. **Certified defenses** (randomized smoothing, interval bound propagation) provide provable guarantees; no adaptive attack can exceed the certified bound
3. **Randomness helps but is not sufficient:** Stochastic defenses need to be robust to EoT attacks
4. **Non-differentiability is not a substitute for robustness:** BPDA circumvents it
5. **Defense must be evaluated under adaptive threat model:** Evaluate against an adversary who knows the defense architecture and parameters

### 5.6 Surviving Defenses and Why

**Adversarial Training (Madry PGD-AT):**
- Trains model to minimize worst-case loss over a PGD attack ball
- No gradient obfuscation: gradients are informative, model is genuinely robust
- Limitation: expensive to train; accuracy-robustness tradeoff; certified radius is small

**Randomized Smoothing (Cohen et al., ICML 2019):**
- Converts any base classifier to a provably robust one by Gaussian noise smoothing
- Certified L2 radius: R = σ * Φ^{-1}(p_A) where p_A = probability of top class under noise
- Cannot be broken by adaptive attacks within the certified radius (by construction)
- On ImageNet: 49% certified accuracy at L2 radius 0.5 (≈127/255)
- Limitation: weak for L∞ attacks; poor at high ε; slow (requires many noisy samples per prediction)

---

## 6. Modern Defenses (2022–2025)

### 6.1 Diffusion-Based Adversarial Purification

**DiffPure (Nie et al., ICML 2022):**
**ArXiv:** https://arxiv.org/abs/2205.07460

```
Algorithm:
1. Forward diffusion: add Gaussian noise to adversarial image x_adv
   x_t = sqrt(ᾱ_t) * x_adv + sqrt(1-ᾱ_t) * ε,  ε ~ N(0,I)
   Use small t* (t* << T) so only adversarial perturbation is destroyed,
   not semantic content
   Typical: t* ∈ [100, 300] out of T=1000 total diffusion steps

2. Reverse denoising: recover clean image via score model
   x_{t*-1}, x_{t*-2}, ..., x_0 = s_θ(x_t*, t*)
   Conditioned or unconditional reverse process

3. Classify purified image: y = f(x_0)
```

**Why it's strong:**
- Diffusion model trained only on clean images; purified output is in-distribution
- Destroys adversarial perturbations because t* adds more noise than the perturbation magnitude
- No assumption on attack type; model-agnostic

**Limitations:**
- Computationally very expensive: 100–1000 NFEs (neural function evaluations) per image
- **Obfuscated gradient concern:** The deep computational graph (hundreds of denoising steps) creates vanishing/exploding gradients. Adaptive attacks exist:
  - **DiffAttack (NeurIPS 2023):** Exploits the gradient flow through diffusion for effective adaptive attacks
  - Adjoint method or truncated BPDA required for accurate gradient computation
- Not certified; empirical robustness only

**VLM applicability:** Direct applicability to pixmask. Pre-purify images with a lightweight diffusion step (t* = 50–100) before passing to VLM. Tradeoff: 100ms+ latency.

### 6.2 Randomized Smoothing for VLMs

**SmoothVLM (arXiv 2405.10529, 2024):**

Applies pixel-wise randomization to defend VLMs against patched visual prompt injectors:

```
Algorithm:
1. Generate N noisy copies: x_i = x + n_i,  n_i ~ N(0, σ²I)  or  mask random pixels
2. Feed each x_i to VLM, collect N responses
3. Majority vote on responses: y_final = mode({y_1, ..., y_N})
```

**Results:** Reduces adversarial patch attack success rate to 0–5% on LLaVA and InstructBLIP, while preserving 67–95% of benign context recovery.

**Key insight:** Patched adversarial prompts are sensitive to pixel-wise randomization while benign content is not. This asymmetry enables detection/mitigation via majority voting.

### 6.3 Multimodal Adversarial Training

**RobustVLM (ICML 2024):**
**GitHub:** https://github.com/chs20/RobustVLM

- Fine-tunes CLIP vision encoder with adversarial examples (unsupervised, no labels needed)
- Replacing the vision encoder of large VLMs with adversarially fine-tuned CLIP yields SOTA robustness
- Does not require training the full VLM; only the encoder changes
- Achieves robust accuracy 2–3× higher than standard VLMs against L∞ attacks at ε = 4/255

**Multimodal Adversarial Training (MAT):**
- Incorporates perturbations in both image and text modalities during training
- Significantly outperforms unimodal defenses
- Aligns adversarial image embeddings with clean text embeddings (contrastive loss)

### 6.4 Input Preprocessing Pipeline for VLMs (2024–2025 State of Art)

Current best practices from multiple papers:

```
Recommended preprocessing stack for VLM sanitization:
1. Coarse detection: run input through feature squeezer (5-bit + 2×2 median)
   → if L∞ divergence > threshold T: flag/reject (fast, ~1ms)
2. Randomized transform: apply random resize+pad (Xie-style)
   → breaks transfer attacks and simple perturbations (~0.5ms)
3. Optional purification: apply lightweight JPEG compression (Q=75) + TVM (λ=0.05, 30 iterations)
   → removes residual high-frequency adversarial content (~10ms)
4. Pass to VLM with adversarially fine-tuned CLIP encoder
```

This layered approach is consistent with the "defense in depth" principle: no single layer provides full robustness, but the combination raises the bar substantially.

### 6.5 Attack Surface Specific to VLMs

VLMs face adversarial threats beyond classification models:

| Attack Type | Description | Relevant Defense |
|-------------|-------------|-----------------|
| **Visual prompt injection** | Adversarial image encodes hidden instructions that override system prompt | SmoothVLM, pixel randomization |
| **Jailbreak via image** | Adversarial image bypasses safety refusals | Adversarial fine-tuning of CLIP encoder |
| **Cross-modal transfer** | Attack transfers from image to text behavior | Multimodal adversarial training |
| **Steganographic embedding** | Hidden data in LSBs carries instructions | Bit-depth reduction (destroys LSBs) |
| **Patch attacks** | Localized visible/invisible adversarial patch | Patch detection + inpainting; SmoothVLM |
| **AnyAttack** | Self-supervised, transfers to GPT-4V, Gemini, Claude | No known reliable preprocessing defense |

---

## 7. Defense Comparison Matrix

| Defense | Algorithm | Defeats | Bypassed By | Comp. Cost | Quality Loss |
|---------|-----------|---------|-------------|------------|--------------|
| **Bit-depth reduction (5-bit)** | Quantize to 32 levels/channel | FGSM, transfer attacks, steganography (LSBs) | Adaptive PGD, EAD at large ε | ~0.1ms | Mild posterization |
| **Median filter (3×3)** | Sort 9 neighbors, take median | FGSM, BIM, some C&W | Adaptive PGD with smooth approx | ~2ms | Mild blur |
| **NLM smoothing** | Non-local patch averaging | C&W L2, JSMA | BPDA + PGD | ~100ms | Moderate texture loss |
| **JPEG compression (Q=75)** | Lossy DCT quantization | FGSM, BIM, low-ε L∞ | Differentiable JPEG + adaptive PGD | ~2ms | Compression artifacts |
| **TVM (λ=0.05)** | Chambolle 30 iterations | L2 attacks, FGSM | Unrolled gradient attack | ~20ms | Edge-preserving smooth |
| **Image quilting** | Stochastic patch synthesis | Most attacks (stochastic) | Weak BPDA + extensive EoT | ~500ms | Moderate texture change |
| **Random resize+pad** | Uniform random size in [299,331) | Non-adaptive iterative attacks, transfers | EoT adaptive attack | ~0.5ms | None |
| **SHIELD SLQ** | Block-level random JPEG | Black/gray-box attacks | White-box EoT | ~5ms | Block artifacts |
| **Feature squeezing (joint)** | Detector: 5-bit + median | 11 attack types (85–98%) | Adaptive dual-objective attacks | 2× forward pass | Detector only |
| **DiffPure** | Forward+reverse diffusion | Model-agnostic, strong | DiffAttack, adjoint BPDA | ~500ms+ | Minimal (t* small) |
| **SmoothVLM** | Pixel randomization + majority vote | Visual prompt injection | High-ε adaptive patch attacks | N × forward pass | Minimal |
| **Adversarial training (PGD-AT)** | Min-max training | All L∞ ε ≤ train_eps | ε > train_eps; L2/L0 attacks outside ball | Training cost | None (model change) |
| **Randomized smoothing** | Gaussian noise + majority vote | Certified L2 robustness | ε > certified radius | N × forward pass | SNR reduction |

---

## 8. Recommendations for pixmask

### What to Implement (in priority order)

**Fast path (target: <5ms total):**
1. **Bit-depth reduction to 5 bits** — near-zero cost, destroys LSB steganography, reduces small ε perturbations. Implement as bitwise shift: `pixel = (pixel >> 3) << 3` (zero lower 3 bits).
2. **Median filter 3×3** — catches FGSM and BIM attacks. Use libopencv or custom kernel; OpenCV's `medianBlur(img, 3)` in C++ is ~1ms for 1MP.
3. **JPEG re-encode at Q=75 + decode** — removes DCT-band adversarial frequencies; use libjpeg-turbo for ~1ms latency.

**Detection path (add ~1ms per squeezer + 1 model forward pass):**
4. **Feature squeezing detector** — compute 5-bit + 3×3 median squeezed images, run model forward pass on each, compare L∞ divergence of softmax outputs. Reject if > threshold T (set on clean validation set for 5% FPR).

**Quality gate for high-security contexts (target: <50ms):**
5. **TVM purification** — implement Chambolle algorithm (30 iterations, λ=0.05); C++ implementation with SIMD can target ~10ms for 224×224.
6. **Random resize+pad** — trivial to implement, free defense against transfer attacks.

### What NOT to Implement

- **Image quilting:** 500ms+ latency; prohibitive for real-time VLM usage
- **DiffPure:** Requires loading a full diffusion model; impractical as a C++ preprocessing layer
- **NLM:** 100ms+ with quality gains not significantly better than median for the adversarial threat model

### Key Design Principles from the Literature

1. **Layer defenses** — no single transform is robust to adaptive attacks; depth matters
2. **Threshold calibration** — set feature squeezer detection thresholds on in-distribution clean data; wrong thresholds create excessive false positives
3. **Non-differentiability alone is not robustness** — BPDA breaks it; rely on randomness (Xie) or genuine adversarial training for provable robustness
4. **The dominant threat for VLMs is visual prompt injection, not misclassification** — standard image classification defenses (bit-depth, TVM) may not address VLM-specific attacks; SmoothVLM-style pixel randomization is more targeted
5. **Test adaptive attacks** — always evaluate with: (a) white-box knowledge of your defense, (b) BPDA for non-differentiable layers, (c) EoT for stochastic layers

### Threat Model Fit for pixmask

```
Likely threats to a VLM preprocessing layer:
  HIGH PRIORITY:
  - Steganographic LSB injection (hidden prompt in image LSBs)
    → Bit-depth reduction (destroys LSBs completely)
  - Transfer attacks (pre-crafted adversarial images)
    → Random resize+pad, JPEG compression
  - Visual prompt injection (adversarial patches with text)
    → Median filter, JPEG, feature squeezing detection

  MEDIUM PRIORITY:
  - Gray-box attacks (adversary knows model, not defense)
    → SHIELD SLQ, random transforms
  - Small ε L∞ perturbations (ε ≤ 8/255)
    → 5-bit reduction + median filter handles well

  LOW PRIORITY (for preprocessing; require model-level defense):
  - White-box adaptive attacks (adversary knows full pipeline)
    → Cannot be solved by preprocessing alone; requires adversarial training
  - ε > 16/255 L∞ attacks (visually obvious distortion)
    → Human review, not automated defense
```

---

## References

1. Xu, W., Evans, D., Qi, Y. (2018). Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks. NDSS 2018. https://arxiv.org/abs/1704.01155

2. Guo, C., Rana, M., Cissé, M., van der Maaten, L. (2018). Countering Adversarial Images using Input Transformations. ICLR 2018. https://arxiv.org/abs/1711.00117

3. Das, N., et al. (2018). SHIELD: Fast, Practical Defense and Vaccination for Deep Learning using JPEG Compression. KDD 2018. https://arxiv.org/abs/1802.06816

4. Xie, C., Wang, J., Zhang, Z., Ren, Z., Yuille, A. (2018). Mitigating Adversarial Effects Through Randomization. ICLR 2018. https://arxiv.org/abs/1711.01991

5. Athalye, A., Carlini, N., Wagner, D. (2018). Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples. ICML 2018. https://arxiv.org/abs/1802.00420

6. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., Vladu, A. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. ICLR 2018. https://arxiv.org/abs/1706.06083

7. Cohen, J., Rosenfeld, E., Kolter, J.Z. (2019). Certified Adversarial Robustness via Randomized Smoothing. ICML 2019. https://arxiv.org/abs/1902.02918

8. Nie, W., et al. (2022). Diffusion Models for Adversarial Purification. ICML 2022. https://arxiv.org/abs/2205.07460

9. Zhang, C., et al. (2024). Safeguarding Vision-Language Models Against Patched Visual Prompt Injectors (SmoothVLM). arXiv 2405.10529. https://arxiv.org/abs/2405.10529

10. Schlarmann, C., et al. (2024). Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models (RobustVLM). ICML 2024. https://github.com/chs20/RobustVLM

11. Chambolle, A. (2004). An Algorithm for Total Variation Minimization and Applications. Journal of Mathematical Imaging and Vision, 20(1-2), 89–97. https://www.ipol.im/pub/art/2013/61/article_lr.pdf

12. Lee, M., et al. (2023). Robust Evaluation of Diffusion-Based Adversarial Purification. ICCV 2023. https://openaccess.thecvf.com/content/ICCV2023/papers/Lee_Robust_Evaluation_of_Diffusion-Based_Adversarial_Purification_ICCV_2023_paper.pdf
