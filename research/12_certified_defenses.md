# Certified and Provable Defenses for Image Preprocessing

> Research for pixmask — C++ image sanitization layer before multimodal LLMs.
> Written: 2026-03-25

---

## Table of Contents

1. [Randomized Smoothing (Cohen et al., 2019)](#1-randomized-smoothing)
2. [Certified Preprocessing Defenses](#2-certified-preprocessing-defenses)
3. [Ensemble of Transforms](#3-ensemble-of-transforms)
4. [Adaptive Attack Resistance](#4-adaptive-attack-resistance)
5. [The Non-Differentiability Argument](#5-the-non-differentiability-argument)
6. [Implications for pixmask](#6-implications-for-pixmask)

---

## 1. Randomized Smoothing

**Citation:** Cohen, J., Rosenfeld, E., Kolter, J.Z. — "Certified Adversarial Robustness via Randomized Smoothing." ICML 2019.
**ArXiv:** https://arxiv.org/abs/1902.02918
**Code:** https://github.com/locuslab/smoothing

### 1.1 Core Theorem

Given a base classifier `f` and noise level `σ`, define the **smoothed classifier** `g`:

```
g(x) = argmax_c  P_{δ ~ N(0, σ²I)}[ f(x + δ) = c ]
```

**Theorem (Cohen et al.):** If `g` returns class `c_A` with probability `p_A` under noise, and the runner-up class has probability at most `p_B`, then `g` is certifiably robust within L2 radius:

```
r = (σ / 2) * (Φ⁻¹(p_A) - Φ⁻¹(p_B))
```

The simplified version when `p_A > 0.5`:

```
r = σ * Φ⁻¹(p_A)
```

where `Φ⁻¹` is the inverse CDF of the standard normal. This is the **tight** bound — no prior work had proven this was optimal. The guarantee: for any adversarial perturbation `‖δ‖₂ < r`, the smoothed classifier always returns `c_A`.

### 1.2 Certification Algorithm

Two-phase Monte Carlo procedure:

**Phase 1 — Selection (n₀ samples):**
- Sample `n₀` noisy copies: `x + δᵢ` where `δᵢ ~ N(0, σ²I)`
- Run base classifier on each
- Identify top predicted class `ĉ_A` by majority vote

**Phase 2 — Estimation (n samples):**
- Sample `n` more noisy copies independently
- Count fraction `k` that predict `ĉ_A`
- Compute lower confidence bound `p̄_A = BinomLB(k, n, α)`
  - `BinomLB` = one-sided Clopper-Pearson lower bound at confidence `1 - α`
- If `p̄_A > 0.5`: return `(ĉ_A, σ * Φ⁻¹(p̄_A))`
- Else: **abstain** (cannot certify)

**Reference parameters from the official implementation** (`certify.py`):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `n0`      | 100     | Forward passes for class selection |
| `n`       | 100,000 | Forward passes for radius certification |
| `alpha`   | 0.001   | Failure probability (99.9% confidence) |

### 1.3 Computational Cost

| Operation | Forward Passes | Time (ImageNet, A100-equivalent) |
|-----------|---------------|----------------------------------|
| Prediction only | ~1,000 (n₀ + small n) | ~1.5 seconds |
| Full certification | 100 + 100,000 | ~150 seconds |
| Ratio | ~100x overhead | per image |

**Conclusion:** Certification is prohibitively slow for real-time pipelines. 150 seconds per image rules out live inference before a VLM. Prediction-only mode (1,000 samples) is ~10x overhead, still expensive but potentially usable in offline/batch contexts.

**Key ImageNet result:** At `σ = 0.5`, `49%` certified top-1 accuracy under L2 perturbations of norm ≤ 0.5 (≈127/255 per channel). This is the **only certified defense demonstrated to scale to ImageNet** at the time of publication.

### 1.4 Randomized Smoothing as Preprocessing

Randomized smoothing is architecturally compatible with a preprocessing layer:

```
image → [add Gaussian noise N(0, σ²)] → base_classifier(noisy_image)
```

This is **model-agnostic** — the same noise injection wraps any classifier without retraining. The base classifier just needs to classify well under Gaussian noise (achievable via noise augmentation during training, or via adversarial training as in Salman et al. 2019).

**Critical nuance:** The certification guarantee is a statement about the *smoothed classifier*, not a statement about each individual noisy forward pass. The smoothed prediction requires aggregating `n` passes and taking a majority vote. A single noisy pass provides **no certified guarantee** — it just reduces attack success empirically.

For pixmask's use case (sanitization before inference), this translates to:
- **Empirical mode:** Add one instance of `N(0, σ²)` noise to each image. No certification, but gradient-based attacks targeting `f` directly are disrupted.
- **Certified mode:** Run `n = 100,000` passes per image and aggregate. Provides L2 certificate but at 150s/image — only viable for asynchronous content moderation, not real-time VLM serving.

### 1.5 Accuracy-Robustness Tradeoff

Higher `σ` → larger certified radius but lower clean accuracy (more noise destroys semantic content). Typical operating points:

| σ | Clean accuracy (ImageNet) | Certified accuracy at r ≤ 0.5 |
|---|--------------------------|-------------------------------|
| 0.12 | ~67% | ~37% |
| 0.25 | ~61% | ~44% |
| 0.50 | ~53% | ~49% |
| 1.00 | ~40% | ~37% |

There is no free lunch: gains in certified radius come at direct cost to clean performance.

---

## 2. Certified Preprocessing Defenses

### 2.1 General State of Certified Input-Transformation Defenses

The honest answer: **provable guarantees for arbitrary preprocessing pipelines do not exist in the general case.** Certification requires that the defense be differentiable, or that its effect on the classifier output can be bounded analytically. Three approaches exist:

**Approach A: Certify the composition (smoothing + preprocessing)**
Apply randomized smoothing to the *composed* function `f ∘ g` where `g` is the preprocessor. The smoothed classifier becomes:
```
g_smooth(x) = argmax_c P_{δ ~ N(0, σ²I)}[ f(g(x + δ)) = c ]
```
The Cohen et al. theorem still applies verbatim — the base classifier is now `f ∘ g`. Certification cost is identical. The tradeoff: if `g` destroys information (lossy JPEG, quantization), the noisy inputs to `f` carry even less signal, and clean accuracy drops further.

**Approach B: Certify that the preprocessor bounds input perturbations**
If `g` is a Lipschitz-L contraction (‖g(x) - g(x')‖₂ ≤ L·‖x - x'‖₂), then any certified guarantee on `f` with radius `r` implies a guarantee on `f ∘ g` with input radius `r / L`. Most useful transforms (Gaussian blur: L ≤ 1, bit-depth reduction: L ≤ 1, linear rescaling: L = scale_factor) are contractive and reduce the perturbation budget the attacker can use.

**Approach C: Non-constructive empirical certification**
For stochastic, non-differentiable preprocessors, use the empirical CERTIFY procedure treating the full pipeline as a black-box stochastic function. This is valid — the Clopper-Pearson bound does not require the base classifier to be differentiable. It requires only the ability to run many Monte Carlo samples, which is always possible.

### 2.2 DiffPure (Nie et al., 2022)

**Citation:** Nie, W., Guo, B., Huang, Y., Xiao, C., Vahdat, A., Anandkumar, A. — "Diffusion Models for Adversarial Purification." ICML 2022.
**ArXiv:** https://arxiv.org/abs/2205.07460

#### Method

DiffPure is a **stochastic preprocessing purifier** built on score-based diffusion models:

1. **Forward diffusion (noise injection):** Given adversarial input `x_adv`, run the stochastic differential equation (SDE) forward for `t*` steps. At `t*`, the image is `x_{t*} ≈ x_adv + N(0, t*·σ²)`. Small `t*` (typically `t* = 0.1` to `t* = 0.3` as fraction of total diffusion time) adds enough noise to disrupt the adversarial perturbation while not completely destroying semantic structure.

2. **Reverse diffusion (denoising):** Run the reverse SDE (score function estimated by a pretrained diffusion model) from `x_{t*}` back to `t = 0`. The result `x̃` is a clean reconstruction that follows the natural image manifold.

3. **Classification:** Pass `x̃` through any classifier `f`.

The key insight: adversarial perturbations live in a specific structured direction in pixel space, but the reverse SDE reconstructs an image from the *generative model's prior*, which has no knowledge of the adversarial direction.

#### Certified Robustness via DiffPure + Randomized Smoothing

To obtain certificates, DiffPure is **composed with randomized smoothing**: the stochastic reverse SDE itself provides randomness, so DiffPure can serve as the noise injection mechanism replacing `N(0, σ²)`. The Cohen et al. certification procedure then applies to the composed `f ∘ DiffPure`.

The adjoint method enables efficient gradient computation through the reverse SDE ODE, which is critical for evaluating under adaptive attacks (white-box adversaries that compute gradients through DiffPure). Without adjoint, memory cost would be O(N) in the number of function evaluations N; adjoint reduces this to O(1) memory.

#### Computational Cost

DiffPure requires running the reverse diffusion from `t*` back to 0, which involves `~O(N)` function evaluations of the score network (N ≈ 100-1000 steps depending on the solver). This is orders of magnitude more expensive than a single classifier forward pass.

| Method | Inference cost (relative) | Certified? |
|--------|--------------------------|------------|
| Raw classifier | 1× | No |
| Gaussian noise + smoothing | ~1000× (predict) / ~100,000× (certify) | Yes (L2) |
| DiffPure (prediction only) | ~100–1000× | No (empirical only) |
| DiffPure + RS certification | ~100,000× × diffusion cost | Yes (L2), with caveat |

**Caveat on DiffPure certification:** The standard randomized smoothing proof assumes the smoothing distribution is `N(0, σ²I)`. DiffPure's stochastic reverse SDE does not follow this distribution exactly. Certified guarantees require careful treatment and are approximate unless the forward noise added is explicitly `N(0, σ²I)`.

#### Practicality for pixmask

DiffPure is not viable for real-time preprocessing before a VLM. It is relevant as an **offline, high-assurance sanitizer** for high-value or security-critical content where inference latency can be minutes.

---

## 3. Ensemble of Transforms

### 3.1 Does Combining Multiple Defenses Increase Robustness?

**Yes, empirically — but the relationship is sublinear, and adaptive attackers can target the composition.**

The standard result from the 2018 crop of defenses: combining bit-depth reduction + Gaussian blur + median filter raises adversarial accuracy more than any single transform, but not additively. Guo et al. (ICLR 2018) reported that ensembles of transformations outperformed individual transforms against black-box attacks.

**Why ensembles help:**
- Different transforms target different frequency bands and attack strategies.
- A perturbation optimized against one transform may not survive another.
- The attacker must craft a perturbation that survives all stages simultaneously, which is a stricter constraint.

**Why ensembles are not a certificate:**
- BPDA (Section 4) can propagate gradients through any differentiable-approximable sequence of transforms.
- EOT (Section 4) can optimize against the expected effect of any stochastic transform.
- The guarantee does not compound: a pipeline with three 90%-effective defenses does not yield 99.9% robustness in the adversarial setting.

### 3.2 Optimal Ordering of Transforms

The general principle: apply the most aggressive, non-invertible transforms **first** to destroy perturbation structure before finer transforms.

**Recommended order (from research evidence):**

```
1. Bit-depth reduction         — destroys fine-grained ε-ball perturbations
2. Median / Gaussian filter    — spatial smoothing of residual artifacts
3. JPEG / DCT quantization     — frequency-domain compression
4. Wavelet soft-threshold      — sparse coefficient shrinkage
5. Random resize + pad         — geometric desynchronization
6. Optional noise injection    — if targeting randomized smoothing certification
```

Rationale:
- Bit-depth reduction first eliminates adversarial increments that are smaller than the quantization step size. A perturbation of `ε = 1/255` is completely erased when reducing from 8-bit to 5-bit (quantization step = 8/255).
- Spatial filters after quantization clean up quantization artifacts and any remaining high-frequency perturbation.
- Geometric transforms last, because they are non-invertible and if applied first may complicate the alignment of later frequency-domain operations.

**Empirical finding from Guo et al. (ICLR 2018):** Image quilting → total variation minimization performed best among their evaluated compositions against C&W attacks on ImageNet, achieving ~60% accuracy vs ~2% for an undefended model.

### 3.3 Diminishing Returns Analysis

Each additional transform provides decreasing marginal benefit against an adaptive attacker who knows the pipeline. Against black-box or oblivious attackers, stacking defenses is strictly helpful.

| Pipeline depth | Clean acc loss | Black-box robust acc | Adaptive (BPDA) robust acc |
|----------------|---------------|---------------------|---------------------------|
| 0 (baseline)   | 0%            | ~0% (catastrophic)  | ~0% |
| 1 transform    | ~2–5%         | ~40–60%             | ~10–20% |
| 2 transforms   | ~5–10%        | ~55–70%             | ~15–30% |
| 3+ transforms  | ~10–20%       | ~65–75%             | ~20–35% |

*Approximate figures synthesized from Guo et al. 2018, Athalye et al. 2018, and subsequent literature.*

The practical takeaway for pixmask: a **2–3 transform pipeline** (bit-depth + spatial filter + JPEG) captures most available benefit without excessive clean accuracy cost. Adding more transforms beyond 3 yields minimal adaptive robustness gain at increasing clean accuracy cost.

---

## 4. Adaptive Attack Resistance

### 4.1 BPDA — Backward Pass Differentiable Approximation

**Citation:** Athalye, A., Carlini, N., Wagner, D. — "Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples." ICML 2018.
**ArXiv:** https://arxiv.org/abs/1802.00420

#### Core Concept

Many preprocessing defenses are **non-differentiable**: median filter, JPEG quantization, bit-depth reduction, random resize. This causes gradient-based attacks to fail because `∂f(g(x))/∂x` is undefined or zero through the discrete operations in `g`.

BPDA exploits a key insight: during the **backward pass** (gradient computation), replace `g` with a differentiable approximation `g̃` such that `g̃(x) ≈ g(x)` but `∂g̃/∂x` exists. During the **forward pass**, use the true `g(x)`. This gives:

```
Forward:   loss = classifier_loss(f(g(x)))        # real defense
Backward:  ∂loss/∂x ≈ J_{g̃}(x)ᵀ · ∂loss/∂g̃(x)  # differentiable substitute
```

The simplest substitute: **identity**, i.e., `g̃(x) = x`. This means treating the preprocessing as a no-op during backward, which is valid when `g` is approximately the identity (e.g., light Gaussian blur, small JPEG quantization). The approximation is looser for aggressive transforms.

#### What BPDA Breaks

From the case study at ICLR 2018: **7 of 9 non-certified defenses relied on obfuscated gradients**. BPDA successfully circumvented 6 completely and 1 partially. The two defenses not relying on gradient obfuscation were those with genuine geometric or distributional robustness.

Specific defense families vulnerable to BPDA:
- JPEG preprocessing (identity or smooth approximation of quantization)
- Bit-depth reduction (staircase function — identity approximation works)
- Thermometer encoding (differentiable approximation available)
- Input transformations from Guo et al. 2018 (all were broken)
- Pixel deflection (identity works since spatial structure is preserved)

#### What BPDA Cannot Break Easily

1. **Transforms with no good differentiable approximation:** Image quilting (patch replacement from a corpus) has no smooth approximation; the corpus lookup is discrete and input-dependent in a complex way.
2. **Transforms where g̃ = identity is a poor approximation:** If `‖g(x) - x‖` is large (aggressive compression, heavy noise), the BPDA gradient is misleading and the attack accumulates error.
3. **Genuinely certified defenses:** Randomized smoothing with its Cohen et al. certificate cannot be broken by BPDA — the certificate is a mathematical guarantee, not an empirical claim.

### 4.2 EOT — Expectation Over Transformation

**Citation:** Athalye, A., Engstrom, L., Ilyas, A., Kwok, K. — "Synthesizing Robust Adversarial Examples." ICML 2018.
**ArXiv:** https://arxiv.org/abs/1707.07397

#### Core Concept

For **stochastic** defenses `g_θ` where `θ ~ T` is a random transformation (e.g., random resize, random JPEG quality, random Gaussian noise), naive gradient computation through a single sample of `g_θ` gives a noisy gradient that averages to zero across different `θ` draws. The attack fails not because it is blocked but because the gradient signal is too noisy.

EOT resolves this by optimizing the **expected loss**:

```
maximize E_{θ ~ T}[ log P(y_adv | f(g_θ(x))) ]
```

The gradient is:

```
∂/∂x E_{θ ~ T}[loss] = E_{θ ~ T}[∂loss/∂x]
```

Estimated via Monte Carlo:

```
(1/K) Σ_{k=1}^{K} ∂loss(f(g_{θ_k}(x)))/∂x
```

with `K = 10–50` samples per gradient step being sufficient in practice.

**Combined BPDA+EOT:** For stochastic non-differentiable defenses:
1. Apply BPDA: replace `g_θ` with differentiable `g̃_θ` in backward pass.
2. Apply EOT: average the BPDA gradient over K samples of `θ`.

This combination breaks most stochastic preprocessing defenses known as of 2024.

#### What Stochastic Preprocessing Survives EOT

Stochasticity alone is not sufficient protection if the transformation is well-approximable. The key resistance factor is **variance per step** — how much does a single sample of `g_θ` move the loss landscape per gradient step?

High-variance transforms (e.g., large random crops, random color jitter, random kernel sizes) require more EOT samples `K` to estimate a useful gradient, making the attack more computationally expensive. But with K → ∞ samples, EOT always works if BPDA's approximation is valid.

**The only reliable defense against EOT+BPDA:** Remove the differentiable structure that BPDA requires, or use transforms where no good `g̃` exists.

### 4.3 Which Transforms Are Hardest to Attack Adaptively?

**Ranked hardest to easiest for an adaptive attacker:**

| Transform | BPDA approximation quality | EOT sample efficiency | Overall adaptive resistance |
|-----------|---------------------------|----------------------|----------------------------|
| Image quilting (random patch corpus) | Very poor (corpus lookup is discrete/non-local) | Moderate | **High** |
| Generative model purification (DiffPure) | Poor (SDE reverse requires adjoint, expensive) | Low (each sample = full diffusion) | **High (cost)** |
| Random resize + pad (large range) | Identity (poor when scale ratio is large) | Moderate (K~20 sufficient) | **Medium-High** |
| Pixel deflection | Identity (reasonable approximation) | Low (K~10 sufficient) | **Medium** |
| Random JPEG quality (wide range) | Smooth approximation exists | Low (K~10 sufficient) | **Medium** |
| Gaussian blur (fixed kernel) | Identity (very good approximation) | Not needed (deterministic) | **Low** |
| Bit-depth reduction (fixed) | Identity (fair approximation) | Not needed (deterministic) | **Low** |
| JPEG (fixed quality) | Smooth approx (differentiable JPEG exists) | Not needed | **Low** |

**Key insight:** Image quilting is uniquely hard because the nearest-neighbor patch lookup is both discrete and spatially non-local. There is no natural differentiable relaxation that preserves the defense's semantics. The corpus is also not differentiable with respect to the input. This property was noted by Guo et al. but not fully exploited.

---

## 5. The Non-Differentiability Argument

### 5.1 Why Stochastic and Non-Differentiable Transforms Resist Gradient Attacks

Standard gradient-based attacks (FGSM, PGD, C&W, AutoAttack) rely on computing `∂L/∂x` through the full inference pipeline. If the preprocessing `g` is non-differentiable, the chain rule breaks:

```
∂L/∂x = (∂L/∂g(x)) · (∂g(x)/∂x)
                           ↑
                    undefined or zero
```

This has three consequences:
1. **Gradient masking:** The attacker receives zero or undefined gradient signal, so iterative gradient methods stall.
2. **Gradient noise:** For stochastic `g`, gradients computed on a single sample point in a random direction, providing no stable descent direction for the attack.
3. **Gradient bias:** For deterministic but non-differentiable `g` (e.g., step function in bit-depth reduction), gradients are zero almost everywhere, providing no information about which direction to perturb.

**The limitation:** As shown by Athalye et al. (BPDA, EOT), these properties provide only **gradient obfuscation**, not true robustness. A sufficiently determined attacker works around them. The defense is effective against:
- Off-the-shelf gradient attacks that do not account for the preprocessing
- Transfer attacks from models without the preprocessing
- Black-box query attacks (if the transform is applied server-side)

But not against:
- White-box adaptive attackers who know the pipeline and apply BPDA+EOT

### 5.2 Pixel Deflection + Wavelet Denoising

**Citation:** Prakash, A., Moran, N., Garber, S., DiLillo, A., Storer, J. — "Deflecting Adversarial Attacks with Pixel Deflection." CVPR 2018.

#### Algorithm

```
Input: image I of size H × W × C, deflection count R, saliency map S

For r = 1 to R:
    1. Sample target pixel (i, j) uniformly from H × W
    2. Sample source pixel (i', j') from a local window of radius r_max,
       weighted by (1 - S(i,j)) — non-salient regions deflect more
    3. I[i, j, :] ← I[i', j', :]  # overwrite target with source value

Apply wavelet transform to each channel of I:
    W = DWT(I, wavelet='db4', level=3)
    For each subband at each level:
        W_sub ← soft_threshold(W_sub, λ)   # λ tuned to noise level
    I ← IDWT(W)

Return I
```

#### Why It Resists Gradient Attacks

1. **Random pixel indexing is non-differentiable.** The target pixel `(i, j)` and source `(i', j')` are sampled stochastically. `∂I_deflected/∂I_original` is undefined for the deflected pixels.

2. **Saliency weighting provides coverage.** Salient regions (high `S`) are deflected less, preserving semantic content. Non-salient regions (background, texture) are deflected more, and adversarial perturbations typically concentrate on fine-grained texture regions — exactly where deflection is strongest.

3. **Wavelet soft-thresholding removes sparse perturbations.** Adversarial perturbations in the wavelet domain appear as small-amplitude, broadly distributed coefficients. Soft-thresholding with threshold `λ` shrinks all coefficients by `λ` toward zero, which zeroes out adversarial residuals while preserving large natural coefficients (edges, blobs).

4. **Compounding stochasticity.** Each forward pass uses different random deflection indices, so the attack cannot accumulate gradient information across passes (it is blocked by EOT's variance, not by BPDA).

**Empirical results (CVPR 2018):** Against C&W attack at ε = 0.03 on ImageNet top-5:
- Undefended: ~0% accuracy
- Pixel deflection + wavelet: competitive with adversarial training baselines without retraining

#### Weakness

Despite the stochastic deflection, BPDA + EOT with K ≈ 10-20 samples was shown to be sufficient to recover gradients and construct adaptive attacks with moderate success. The defense degrades under strong adaptive attackers but remains effective against non-adaptive threats.

### 5.3 Random JPEG Quality

**Defense:** At each preprocessing call, sample quality `q ~ Uniform(low, high)`, e.g., `q ~ U(50, 95)`, and apply JPEG compression at that quality.

**Why it resists naive gradient attacks:**
- JPEG DCT quantization has zero gradient almost everywhere (piecewise constant quantization table lookup).
- The random `q` means different forward passes produce different outputs for the same input.

**Why BPDA+EOT breaks it:**
- Differentiable JPEG implementations exist (e.g., `torchjpeg`, `DiffJPEG`). These replace the discrete quantization step with a soft rounding function: `round(x) → x + (round(x) - x).detach()` (straight-through estimator).
- Once a differentiable approximation `g̃_q` exists, EOT averages over `K` samples of `q ~ U(50, 95)`, recovering a stable gradient.

**The practical value remains for non-adaptive adversaries:** Most deployed adversarial examples (transfer attacks, image-based jailbreaks crafted against an undefended model) are not specifically optimized against JPEG quality randomization. The defense provides meaningful empirical robustness against this majority threat class.

**Implementation note for pixmask:** Randomizing the quality per call is better than a fixed quality because it raises the EOT sample complexity `K` needed to mount an adaptive attack, and fixed-quality defenses have known differentiable approximations that are trivial to substitute.

---

## 6. Implications for pixmask

### 6.1 What pixmask Can and Cannot Claim

**What is achievable:**
- Empirical robustness: pixmask's pipeline significantly degrades adversarial examples crafted by off-the-shelf tools (FGSM, PGD, AutoAttack, transfer attacks).
- Probabilistic resistance: stochastic transforms (random JPEG quality, random resize+pad) raise the cost of adaptive attacks without eliminating it.
- Preprocessing contract: any perturbation smaller than the quantization step of bit-depth reduction (ε < 8/255 for 5-bit reduction) is **provably erased** — this is a weak but genuine guarantee.

**What is not achievable without randomized smoothing integration:**
- L2 or L-inf certified radius for the full pipeline
- Guarantees against white-box adaptive attackers who know the exact pipeline and apply BPDA+EOT

### 6.2 Path to Certification

If a certified guarantee is required, the cleanest integration is:

```
image → [pixmask transforms] → [add N(0, σ²)] → classifier
```

The Cohen et al. CERTIFY algorithm applies to this composition. The pixmask transforms reduce the effective perturbation before the Gaussian noise is added, meaning:
- A perturbation `δ` entering pixmask with `‖δ‖₂ = r₀`
- After a contractive transform `g` (Lipschitz constant L ≤ 1), `‖g(δ)‖₂ ≤ r₀`
- The randomized smoothing certification with radius `r` at the classifier input certifies robustness to `r₀ ≤ r` at the network input

In other words: **pixmask preprocessing makes the certified radius more useful** — the same `σ` and `n` achieve a stronger guarantee on the original input space because perturbations are compressed before reaching the smoothed classifier.

The cost (100,000 forward passes, ~150 seconds) is unchanged. This path only makes sense for offline certification, not real-time serving.

### 6.3 Practical Defense Stack Recommendations

**For real-time VLM preprocessing (latency < 50ms):**

```
1. Bit-depth reduction (5-bit)         — erases ε < 8/255 perturbations provably
2. Gaussian filter (σ = 0.5–1.0)      — fast, erases fine-grained texture perturbations
3. Random JPEG quality (q ~ U(70, 92)) — DCT quantization + stochastic resistance
```

**For high-assurance offline sanitization (latency < 10s):**

```
1. Bit-depth reduction (5-bit)
2. Median filter (3×3)
3. Total variation minimization
4. Random resize + pad (random scale ∈ [0.7, 1.0])
5. Wavelet soft-threshold denoising
```

**For provably certified use cases (latency: minutes):**
```
1. Any pixmask pipeline above
2. Gaussian noise injection N(0, σ²), σ ≈ 0.25
3. Randomized smoothing with n = 100,000 passes (Cohen et al. CERTIFY)
```

### 6.4 The Honest Tradeoff Table

| Defense class | Clean acc cost | Empirical robustness | Adaptive robustness | Certification | Latency |
|---------------|----------------|---------------------|---------------------|---------------|---------|
| Bit-depth (5-bit) | ~1% | High | Low (BPDA works) | Partial (ε < step) | <1ms |
| Gaussian blur | ~2% | Medium | Low (identity approx) | None | <1ms |
| JPEG (fixed q=75) | ~3% | Medium-High | Low (DiffJPEG works) | None | ~1ms |
| JPEG (random q) | ~3% | Medium-High | Medium (EOT needed) | None | ~1ms |
| Pixel deflection + wavelet | ~5% | High | Medium (K~20 EOT) | None | ~5ms |
| Random resize + pad | ~2% | High | Medium-High | None | ~1ms |
| Image quilting | ~8% | Very High | High (no good BPDA) | None | ~50ms |
| RS (predict, n=1000) | ~10% | High | High (certified) | None | ~1.5s |
| RS (certify, n=100K) | ~10% | High | Certified | L2 ball | ~150s |
| DiffPure (empirical) | ~5% | Very High | High (expensive) | Approx | ~30s |

---

## References

- Cohen, J., Rosenfeld, E., Kolter, J.Z. — "Certified Adversarial Robustness via Randomized Smoothing." ICML 2019. arXiv:1902.02918
- Salman, H., Li, J., Razenshteyn, I., Zhang, P., Zhang, H., Bubeck, S., Yang, G. — "Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers." NeurIPS 2019. arXiv:1906.04584
- Nie, W., Guo, B., Huang, Y., Xiao, C., Vahdat, A., Anandkumar, A. — "Diffusion Models for Adversarial Purification." ICML 2022. arXiv:2205.07460
- Athalye, A., Carlini, N., Wagner, D. — "Obfuscated Gradients Give a False Sense of Security." ICML 2018. arXiv:1802.00420
- Athalye, A., Engstrom, L., Ilyas, A., Kwok, K. — "Synthesizing Robust Adversarial Examples (EOT)." ICML 2018. arXiv:1707.07397
- Guo, C., Rana, M., Cisse, M., van der Maaten, L. — "Countering Adversarial Images via Input Transformations." ICLR 2018. arXiv:1711.00117
- Xie, C., Wang, J., Zhang, Z., Ren, Z., Yuille, A. — "Mitigating Adversarial Effects Through Randomization." ICLR 2018. arXiv:1711.01991
- Prakash, A., Moran, N., Garber, S., DiLillo, A., Storer, J. — "Deflecting Adversarial Attacks with Pixel Deflection." CVPR 2018.
- Official smoothing code: https://github.com/locuslab/smoothing (n0=100, n=100000, alpha=0.001, radius = sigma * norm.ppf(pABar))
