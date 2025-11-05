# Pre‑Processing Defenses to Neutralize Adversarial Image Attacks

**Scope.** Model‑agnostic input transformations applied *before* inference. These methods alter an image to erase/blur subtle, high‑frequency perturbations while preserving human‑salient content. They require no model retraining.

---

## A. Model‑agnostic input transformations (by family)

### 1) **Feature Squeezing (bit‑depth reduction + smoothing)**

* **Mechanism.** Reduce per‑channel bit‑depth (e.g., 8→5 bits) and/or apply spatial smoothing (e.g., median filtering). Coalescing nearby pixel values collapses tiny, adversarial changes (e.g., 128→129) onto the same quantized value and removes pixel‑level outliers.
* **Why it helps.** Adversarial signals tend to be low‑amplitude and high‑frequency; quantization and smoothing directly suppress both.
* **Use in practice.**

  * Bit‑depth: 4–5 bits/channel often preserves semantics;
  * Smoothing: 3×3 or 5×5 median; optional Gaussian blur.
* **Key paper / code.** Xu, Evans, Qi, **NDSS 2018** “Feature Squeezing: Detecting Adversarial Examples…” (also usable as a preprocessing defense). Official paper & NDSS slides; reference implementation (UVA) include bit‑depth and median filters. ([NDSS Symposium][1])

---

### 2) **Lossy compression & frequency‑domain filtering**

**2a. JPEG (DCT quantization) + variants**

* **Mechanism.** Blockwise 8×8 DCT with quantization removes high‑frequency coefficients; re‑encoding at moderate/low quality “washes out” small perturbations.
* **Why it helps.** Most adversarial noise resides in high frequencies; JPEG’s quantization discards or attenuates them.
* **Evidence.** Early empirical note on JPG weakening adversarial artifacts (Dziugaite et al., 2016). JPEG typically outperforms several basis‑function defenses across threat models (Shaham et al., 2018). SHIELD (KDD 2018) combines JPEG with *spatially randomized* qualities and model “vaccination.” ([ResearchGate][2])
* **Code.** SHIELD: randomized JPEG defense (GitHub: poloclub/jpeg‑defense). ([GitHub][3])

**2b. WebP compression (+ simple flips)**

* **Mechanism.** WebP’s modern lossy pipeline (DCT‑like + in‑loop deblocking) prunes imperceptible details and can outperform JPEG; a single horizontal flip further disrupts structured perturbations.
* **Key paper / code.** Wang et al., 2019: “An Efficient Pre‑processing Method to Eliminate Adversarial Noise” (WebP + flip). ([arXiv][4])

**2c. Frequency transforms (DCT/FFT/wavelet) & PCA**

* **Mechanism.** Project to a compact basis (DCT, FFT, wavelet; PCA), suppress/threshold high‑frequency or low‑energy components, reconstruct.
* **Why it helps.** Directly targets the typical spectral footprint of pixel‑level attacks.
* **Evidence & code pointers.**

  * Basis‑function study (PCA, DCT, wavelet soft‑thresholding) with systematic evaluation (Shaham et al., 2018). ([arXiv][5])
  * Wavelet soft‑thresholding available in scikit‑image / PyWavelets (BayesShrink, hard/soft). ([scikit-image][6])
  * FFT low‑pass/blurring tutorials & OpenCV filtering docs (for direct implementation). ([OpenCV-Python Tutorials][7])

---

### 3) **Spatial smoothing & classical denoisers**

* **Mechanism.** Low‑pass filters (Gaussian), median/mean filters, bilateral (edge‑preserving), and non‑local means (NLM) reduce outliers and texture‑scale noise.
* **Why it helps.** These methods remove high‑frequency and pixel‑outlier patterns typical of many adversarial perturbations; NLM is especially effective at preserving structure while averaging similar patches.
* **Evidence / implementations.**

  * Gaussian/low‑pass used as a baseline defense in basis‑function studies. ([arXiv][5])
  * scikit‑image: `denoise_bilateral`, `denoise_nl_means`, `denoise_wavelet` (documentation and examples). ([scikit-image][8])

---

### 4) **Total Variation Minimization (TVM) & Image Quilting**

* **Mechanism.**

  * **TVM:** Optimization that finds an image close to the input but with minimal total variation—encouraging piecewise smoothness and removing tiny oscillations.
  * **Image Quilting:** Replace each patch with the closest patch from a library of *clean* patches; adversarial micro‑textures are overwritten by natural texture statistics.
* **Why it helps.** TVM denoises small, oscillatory corruptions; quilting restores patches to a *natural manifold* that adversarial noise rarely matches.
* **Key paper / code.** Guo et al., **ICLR 2018**: “Countering Adversarial Images via Input Transformations.” Facebook Research released a reference implementation (JPEG, pixel quantization, TVM solver, quilting). ([OpenReview][9])
* **General TV implementations.** scikit‑image (`denoise_tv_chambolle`, `denoise_tv_bregman`) and research examples (ADMM / SCICO). ([scikit-image][8])
* **Image quilting references (non‑adversarial origin).** Public repos demonstrate Efros–Freeman quilting; useful if building a quilting defense prototype. ([GitHub][10])

---

### 5) **Randomized input transformations**

**5a. Random resizing + padding**

* **Mechanism.** At inference, resize the image to a *random* size within a band and zero‑pad back to the expected size.
* **Why it helps.** Misaligns attack‑calibrated pixel locations and is non‑differentiable/stochastic, frustrating gradient‑based optimization.
* **Key paper / code.** Xie et al., **ICLR 2018**; official repo used in the NIPS 2017 defense track. ([arXiv][11])

**5b. Pixel Deflection (+ wavelet denoising)**

* **Mechanism.** Randomly replace a subset of pixels with values from random neighbors (small local shuffle), then apply wavelet soft‑thresholding to smooth artifacts.
* **Why it helps.** CNNs are tolerant to small natural jitter; local pixel shuffling disrupts finely tuned attack patterns.
* **Key paper / code.** Prakash et al., **CVPR 2018**; official arXiv/CVPR PDF; open‑source reference implementation (Keras/NumPy). ([CVF Open Access][12])

> **Caution on randomized defenses.** Several notable randomized/non‑differentiable defenses were later broken under *adaptive white‑box* evaluations (e.g., Athalye & Carlini, 2018; “obfuscated gradients”). Use these transformations as *first‑pass, model‑agnostic hardening*, and **evaluate adaptively**. ([arXiv][13])

---

### 6) **Defenses against image‑scaling attacks (adversarial preprocessing)**

* **Threat.** Attacker crafts an image that looks benign at full size but, after *downscaling*, reveals a different (illicit) image—exploiting the resampler so a few strategically located pixels dominate the thumbnail/scaled output.
* **Mechanism / mitigations.**

  1. **Identify & restore high‑influence pixels** before resizing;
  2. **Improve resampling** to avoid disproportionate pixel influence;
  3. **Randomize interpolation** or apply small perturbations to destroy hidden patterns.
* **Key paper / code.** Quiring et al., **USENIX Security 2020**; project site with code; maintained GitHub repo implements both attacks and defenses. ([USENIX][14])

---

## B. Core SP/CV “recipes” (drop‑in algorithms & how they map to the families above)

1. **DCT Quantization (the core of JPEG)**

   * **How.** Block DCT → quantize high‑frequency coefficients → inverse DCT.
   * **Why.** Directly discards high‑frequency adversarial content.
   * **Use.** Re‑save to JPEG (quality 50–90) or use randomized blockwise qualities (SHIELD). Code readily available in Pillow/OpenCV; SHIELD repo for randomized variants. ([GitHub][3])

2. **FFT Low‑Pass Filtering**

   * **How.** FFT → apply circular/gaussian low‑pass mask → inverse FFT.
   * **Why.** Blunt but effective high‑frequency removal.
   * **Use.** NumPy/SciPy or OpenCV; see OpenCV Fourier tutorials. ([OpenCV-Python Tutorials][7])

3. **Median Filtering**

   * **How.** Replace each pixel by the median in a k×k window.
   * **Why.** Robust to “salt‑and‑pepper” and small patches; removes outliers while preserving edges better than mean filtering.
   * **Use.** OpenCV/scikit‑image built‑ins. (Median also compared in SHIELD.) ([poloclub.github.io][15])

4. **Bilateral Filtering (edge‑preserving)**

   * **How.** Spatially local averaging weighted by *intensity similarity*; blurs flat regions while preserving edges.
   * **Why.** Dampens low‑amplitude noise without harming object contours.
   * **Use.** `skimage.restoration.denoise_bilateral` or OpenCV `bilateralFilter`. ([scikit-image][8])

5. **Bit‑Depth Reduction (Quantization)**

   * **How.** Map 256 levels → 32 (or fewer), then rescale.
   * **Why.** Snaps tiny adversarial increments to the same bucket.
   * **Use.** Simple lookup/rescale; included in feature squeezing code. ([GitHub][16])

6. **Total Variation (TV) Denoising**

   * **How.** Solve ROF minimization (e.g., Chambolle/Bregman/ADMM).
   * **Why.** Suppresses oscillatory perturbations; preserves piecewise‑constant regions.
   * **Use.** `denoise_tv_chambolle` / `denoise_tv_bregman` (scikit‑image); ADMM examples in SCICO. ([scikit-image][8])

7. **Wavelet Soft‑Thresholding**

   * **How.** Decompose; shrink small coefficients (e.g., BayesShrink); reconstruct.
   * **Why.** Natural‑image sparsity in wavelet domain makes small adversarial coefficients easy to remove.
   * **Use.** `denoise_wavelet` (scikit‑image), PyWavelets thresholding utilities. ([scikit-image][6])

---

## C. Notable systems & evaluations (papers with implementations)

* **Feature Squeezing (bit‑depth + smoothing).** Xu, Evans, Qi, **NDSS 2018**; code (UVA). ([NDSS Symposium][1])
* **JPEG/WebP compress‑and‑restore.**

  * Dziugaite et al., 2016 (early empirical note on JPG). ([ResearchGate][2])
  * Shaham et al., 2018 (JPEG, wavelet, PCA comparisons). ([arXiv][5])
  * **SHIELD** (randomized JPEG + vaccination), **KDD 2018**; code. ([arXiv][17])
  * **WebP + flip** (Wang et al., 2019), preprint. ([arXiv][4])
* **TVM & Image Quilting** (Guo et al., **ICLR 2018**) + Facebook Research implementation. ([OpenReview][9])
* **Randomization: resize+pad** (Xie et al., **ICLR 2018**), code; **Pixel Deflection** (Prakash et al., **CVPR 2018**), code. ([arXiv][11])
* **Scaling‑attack defenses** (Quiring et al., **USENIX Security 2020**) with public code/site. ([USENIX][14])

---

## D. Practical integration tips (for readers implementing these as a pre‑inference layer)

* **Chain lightly, measure trade‑offs.** Start with JPEG/WebP or bit‑depth + small median filter; add TV or wavelet only if needed to balance *robustness vs. benign accuracy*. (See SHIELD’s comparisons for JPEG vs. median vs. TV.) ([poloclub.github.io][15])
* **Randomize when feasible.** Light random resize/pad or randomized JPEG qualities (SHIELD) raise attacker cost in black‑/gray‑box settings; always evaluate under an adaptive threat model. ([arXiv][11])
* **Mind the pipeline.** If your service resizes images (thumbnails, VLM adapters), *sanitize before scaling* to neutralize scaling attacks. Use robust resampling and pixel‑influence restoration if you must downscale. ([USENIX][14])
* **Libraries.** Pillow/OpenCV for JPEG/WebP/filters; scikit‑image/PyWavelets for TV and wavelets; official repos for quilting, randomization, and pixel deflection. ([GitHub][18])

---

## E. Caveats & evaluation guidance

* **Adaptive attacks matter.** Many non‑differentiable or stochastic defenses were later bypassed via *obfuscated‑gradient* analyses; report results against white‑box adaptive attacks, not only transfer or black‑box attacks. ([arXiv][13])
* **Benign‑accuracy impact.** Over‑aggressive denoising (heavy blur, strong quantization) can reduce clean accuracy; tune parameters (e.g., JPEG quality, TV weight, NLM patch/stride) with validation on clean data. (See SHIELD’s runtime/accuracy table vs. median/TV.) ([poloclub.github.io][15])

---

## F. Quick “where to find” (papers & code)

* **Feature Squeezing** — Paper & NDSS slides; code (UVA). ([NDSS Symposium][1])
* **JPEG effectiveness (early note)** — Dziugaite et al., 2016. ([ResearchGate][2])
* **Basis‑function transforms (PCA/DCT/wavelet)** — Shaham et al., 2018 (arXiv). ([arXiv][5])
* **SHIELD (randomized JPEG)** — Paper & GitHub. ([arXiv][17])
* **WebP + flip** — Wang et al., 2019 (arXiv). ([arXiv][4])
* **TVM & Image Quilting** — OpenReview page; Facebook Research repo. ([OpenReview][9])
* **Random resize + pad** — Paper & GitHub. ([arXiv][11])
* **Pixel Deflection** — CVPR paper & GitHub. ([CVF Open Access][12])
* **Scaling‑attack defenses** — USENIX paper; project site; code. ([USENIX][14])
* **Wavelet/TV/NLM implementations** — scikit‑image & PyWavelets docs. ([scikit-image][8])

---

## References (selected)

* Xu, Evans, Qi. *NDSS 2018.* “Feature Squeezing…” + code. ([NDSS Symposium][1])
* Dziugaite et al., 2016. “A study of the effect of JPG compression on adversarial images.” ([ResearchGate][2])
* Shaham et al., 2018. “Defending against Adversarial Images using Basis Functions Transformations.” ([arXiv][5])
* Guo et al., *ICLR 2018.* “Countering Adversarial Images via Input Transformations.” + Facebook Research code. ([OpenReview][9])
* Xie et al., *ICLR 2018.* “Mitigating Adversarial Effects Through Randomization.” + code. ([arXiv][11])
* Prakash et al., *CVPR 2018.* “Deflecting Adversarial Attacks with Pixel Deflection.” + code. ([CVF Open Access][12])
* Wang et al., 2019. “An Efficient Pre‑processing Method to Eliminate Adversarial Noise” (WebP + flip). ([arXiv][4])
* Quiring et al., *USENIX Security 2020.* “Adversarial Preprocessing: Understanding and Preventing Image‑Scaling Attacks.” + site & code. ([USENIX][14])
* Athalye, Carlini, Wagner, 2018. “Obfuscated Gradients Give a False Sense of Security.” (evaluation caveats). ([arXiv][13])

---

**Note.** : * All methods here are *preprocessing only* (no retraining required).

[1]: https://www.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_03A-4_Xu_paper.pdf?utm_source=chatgpt.com "Feature Squeezing:"
[2]: https://www.researchgate.net/publication/305779814_A_study_of_the_effect_of_JPG_compression_on_adversarial_images?utm_source=chatgpt.com "A study of the effect of JPG compression on adversarial ..."
[3]: https://github.com/poloclub/jpeg-defense?utm_source=chatgpt.com "poloclub/jpeg-defense: SHIELD: Fast, Practical ..."
[4]: https://arxiv.org/pdf/1905.08614?utm_source=chatgpt.com "An Efficient Pre-processing Method to Eliminate ..."
[5]: https://arxiv.org/abs/1803.10840 "[1803.10840] Defending against Adversarial Images using Basis Functions Transformations"
[6]: https://scikit-image.org/docs/0.25.x/auto_examples/filters/plot_denoise_wavelet.html?utm_source=chatgpt.com "Wavelet denoising — skimage 0.25.2 documentation"
[7]: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html?utm_source=chatgpt.com "Fourier Transform - OpenCV-Python Tutorials - Read the Docs"
[8]: https://scikit-image.org/docs/0.25.x/api/skimage.restoration.html?utm_source=chatgpt.com "skimage.restoration"
[9]: https://openreview.net/forum?id=SyJ7ClWCb&utm_source=chatgpt.com "Countering Adversarial Images using Input Transformations"
[10]: https://github.com/axu2/image-quilting?utm_source=chatgpt.com "axu2/image-quilting: A numpy implementation of the paper ..."
[11]: https://arxiv.org/abs/1711.01991 "[1711.01991] Mitigating Adversarial Effects Through Randomization"
[12]: https://openaccess.thecvf.com/content_cvpr_2018/papers/Prakash_Deflecting_Adversarial_Attacks_CVPR_2018_paper.pdf?utm_source=chatgpt.com "Deflecting Adversarial Attacks With Pixel Deflection"
[13]: https://arxiv.org/abs/1802.00420?utm_source=chatgpt.com "Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples"
[14]: https://www.usenix.org/system/files/sec20fall_quiring_prepub.pdf?utm_source=chatgpt.com "Understanding and Preventing Image-Scaling Attacks in ..."
[15]: https://poloclub.github.io/polochau/papers/18-kdd-shield.pdf?utm_source=chatgpt.com "Fast, Practical Defense and Vaccination for Deep Learning ..."
[16]: https://github.com/uvasrg/FeatureSqueezing?utm_source=chatgpt.com "Detecting Adversarial Examples in Deep Neural Networks"
[17]: https://arxiv.org/abs/1802.06816?utm_source=chatgpt.com "Shield: Fast, Practical Defense and Vaccination for Deep Learning using JPEG Compression"
[18]: https://github.com/facebookarchive/adversarial_image_defenses?utm_source=chatgpt.com "Countering Adversarial Image using Input Transformations."
