# Evaluation Methodology for Image Sanitization Defenses

> Research for pixmask — a C++ image sanitization layer before multimodal LLMs.
> Defense families: Feature Squeezing, JPEG/frequency filtering, spatial smoothing, TVM, randomized transforms.

---

## 1. Defense Effectiveness: Attack Success Rate (ASR)

### Definition

**ASR** = fraction of adversarial inputs that elicit the attacker's desired output after passing through the defense.

For classifiers:
```
ASR = |{x_adv : f(defense(x_adv)) == target_class}| / |X_adv|
```

For VLM jailbreaks (following Zou et al. 2023 / GCG threat model):
```
ASR = P[model produces affirmative / harmful response | adversarial image input]
```

Operationally, judge with a refusal classifier or substring matching on refusal phrases ("I cannot", "I'm sorry", "As an AI...").

### Threat Models to Cover

| Threat Model | Attack | Epsilon |
|---|---|---|
| Lp white-box | PGD-40, C&W L2 | ε = 8/255 (L∞), ε = 0.5 (L2) |
| Lp black-box | Square Attack, SimBA | same budgets |
| VLM jailbreak | GCG visual suffix, PGD-on-embedding | perturbation-bounded |
| Adaptive (BPDA) | Attacker approximates non-differentiable defense | same budgets |
| Adaptive (EOT) | Expectation Over Transformation for stochastic defenses | 30–50 samples |

### Reporting Protocol

Report **four numbers per attack × defense combination**:
1. Clean accuracy / benign task score (no attack, no defense)
2. Accuracy under attack, no defense (attacker's baseline)
3. Accuracy under attack + pixmask (defense effectiveness)
4. Accuracy on clean image + pixmask (utility cost)

This exposes the security–utility tradeoff cleanly.

---

## 2. Image Quality Metrics

These measure how much the sanitized image deviates from the original clean image. Lower degradation = better utility preservation.

### PSNR (Peak Signal-to-Noise Ratio)

```
PSNR = 10 * log10(MAX_I^2 / MSE)
```

- Units: dB. Higher is better.
- Typical thresholds: >30 dB acceptable, >40 dB near-lossless.
- Limitation: purely pixel-level; does not correlate well with perceptual quality.
- Implementation: `cv::PSNR()` in OpenCV; trivial to compute in C++.

### SSIM (Structural Similarity Index)

```
SSIM(x, y) = (2μ_x μ_y + c1)(2σ_xy + c2) / ((μ_x² + μ_y² + c1)(σ_x² + σ_y² + c2))
```

- Range: [−1, 1]. Values >0.95 are generally considered high quality.
- Computed per patch (11×11 window) then averaged.
- Better than PSNR at capturing luminance, contrast, and structure jointly.
- Implementation: `cv::quality::QualitySSIM` (OpenCV contrib) or compute manually.

### LPIPS (Learned Perceptual Image Patch Similarity)

- Uses deep features (AlexNet/VGG) to measure perceptual distance.
- Lower is better (distance, not similarity). Range: [0, 1].
- Far better than PSNR/SSIM at predicting human perceptual judgments.
- Reference: Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric", CVPR 2018.
- Python: `lpips` package. C++ inference requires ONNX export of the backbone.

### FID (Fréchet Inception Distance)

- Measures distribution-level divergence between sanitized and clean image sets.
- Used when evaluating entire output distributions (e.g., across a test set).
- Lower is better.
- Appropriate for comparing defense modes (e.g., JPEG vs. wavelet) at population scale.
- Not per-image; requires at minimum ~2k samples for stable estimates.

### Recommended Reporting

For a sanitization defense paper, report PSNR and SSIM per-image (mean ± std) and LPIPS per-image (mean ± std). FID for population-level comparisons. Do not report only PSNR — reviewers will ask for SSIM/LPIPS.

---

## 3. Task Preservation: LLM Understanding After Sanitization

The sanitized image must still be useful. Three measurement approaches:

### 3a. Classification Accuracy on Clean Images

Run the pixmask pipeline on ImageNet clean validation set. Measure top-1 accuracy drop relative to no preprocessing. Target: <2% drop for mild defenses.

### 3b. VLM Captioning Quality

Feed sanitized images to a reference VLM (LLaVA-1.6, InternVL2). Score with:
- **CIDEr** / **BLEU-4**: n-gram overlap vs. ground-truth captions (MS-COCO captions).
- **BERTScore**: semantic similarity using BERT embeddings.
- **GPT-4V judge**: binary "does the caption correctly describe the scene" on a sample.

### 3c. VQA Score

Use VQAv2 or MMMU. Pass sanitized images; measure exact-match VQA accuracy. Acceptable degradation: <3% absolute.

### 3d. OCR Preservation (For Text-in-Image Attacks)

If sanitization is applied upstream of document understanding, measure OCR word error rate (WER) before/after.

### Protocol

Evaluate task preservation across all defense configurations independently of adversarial inputs — this isolates utility degradation from defense overhead.

---

## 4. Adaptive Attack Robustness

Defenses that obscure gradients (quantization, randomization, non-differentiable ops) are vulnerable to adaptive attacks. Following Athalye, Carlini & Wagner, "Obfuscated Gradients Give a False Sense of Security" (ICML 2018).

### BPDA (Backward Pass Differentiable Approximation)

For a non-differentiable defense `g`:
- Forward pass: `g(x)` (true defense)
- Backward pass: substitute identity `∂g/∂x ≈ I` or a smooth surrogate

Implementation: PGD loop where the gradient is computed through the surrogate. If the defense is a spatial filter, the surrogate is a smooth version of the same filter (e.g., replace median filter with Gaussian of equivalent radius).

pixmask-specific BPDA surrogates:
- Bit-depth reduction → sigmoid approximation with steep slope
- Median filter → Gaussian filter of same radius
- JPEG quantization → differentiable JPEG (DiffJPEG implementation)
- Wavelet threshold → soft-thresholding (already differentiable)
- Randomized resize → fixed resize at mean scale

### EOT (Expectation Over Transformation)

For stochastic defenses (random JPEG quality, random pad/resize):
```
∇_x E_{t~T}[L(f(t(x)), y)] ≈ (1/N) Σ_i ∇_x L(f(t_i(x)), y)
```

Compute gradient as average over N=30–50 sampled transformations per step. PGD with EOT gradient then attacks randomized defenses properly.

### Minimum Standard

A defense is not credible without BPDA/EOT evaluation. Report:
- ASR under standard PGD (naive)
- ASR under BPDA (for non-differentiable transforms)
- ASR under EOT-PGD (for stochastic transforms)
- If all three are low, the defense is meaningful

---

## 5. Benchmark Datasets

### For Classifier Defense Evaluation

| Dataset | Size | Use |
|---|---|---|
| ImageNet val (ILSVRC 2012) | 50k images, 1000 classes | Standard clean/robust accuracy |
| ImageNet-C | 75 corruptions × 50k | Common corruption robustness |
| CIFAR-10 | 10k test, 10 classes | Fast iteration / ablation |
| RobustBench test sets | Curated adversarial | Comparison to leaderboard |

Standard threat models (from RobustBench):
- ImageNet Lp: ε = 4/255 (L∞)
- CIFAR-10 Lp: ε = 8/255 (L∞), ε = 0.5 (L2)

### For VLM/Jailbreak Defense Evaluation

| Dataset | Description |
|---|---|
| MM-SafetyBench | 5,040 text-image pairs across 13 harmful scenarios. Image-based manipulation attacks on 12 VLMs. Liu et al. (2023). |
| AdvBench (text portion) | 520 harmful instructions; use with visual adversarial suffixes. Zou et al. (2023). |
| HarmBench | Standardized red-teaming benchmark; includes multimodal behaviors. |
| RTVLM | Red-teaming dataset across faithfulness, privacy, safety, fairness for VLMs. |
| MMStar / MMMU | Benign capability benchmarks for measuring utility preservation. |

### For Scaling Attack Evaluation

Generate custom pairs using Quiring et al. (2020) attack: craft images that appear benign at native resolution but contain hidden content after downscaling. Test pixmask's randomized interpolation defense.

---

## 6. Prior Work Evaluation Protocols

### Feature Squeezing (Xu et al., NDSS 2018)

- **Datasets**: MNIST, CIFAR-10, ImageNet
- **Detection framing**: adversarial detection, not just accuracy. Reports TPR (true positive rate for adversarial) and FPR (false positive rate on clean inputs).
- **Attacks**: FGSM, BIM, DeepFool, C&W L2, JSMA
- **Metric**: ROC curve + AUC for detection. Also reports classification accuracy under attack.
- **Key protocol**: joint squeezer framework — if any squeezed version disagrees with original by more than threshold τ, flag as adversarial. τ selected on validation set to hit target FPR.
- **Limitation exposed by Athalye et al.**: BPDA breaks the detection because squeezers are non-differentiable. Adaptive attacker produces inputs that are consistent across all squeezers.

### SHIELD (Das et al., KDD 2018)

- **Dataset**: ImageNet
- **Attack types**: black-box (transfer), gray-box (known model, unknown defense), C&W L2, DeepFool
- **Metrics**: attack elimination rate (fraction of adversarial examples neutralized), not clean accuracy drop
- **Results**: 94% black-box elimination, 98% gray-box elimination
- **Protocol**: combines vaccination (fine-tuning on JPEG-compressed images at varying quality levels) + ensemble of vaccinated models + random quality selection at test time
- **Evaluation gap**: does not report BPDA/EOT results; the randomization provides stochasticity but not adaptive-attack robustness

### Input Transformations (Guo et al., ICLR 2018)

- **Datasets**: ImageNet
- **Defenses evaluated**: bit-depth reduction, JPEG compression, total variation minimization, image quilting, cropping+rescaling
- **Attacks**: FGSM, BIM, DeepFool, C&W L2, C&W L∞
- **Metric**: top-1 accuracy on adversarial examples after defense (not detection rate)
- **Key finding**: quilting and TVM most effective; JPEG effective against some attacks but not C&W
- **Protocol weakness**: Athalye et al. showed these break under BPDA since transforms are non-differentiable

### Modern VLM Defense Papers (2023–2024)

Current convention for VLM adversarial defense papers:
1. Report ASR against PGD, PGD+EOT, and transfer attacks
2. Report benign VQA accuracy before/after defense
3. Use at least one open VLM (LLaVA, InternVL) and one closed (GPT-4V via API)
4. Include PSNR/SSIM to show image quality is preserved
5. Test on MM-SafetyBench 13-scenario breakdown (not just aggregate)

---

## 7. C++ Performance Benchmarking

### Google Benchmark Setup

```cpp
#include <benchmark/benchmark.h>
#include "fsq/pipeline.hpp"

static void BM_BitDepthReduction(benchmark::State& state) {
    const int width = state.range(0);
    const int height = state.range(1);
    // Allocate image once outside the loop
    std::vector<uint8_t> img(width * height * 3, 128);
    fsq::ImageView view(img.data(), width, height, 3);

    for (auto _ : state) {
        fsq::bit_depth_reduce(view, /*bits=*/5);
        benchmark::DoNotOptimize(view);
        benchmark::ClobberMemory();
    }

    // Report throughput: pixels processed per second
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) * width * height * 3
    );
    // Report images per second as a counter
    state.counters["img/s"] = benchmark::Counter(
        state.iterations(),
        benchmark::Counter::kIsRate
    );
}

BENCHMARK(BM_BitDepthReduction)
    ->Args({224, 224})
    ->Args({512, 512})
    ->Args({1024, 1024})
    ->Args({2048, 2048});

BENCHMARK_MAIN();
```

### Latency Percentiles

Google Benchmark does not natively emit p95/p99. Use `--benchmark_repetitions=100` and `--benchmark_report_aggregates_only=false` to get per-run times, then compute percentiles in post-processing:

```bash
./pixmask_bench --benchmark_format=json --benchmark_repetitions=100 \
    --benchmark_report_aggregates_only=false \
    > bench_raw.json
python3 scripts/percentiles.py bench_raw.json  # compute p50/p95/p99
```

Alternatively, use `benchmark::ComputeStatistics` with a custom percentile lambda:
```cpp
BENCHMARK(BM_Filter)->ComputeStatistics("p99", [](const std::vector<double>& v) {
    auto sorted = v;
    std::sort(sorted.begin(), sorted.end());
    return sorted[static_cast<size_t>(0.99 * sorted.size())];
})->Repetitions(100);
```

### Memory Profiling

Implement a `benchmark::MemoryManager` subclass or use Valgrind/massif externally:

```bash
# Peak RSS via valgrind
valgrind --tool=massif --pages-as-heap=yes ./pixmask_bench
ms_print massif.out.* | head -40

# Or use /proc/self/status inside benchmark
```

For allocator pressure (useful for pipeline that may malloc per-filter), track allocation count via a custom allocator or `jemalloc` with stats.

### What to Report

| Metric | Method |
|---|---|
| Throughput (img/s) | `Counter::kIsRate` on image count |
| Bytes/second | `state.SetBytesProcessed()` |
| Latency p50/p95/p99 | `--benchmark_repetitions=100` + post-process |
| Memory per image | `MemoryManager` or `/proc/self/status` delta |
| SIMD vs scalar speedup | Separate `BM_*_AVX2` variants |
| Thread scaling | `->Threads(1)->Threads(4)->Threads(8)` |

### Resolutions to Benchmark

Match common VLM input sizes:
- 224×224 (ImageNet/ViT-B standard)
- 336×336 (LLaVA-1.5)
- 448×448 (InternVL2 tile)
- 512×512
- 1024×1024 (high-res tile)
- 2048×2048 (document understanding)

---

## 8. Baseline Comparisons

A fair comparison requires the same attack × dataset × metric across all methods.

### Baselines to Include

| Baseline | Description | Why Include |
|---|---|---|
| No defense | Raw adversarial image → model | Lower bound on robustness; upper bound on utility |
| JPEG q=75 | `cv::imencode(".jpg", img, buf, {cv::IMWRITE_JPEG_QUALITY, 75})` | Canonical compression baseline; widely used |
| JPEG q=50 | Aggressive JPEG | More robust but higher quality cost |
| Gaussian blur σ=1.0 | `cv::GaussianBlur(src, dst, {5,5}, 1.0)` | Simplest spatial smoothing |
| Median filter k=3 | `cv::medianBlur(src, dst, 3)` | Salt-and-pepper removal |
| Bit-depth reduction (5-bit) | Xu et al. baseline | Feature Squeezing reference |
| Random resize+pad | Xie et al. ICLR 2018 | Randomization reference |
| pixmask (each filter family) | Individual filters | Ablation |
| pixmask (full pipeline) | All filters composed | Proposed system |

### Fair Comparison Protocol

1. Fix attack: same attacker, same ε, same number of steps (PGD-40) for all baselines.
2. Fix quality target: tune each baseline so SSIM ≥ 0.90 on clean ImageNet — then compare robustness at matched quality.
3. Report both robust accuracy and clean accuracy in a single table. Do not cherry-pick one.
4. For throughput comparison: same hardware, same image resolution, single-threaded latency, then multi-threaded throughput.

### Comparison Table Template

```
Method        | Clean Acc | PGD-40 Acc | C&W Acc | PSNR↑  | SSIM↑  | LPIPS↓ | Latency (ms) |
No defense    | 76.1%     | 1.2%       | 0.3%    | ∞      | 1.000  | 0.000  | 0            |
JPEG q=75     | 75.4%     | 42.1%      | 15.3%   | 38.2   | 0.961  | 0.041  | 2.1          |
Gaussian σ=1  | 74.8%     | 35.2%      | 8.7%    | 35.1   | 0.942  | 0.063  | 0.8          |
Median k=3    | 75.1%     | 38.4%      | 10.2%   | 36.5   | 0.951  | 0.055  | 1.4          |
Bit-depth 5b  | 74.3%     | 47.3%      | 22.1%   | 33.8   | 0.921  | 0.082  | 0.3          |
pixmask (full)| 74.7%     | 61.2%      | 41.5%   | 34.1   | 0.932  | 0.071  | 3.2          |
```

---

## 9. VLM-Specific Evaluation Extensions

Beyond classification, pixmask targets VLM pipelines. Extra evaluation points:

### Jailbreak ASR on MM-SafetyBench

- 13 scenarios: illegal activities, hate speech, malware, physical harm, fraud, privacy, adult content, etc.
- For each scenario: report ASR before/after pixmask, using LLaVA-1.6 and InternVL2-8B as victim models.
- Use keyword-based refusal detection + GPT-4o judge for ambiguous cases.

### Scaling Attack Evaluation

Quiring et al. (2020) showed images can be crafted to show different content at display vs. model-input resolution. Evaluation:
- Generate scaling attack images for bilinear, bicubic, area interpolation.
- Measure whether pixmask's random interpolation defense neutralizes content mismatch.
- Metric: "content divergence" — does the model at native vs. downscaled resolution predict the same class/caption?

### Embedding-Space Attacks

Attacks that craft perturbations in embedding space (e.g., attacking CLIP vision encoder directly) rather than pixel space. Test whether pixel-level sanitization disrupts embedding-targeted attacks. Metric: cosine similarity of sanitized vs. original image embeddings.

---

## 10. Statistical Rigor

- Report mean ± standard deviation across at least 3 independent runs.
- For detection-based metrics (TPR/FPR), report AUC-ROC over threshold sweep.
- For VLM experiments: minimum 500 adversarial samples per attack type for ASR estimates to be stable (±3% at 95% CI).
- For latency: 1000+ timed iterations after warmup (first 100 discarded). Report p50, p95, p99 — not mean, which is skewed by outliers.
- Hardware context is mandatory: CPU model, SIMD support, RAM, OS, compiler version, optimization flags.

---

## References

- Xu, Evans, Qi — "Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks", NDSS 2018
- Das et al. — "SHIELD: Fast, Practical Defense and Vaccination for Deep Learning", KDD 2018
- Guo et al. — "Countering Adversarial Images Using Input Transformations", ICLR 2018
- Xie et al. — "Mitigating Adversarial Effects Through Randomization", ICLR 2018
- Athalye, Carlini, Wagner — "Obfuscated Gradients Give a False Sense of Security", ICML 2018
- Madry et al. — "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018
- Carlini & Wagner — "Towards Evaluating the Robustness of Neural Networks", S&P 2017
- Zhang et al. — "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric", CVPR 2018 (LPIPS)
- Croce et al. — "RobustBench: A Standardized Adversarial Robustness Benchmark", NeurIPS 2021
- Zou et al. — "Universal and Transferable Adversarial Attacks on Aligned Language Models", 2023 (GCG / AdvBench)
- Liu et al. — "MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models", 2023
- Quiring et al. — "Adversarial Preprocessing: Understanding and Preventing Image-Scaling Attacks in Machine Learning", USENIX Security 2020
- Google Benchmark — https://github.com/google/benchmark
