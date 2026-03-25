# Malformed/Corrupt Image File Security

> Research for pixmask — C++ image sanitization layer before multimodal LLMs.
> Covers parser vulnerabilities, format-specific attacks, safe re-encoding, polyglot files, and C++ safe parsing patterns.

---

## 1. Image Parser Vulnerabilities: CVE Landscape (2021–2025)

### 1.1 libwebp

libwebp has had the most severe recent CVEs of any image parsing library, culminating in a critical actively-exploited zero-day.

**CVE-2023-4863 — Heap Buffer Overflow (CVSS 8.8, CRITICAL in practice)**

This is the most consequential image library CVE of the 2020s decade. Root cause: the lossless VP8L decoder's `BuildHuffmanTable` function pre-allocates Huffman table memory using a fixed lookup table derived from Mark Adler's "enough" tool. The allocation assumes certain structural bounds on Huffman trees. An attacker can construct a WebP file with:
- Four maximally-sized valid Huffman tables for alphabet sizes 280 and 256
- A fifth table with a deliberately unbalanced tree (internal nodes lacking children)

The unbalanced tree causes `BuildHuffmanTable` to produce an unexpectedly large key index, triggering `ReplicateValue` to write beyond the allocated buffer. The patch implements a two-pass approach: a first dry-run pass calculates the required size before any memory writes.

Why fuzzing failed to find it pre-disclosure: the trigger requires a specific sequence of 4 valid large Huffman tables before the malformed 5th, making random mutation-based fuzzing astronomically unlikely to hit it.

- Actively exploited in the wild before disclosure (CISA KEV catalog, deadline 2023-10-04)
- Affected: libwebp < 1.3.2, Chrome < 116.0.5845.187, Firefox < 117.0.1, Edge < 117.0.2045.31, Pillow < 10.0.1, Electron (22.x, 24.x–27.x), SkiaSharp, Magick.NET, libwebp-sys (Rust), chai2010/webp (Go)
- Fix: update to libwebp >= 1.3.2

**CVE-2023-1999 — Use-After-Free / Double-Free (CVSS 7.5 HIGH)**

In `ApplyFiltersAndEncode()`: after an Out-of-Memory error in the VP8 encoder, a pointer is still assigned to `trial` when freed, enabling a subsequent double-free. Affects libwebp 0.4.2–1.3.0. Fixed in 1.3.1.

### 1.2 libjpeg-turbo

**CVE-2023-2804 — Heap Buffer Overflow (CVSS 6.5 MEDIUM)**

Location: `h2v2_merged_upsample_internal()` in `jdmrgext.c`. Trigger: a crafted 12-bit lossless JPEG with out-of-range 12-bit samples fed through color quantization or merged chroma upsampling paths. These code paths were never designed for lossless mode. Fix: disable color quantization and merged chroma upsampling for lossless JPEG entirely (libjpeg-turbo 2.1.90).

**CVE-2021-29390 — Heap Over-Read (CVSS 7.1 HIGH)**

Location: `decompress_smooth_data()` in `jdcoefct.c`. Two-byte heap over-read in libjpeg-turbo 2.0.90. Can read adjacent heap memory, risk of minor information disclosure plus crash.

**Additional hardening in libjpeg-turbo 3.x (2023–2024):**
- 3.0.4: Fixed exponential CPU growth from excessive marker processing (algorithmic DoS)
- 3.0.4: Hardened default marker processor against segfaults from malformed SOF segments
- 3.1.1: Protected against erroneous `data_precision` field modifications
- 3.1.4: Protected quantization table handling against zero values; fixed division-by-zero in jpegtran

### 1.3 libpng

**CVE-2025-28162 / CVE-2025-28164 — Buffer Overflow + Memory Leak (CVSS 5.5 MEDIUM)**

Both affect libpng 1.6.43–1.6.46. CVE-2025-28162 is a buffer copy without checking input size (CWE-120). CVE-2025-28164 involves missing release of memory after effective lifetime (CWE-401), causing AddressSanitizer-detected memory leaks leading to unbounded memory growth and process hang. Trigger path: `png_create_read_struct()` with a malformed PNG. Local attack vector, no remote network exposure. Fixed in libpng 1.6.47+.

**Note:** CVE-2022-3857 was rejected as a false positive — the libpng maintainer confirmed no actual flaw existed.

### 1.4 libtiff

libtiff has an extensive CVE history due to its format complexity (TIFF supports ~50+ IFD tag types, multiple compression codecs, tiling, striping, offsets).

**CVE-2023-40745 — Integer Overflow to Heap Buffer Overflow (CVSS 6.5 MEDIUM)**

Integer overflow in a size calculation when processing a crafted TIFF file triggers a heap-based buffer overflow. Remote attacker can cause crash or possibly execute arbitrary code with user interaction. Affects libtiff < 4.6.0. CWE-190.

**CVE-2023-25433 — Heap Buffer Overflow (CVSS 5.5 MEDIUM)**

Location: `tiffcrop.c:8499`. Incorrect buffer size update after `rotateImage()`. Causes segfault. Affects libtiff 4.5.0.

### 1.5 ImageMagick

ImageMagick's most catastrophic incident was ImageTragick (2016), but it illustrates architectural issues still relevant today:

**ImageTragick (CVE-2016-3714 + 4 related) — RCE via Format Delegation**

Core flaw: insufficient shell character filtering in delegate commands. ImageMagick invokes external programs (ghostscript, ffmpeg, etc.) via shell for many formats. A crafted filename with shell metacharacters in an SVG/MVG file could execute arbitrary commands. Related CVEs: SSRF (CVE-2016-3718), arbitrary file deletion (CVE-2016-3715), file movement (CVE-2016-3716), local file disclosure (CVE-2016-3717).

Lesson: **ImageMagick is unsafe to use as a trusted parser** in a security-sensitive context without a strict `policy.xml` disabling all format delegations. Even then, its attack surface (70+ format decoders, external process spawning) is fundamentally incompatible with a defense layer.

ImageMagick's security policy does support resource limits (memory, area, width, height, time, disk) and format allowlisting (e.g., `{GIF,JPEG,PNG,WEBP}` only), which can reduce but not eliminate risk.

---

## 2. Image Format Security: Attack Surface by Format

### 2.1 PNG

**Structure:** 8-byte magic signature `\x89PNG\r\n\x1a\n`, followed by chunks. Each chunk: 4-byte length, 4-byte type, data, 4-byte CRC-32. Critical chunks: IHDR (must be first), IDAT (must be consecutive), IEND (must be last), PLTE (required for indexed color).

**Valid IHDR dimensions:** 0 is invalid; maximum is `2^31 - 1` (per spec, 4-byte signed integer). Practical parsers should enforce much lower limits (e.g., 65535 x 65535 maximum, configurable).

**Valid bit depth / color type combinations:**
| Color Type | Allowed Bit Depths |
|---|---|
| 0 (grayscale) | 1, 2, 4, 8, 16 |
| 2 (truecolor) | 8, 16 |
| 3 (indexed) | 1, 2, 4, 8 |
| 4 (grayscale+alpha) | 8, 16 |
| 6 (truecolor+alpha) | 8, 16 |

**Attack vectors:**

1. **Decompression bombs (IDAT).** IDAT chunks are DEFLATE-compressed. DEFLATE can achieve compression ratios of 1000:1 or more for repetitive data. A PNG claiming dimensions of 30000x30000 RGBA (3.6 GB uncompressed) compresses to a few kilobytes. Without a pre-parse dimension check before initiating decompression, a parser will attempt to allocate gigabytes of memory. The IHDR chunk appears before IDAT, so the correct defense is: parse IHDR first, validate `width * height * channels * bytes_per_channel` against a hard memory budget, and refuse before touching IDAT. libpng has a `png_set_user_limits()` API for this.

2. **Integer overflow in dimension arithmetic.** `width=65537, height=65537, channels=4` → `65537 * 65537 * 4 = 17,180,000,004` which overflows a 32-bit `uint32_t` to `884,999,908`. This causes under-allocation followed by heap overflow writes. Must use `uint64_t` or `__int128` for intermediate dimension product checks, or use checked-multiply primitives.

3. **Invalid CRC.** The spec mandates CRC-32 validation over chunk type + data. Most parsers tolerate CRC failures silently. An attacker can modify chunk data after computing a valid CRC to confuse cached-checksum logic, or inject corrupted ancillary chunks. Strict parsers must treat CRC failure as hard parse failure.

4. **Chunk ordering violations.** IHDR not first, IDAT chunks not consecutive (interspersed with other chunks), multiple IHDR chunks, IEND not last. Lenient parsers attempt recovery and may enter undefined states. Safari/macOS WebKit historically was stricter than most; most implementations tolerate reordering.

5. **Oversized ancillary chunks.** tEXt, zTXt, iTXt chunks can contain arbitrarily large text data. A single tEXt chunk with a 100MB "comment" causes memory exhaustion in parsers that buffer the entire chunk. Limit ancillary chunk sizes or skip them entirely.

6. **Web shells in IDAT.** Appending PHP/code after the PNG IDAT DEFLATE stream creates files that are valid PNGs but also contain executable content. Re-encoding strips this — the decoded pixel data is re-compressed from scratch, discarding all trailing data.

### 2.2 JPEG

**Structure:** Marker-segment format. SOI (`FF D8`) + sequences of `FF XX LL LL DATA` segments + EOI (`FF D9`). The `FF` byte in entropy-coded data is stuffed as `FF 00`. There is no fixed-length segment for the entropy-coded scan data (ECS); it runs until the next valid `FF XX` marker.

**Key markers:**
- SOF0/SOF1/SOF2 (`FF C0/C1/C2`): Start-of-Frame (baseline/extended/progressive); contains width, height, number of components, precision
- DHT (`FF C4`): Define Huffman Table
- DQT (`FF DB`): Define Quantization Table
- SOS (`FF DA`): Start-of-Scan; precedes entropy-coded data
- APP0–APP15 (`FF E0`–`FF EF`): Application metadata (JFIF, EXIF, XMP, ICC profiles)
- COM (`FF FE`): Comment segment

**Attack vectors:**

1. **Malformed Huffman tables (DHT).** Huffman code lengths that sum to an invalid tree, or that declare more symbols than the alphabet permits. CVE-2023-4863 in libwebp was triggered via this exact mechanism. libjpeg historically had issues with crafted DHT entries causing heap overflows.

2. **Truncated streams.** JPEG files that terminate mid-scan cause parsers that don't handle EOF in entropy decoding to read past end-of-buffer. libjpeg's `jpeg_finish_decompress()` must be called even on truncated inputs to trigger proper cleanup.

3. **Invalid precision in SOF.** JPEG supports 8-bit and 12-bit precision. Most implementations only handle 8-bit. CVE-2023-2804 in libjpeg-turbo was triggered by 12-bit lossless input reaching code paths that assumed 8-bit.

4. **Algorithmic DoS via excessive markers.** libjpeg-turbo 3.0.4 fixed exponential CPU growth from a JPEG with an excessive number of APP/COM markers. A parser iterating all markers in O(n) is acceptable; one that does O(n^2) work (e.g., reprocessing from the start on each marker) creates a DoS vector. A limit of ~64 or 128 APP markers is reasonable.

5. **JPEG/ZIP polyglots.** ZIP files begin with a local file header at offset 0 (`PK\x03\x04`). JPEG begins with `FF D8 FF`. The JPEG APP1 segment (`FF E1 LL LL`) can contain arbitrary data. By placing a ZIP local file header at the start of the APP1 payload, the file is simultaneously a valid JPEG (the ZIP bytes are opaque application data to the JPEG parser) and a valid ZIP (ZIP parsers scan for the End-of-Central-Directory record from the end). This is the classical JPEG/ZIP polyglot. See portswigger.net JPEG/JS polyglot: JPEG initial bytes `FF D8 FF E0` form a valid non-ASCII JS variable; the APP0 segment length can be set to `0x2F2A` which is `/*` in ASCII, opening a JS block comment. Mitigation: always reparse; don't trust extension alone.

6. **EXIF/XMP injection.** APP1 segments contain EXIF data (TIFF-formatted IFDs) and XMP (XML). Both have their own parser attack surfaces. EXIF can contain pointers and offsets; malformed EXIF can cause out-of-bounds reads. XMP is XML and subject to XXE if parsed with an external-entity-resolving XML parser.

### 2.3 WebP

**Structure:** RIFF container. 4-byte "RIFF", 4-byte file size, 4-byte "WEBP", then chunks. Main chunk types: VP8 (lossy), VP8L (lossless VP8 Lossless), VP8X (extended format with alpha, animation, EXIF, XMP).

**Attack vectors:**

1. **CVE-2023-4863 Huffman overflow** (described in section 1.1). The root format issue: VP8L's Huffman coding scheme allows configurations not covered by the pre-computed allocation table. Any validator must enforce strict Huffman table structural constraints before allocation.

2. **VP8X extended format complexity.** VP8X adds ANIM (animation), ALPH (alpha plane), EXIF, XMP chunks. Each is a separate parsing context. More surface area = more attack surface. The ALPH chunk uses its own filtering scheme.

3. **Dimension extraction before decode.** `WebPGetInfo()` and `WebPGetFeatures()` provide dimension and feature information without full decoding — use these for pre-validation. `WebPGetFeatures()` returns width, height, has_alpha, has_animation, format.

### 2.4 GIF

**Structure:** `GIF87a` or `GIF89a` header, logical screen descriptor, optional global color table, then a sequence of extension blocks and image descriptors, each image followed by LZW-compressed data, terminated by `;` (`0x3B`).

**LZW decompression attacks:**

GIF uses LZW compression with variable code sizes (2–12 bits). Attack vectors:

1. **LZW bombs.** Similar to DEFLATE bombs for PNG. A GIF image with very small pixel dimensions declared in the image descriptor but an LZW stream that decompresses to a large amount of data (the LZW decoder fills a scratch buffer). Modern GIF parsers cap the decompressed output size against the declared canvas dimensions.

2. **stb_image GIF parser vulnerabilities (open 2025):**
   - **Issue #1838**: Double-free when stride is 0, causing `realloc(ptr, 0)` to return `NULL` and error handler frees the wrong pointer. 25 distinct crash conditions found via fuzzing, 84% double-free. Minimal 60-byte GIF triggers it reliably.
   - **Issue #1916**: Heap-buffer-overflow read in `stbi__gif_load_next`. The `two_back` pointer calculated as `out - 2 * stride` points before the allocated buffer instead of `out + (layers - 2) * stride`. Triggered by GIF disposal method 3.
   - **Status**: Open as of the research date (March 2026). stb_image GIF parsing should be considered unsafe for untrusted input.

3. **Multi-frame GIF resource exhaustion.** GIF89a supports animation (multiple frames). A file can declare thousands of frames each requiring separate allocation. Limit maximum frame count (e.g., 256 frames).

### 2.5 TIFF

TIFF is the most complex raster format in common use. It supports: multiple compression codecs (none, LZW, JPEG, Deflate, PackBits, CCITT, JBIG, etc.), tiling and striping, arbitrary IFD tag types, multiple sub-files, big-endian and little-endian variants, BigTIFF (64-bit offsets).

**Attack surface:**

1. **IFD pointer chains.** TIFF's Image File Directory is a linked list; each IFD contains a pointer to the next. Crafted TIFFs can create circular IFD chains, causing infinite loops. libtiff historically crashed on circular IFDs; modern versions track visited offsets.

2. **Integer overflow in dimension/offset arithmetic.** CVE-2023-40745: integer overflow in a size calculation triggers heap buffer overflow. TIFF supports `BitsPerSample` per-channel and complex planar configurations; dimension arithmetic is correspondingly complex with many multiplication paths.

3. **Compression codec delegation.** TIFF/JPEG embeds a full JPEG stream within a TIFF container; the TIFF parser delegates to libjpeg. TIFF/Deflate delegates to zlib. Each codec adds its own attack surface and resource consumption.

4. **Large offset attacks.** TIFF allows data to be stored anywhere in the file by absolute offset. A crafted TIFF can point strip/tile offsets to locations far beyond the file end, causing out-of-bounds reads.

5. **Recommendation for pixmask:** Reject TIFF input entirely, or transcode to PNG/JPEG immediately using a sandboxed process. The attack surface is too large for inline trust.

### 2.6 SVG

SVG is XML-based vector graphics, not pixel-based. It is not safe to pass to any image decoder used as a defense layer. Key risks:

1. **XXE (XML External Entity).** SVG files can contain DOCTYPE declarations with external entity references: `<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>`. Any XML parser that resolves external entities will read local files or make network requests (SSRF).

2. **Script injection.** SVG supports embedded `<script>` elements and event handlers (`onload`, `onclick`, etc.). If the SVG is rendered in a browser context (e.g., served as `image/svg+xml`), it executes JavaScript.

3. **CSS/style injection.** `@import` rules in SVG `<style>` elements can trigger external resource loads.

**Recommendation:** Reject SVG at the format detection stage. Do not attempt to parse or render SVG. Return an error if the magic bytes or MIME type indicate SVG.

---

## 3. Safe Image Re-Encoding as Defense

### 3.1 The Decode-Validate-Reencode Pipeline

The canonical defense against malformed image attacks is a three-phase pipeline:

```
[raw bytes] → [strict decode] → [validated pixel buffer] → [re-encode to canonical format]
```

What this strips:
- All metadata (EXIF, XMP, ICC profiles, comments, tEXt chunks)
- All ancillary/unknown chunks
- Steganographic payloads in format-specific fields
- Web shells in trailing data (IDAT post-stream, JPEG after EOI)
- Polyglot dual-format payloads (ZIP header in JPEG APP1, etc.)
- Format-specific attack state (crafted Huffman tables, malformed IFDs)

After re-encoding, the output is pixel data reconstructed from scratch into a well-formed container. The encoder cannot reproduce malformed input structures because it operates on the decoded pixel array, not the input bytes.

**What re-encoding does NOT strip:**
- Adversarial pixel patterns (the subject of pixmask's primary defense pipeline)
- Steganography embedded at the semantic pixel level (e.g., LSB steganography — the pixel values themselves encode hidden data)
- Very large images (must validate dimensions before decoding)

### 3.2 What Constitutes a "Strict" Parser

A strict parser enforces:

1. **Signature validation.** Verify magic bytes at the declared offset before any further parsing. Do not rely on file extension.

2. **Dimension bounds.** Reject width or height of 0, or exceeding a configurable maximum (recommendation: 16384 for each dimension in LLM preprocessing; this accommodates high-res but prevents trivial memory exhaustion). Check `width * height * channels * bytes_per_channel` against an absolute byte limit (recommendation: 512 MB).

3. **Integer overflow prevention.** All dimension arithmetic in `uint64_t` with explicit overflow checks before casting to `size_t` for allocation. Use `__builtin_mul_overflow` or `std::numeric_limits` checks.

4. **Decompression ratio limit.** For DEFLATE (PNG IDAT, TIFF/Deflate) and LZW (GIF, TIFF/LZW): track ratio of compressed input bytes consumed vs. decompressed output bytes produced. Abort if ratio exceeds threshold (recommendation: 1:1000 or configured value). This catches decompression bombs.

5. **Format-specific structure validation.** PNG: verify chunk CRCs, enforce IHDR-first and IEND-last ordering, reject unknown critical chunks. JPEG: enforce valid marker sequence, limit APP marker count, reject invalid precision values. WebP: use `WebPGetFeatures()` before `WebPDecode()` for pre-validation.

6. **Hard time limit.** Parsing must complete within a wall-clock limit (e.g., 5 seconds). Use `alarm()`/`setitimer()` or a watchdog thread.

7. **Memory limit enforcement.** Parsing must not allocate more than a configurable memory budget. On Linux, use `setrlimit(RLIMIT_AS, ...)` in a forked process, or track allocations with a custom allocator.

### 3.3 Memory Limits and Dimension Limits: Recommended Values

| Parameter | Recommended Limit | Rationale |
|---|---|---|
| Max width | 16384 px | 16K is the JPEG/PNG practical ceiling; LLM input is rarely > 4096 |
| Max height | 16384 px | Same |
| Max decoded bytes | 512 MB | `16384 * 16384 * 4 = 1 GB`; use 512 MB to leave headroom |
| Max compression ratio | 1:1000 | Well above legitimate image ratios (~1:5 to 1:50) |
| Max file size (input) | 64 MB | Most LLM vision models accept <<10 MB; 64 MB is generous |
| Max JPEG APP markers | 128 | Prevents algorithmic DoS from excessive marker scanning |
| Max GIF frames | 256 | Reasonable animation limit |
| Max PNG ancillary chunk size | 1 MB | Prevents tEXt/zTXt memory exhaustion |
| Parse timeout | 5 s | Hard deadline prevents infinite-loop exploits |

---

## 4. Polyglot Files

Polyglot files are valid under two (or more) different file format parsers simultaneously. They exploit the fact that most formats have "don't-care" regions where extra data is tolerated.

### 4.1 JPEG/ZIP Polyglot

Construction mechanism:
1. JPEG begins with SOI (`FF D8`), then APP segments. APP1 (`FF E1 LL LL DATA`) allows arbitrary data in the payload.
2. ZIP parsers locate the End-of-Central-Directory (EOCD) record by scanning backward from the file end (signature `PK\x05\x06`). The central directory and local file headers can be embedded within JPEG's APP segment data.
3. The resulting file: a JPEG parser sees a valid image (APP1 payload is opaque); a ZIP parser finds valid ZIP structure.

Use in attacks:
- Upload filter bypass: file passes image validation, is served as an image, but a ZIP-aware component also processes the same bytes as an archive.
- Content Security Policy bypass: the file executes as a script when loaded with appropriate MIME type (see JPEG/JS polyglot above).

### 4.2 PNG/Trailing-Data Polyglots

As demonstrated by `tweetable-polyglot-png`: many platforms (Twitter, Imgur, GitHub, Discord) strip PNG metadata chunks but preserve trailing DEFLATE data. A PNG can contain a valid ZIP, PDF, or MP3 after the IEND chunk and some platforms will serve it intact. Detection requires checking for data beyond IEND.

Specifically: after the IEND chunk (`00 00 00 00 49 45 4E 44 AE 42 60 82`), there should be no further bytes. Any bytes after IEND are non-standard and should be flagged or truncated.

### 4.3 JPEG/JavaScript Polyglot

The PortSwigger technique: JPEG bytes `FF D8 FF E0` form a valid non-ASCII JavaScript variable when interpreted as Latin-1. By crafting the APP0 segment length to encode `/*` (`0x2F 0x2A`), the parser opens a JavaScript block comment that contains the rest of the JPEG binary data. A closing `*/` at the end of the JPEG's APP0 payload terminates the comment, and JavaScript code follows. Firefox (patched), Safari, Edge, and IE11 historically executed such files when loaded via `<script charset="ISO-8859-1">`. Chrome did not.

### 4.4 SVG/HTML Polyglots

SVG is a subset of XML and structurally compatible with HTML5 `<svg>` elements. SVG files containing `<script>` are simultaneously valid SVG images and XSS payloads when served with permissive MIME types. **Reject SVG at the format detection stage.**

### 4.5 Polyglot Detection Strategies

1. **Magic byte validation (necessary, not sufficient).** Verify the first N bytes match the expected format signature exactly. This rules out trivial misclassification but does not catch polyglots where the primary magic bytes are legitimate.

2. **Tail-of-file scanning.** After identifying the expected end-of-file marker (JPEG `FF D9`, PNG IEND, GIF `;`), verify no significant data follows. Recommended: flag any file with >16 bytes after the end marker.

3. **Structural completeness check.** Parse the entire file and verify all offsets, chunk sizes, and segment lengths account for every byte. Unaccounted regions are potential embedded payloads.

4. **Re-encoding as the definitive defense.** A re-encoded output is constructed from the pixel array alone. It cannot contain embedded ZIP data, trailing JavaScript, or any non-pixel payload. Re-encoding is the only complete polyglot defense.

5. **Multi-format parse.** Attempt to parse the input file with more than one format parser (e.g., "does this JPEG also parse as a valid ZIP?"). Reject if any secondary format succeeds. This is expensive but thorough.

**Magic byte reference:**
| Format | Magic Bytes (hex) | Offset |
|---|---|---|
| PNG | `89 50 4E 47 0D 0A 1A 0A` | 0 |
| JPEG | `FF D8 FF` | 0 |
| WebP | `52 49 46 46 ?? ?? ?? ?? 57 45 42 50` | 0 (RIFF + WEBP at +8) |
| GIF87a | `47 49 46 38 37 61` | 0 |
| GIF89a | `47 49 46 38 39 61` | 0 |
| TIFF (LE) | `49 49 2A 00` | 0 |
| TIFF (BE) | `4D 4D 00 2A` | 0 |
| BMP | `42 4D` | 0 |
| SVG/XML | `3C 3F 78 6D 6C` or `3C 73 76 67` | 0 |
| ZIP | `50 4B 03 04` | 0 |
| PDF | `25 50 44 46` | 0 |

---

## 5. C++ Safe Image Parsing

### 5.1 stb_image

stb_image is a popular single-header C library (`stb_image.h`). It is convenient but has known security issues.

**Safety properties:**
- Integer overflow helpers: `stbi__addsizes_valid`, `stbi__mul2sizes_valid`, `stbi__mad2sizes_valid`, `stbi__mad3sizes_valid`, `stbi__mad4sizes_valid` — these check for overflow before allocation.
- Format-specific header checks: `stbi__check_png_header`, `stbi__parse_entropy_coded_data` (JPEG), `stbi__gif_header`.

**Known open vulnerabilities (as of March 2026):**

| Issue | Type | Status | Description |
|---|---|---|---|
| #1838 | Double-free (GIF) | Open | `stride=0` causes `realloc(ptr, 0)` → null, error handler frees wrong pointer. 25 crash variants. |
| #1916 | OOB read (GIF) | Open | `two_back = out - 2 * stride` points before buffer. Triggered by disposal method 3. |
| #1860 | Heap overflow (PNG 16-bit) | PR open | `stbi__convert_format16` allocates by source channels, writes by output channels. |
| #1915 | OOB read (GIF multi-frame) | Not planned | Out-of-bounds access in multi-frame GIF handling. |
| #1921 | Null deref (format conversion) | Not planned | NULL pointer dereference in `stbi__convert_format`. |

**Assessment:** stb_image is not suitable as the primary trust boundary for untrusted input in a security-critical context. Its GIF parser in particular is actively broken for adversarial inputs. If used, restrict input formats to PNG and JPEG only (avoid GIF via stb_image). Combine with pre-validation using a separate strict validator.

### 5.2 lodepng

lodepng is a single-file C/C++ PNG encoder/decoder. Its security properties are stronger than stb_image for PNG specifically:

**Safety properties (from code review):**
- Overflow detection: `lodepng_addofl()`, `lodepng_mulofl()`, `lodepng_gtofl()` — explicit overflow checks on every size computation.
- Pre-allocation validation: `lodepng_get_raw_size_lct()`, `lodepng_pixel_overflow()` — compute and check required size before any allocation.
- Bit reader bounds: `ensureBits9/17/25/32()` — validate available data before reading compressed bitstream.
- LZ77 validation: `addLengthDistance()` validates backward reference validity.
- CERT C compliance claim.

**Assessment:** lodepng has stronger security architecture than libpng for the specific task of PNG decoding from untrusted sources. Its checked-arithmetic approach and dimension pre-validation make it substantially harder to exploit with dimension-overflow or decompression-bomb attacks.

### 5.3 libspng

libspng is a modern C PNG decoder designed with security as a primary goal:

- Follows the CERT C Coding Standard; all integer arithmetic is overflow-checked.
- Continuous fuzzing via Google OSS-Fuzz.
- Static analysis with Clang Static Analyzer, PVS-Studio, and Coverity.
- 1000+ test cases including adversarial inputs.
- API: `spng_get_image_info()` for pre-decode dimension check; `spng_decode_image()` for full decode.
- Hard limit: refuses images exceeding 4 GB per row.
- Claims no publicly known security vulnerabilities.

**Assessment:** libspng is the recommended PNG decoder for security-sensitive contexts as of 2025–2026.

### 5.4 Memory-Safe Parsing Patterns in C++

**Pattern 1: Validate dimensions before allocating decode buffer**

```cpp
// Get image info without decoding
uint32_t w, h;
if (!parser_get_info(data, size, &w, &h)) return Error::InvalidFormat;

// Overflow-safe byte budget check
constexpr uint64_t kMaxDecodedBytes = 512ULL * 1024 * 1024; // 512 MB
constexpr uint32_t kMaxDimension = 16384;
if (w == 0 || h == 0 || w > kMaxDimension || h > kMaxDimension) return Error::DimensionLimit;

uint64_t bytes_needed;
if (__builtin_mul_overflow((uint64_t)w, (uint64_t)h, &bytes_needed)) return Error::Overflow;
if (__builtin_mul_overflow(bytes_needed, 4ULL, &bytes_needed)) return Error::Overflow; // RGBA
if (bytes_needed > kMaxDecodedBytes) return Error::MemoryLimit;

// Only now allocate and decode
```

**Pattern 2: Decompression ratio watchdog**

```cpp
// Wrap the decompressor's output stream
class RatioLimitedDecompressor {
    size_t compressed_in_ = 0;
    size_t decompressed_out_ = 0;
    static constexpr size_t kMaxRatio = 1000;
public:
    bool feed(const uint8_t* in, size_t in_len, uint8_t* out, size_t* out_len) {
        compressed_in_ += in_len;
        // ... decompress ...
        decompressed_out_ += *out_len;
        if (compressed_in_ > 0 && decompressed_out_ / compressed_in_ > kMaxRatio) {
            return false; // decompression bomb
        }
        return true;
    }
};
```

**Pattern 3: Post-parse end-of-file polyglot check**

```cpp
// After successfully parsing a PNG, verify IEND is the last chunk
size_t iend_offset = find_iend(data, size);
if (iend_offset == npos) return Error::MissingIEND;
// IEND = 00 00 00 00 49 45 4E 44 AE 42 60 82 (12 bytes total)
size_t expected_eof = iend_offset + 12;
if (expected_eof < size && (size - expected_eof) > 16) {
    return Error::TrailingData; // potential polyglot
}
```

**Pattern 4: Subprocess isolation (Chromium Rule of 2)**

Chromium's Rule of 2: do not combine (1) untrustworthy input + (2) unsafe language (C/C++) + (3) high privilege. Image parsing involves all three. The recommended mitigation: parse in a sandboxed subprocess with:
- No network access
- No filesystem write access
- `RLIMIT_AS` set to decode budget + parser overhead (e.g., 1 GB)
- `RLIMIT_CPU` set (e.g., 5 seconds)
- Communication via shared memory or pipe (no file descriptors)

In practice for pixmask: fork a child process for the decode step with resource limits, then pass the validated pixel buffer back via shared memory.

**Pattern 5: Strict format allowlist**

```cpp
enum class ImageFormat { PNG, JPEG, WEBP, UNKNOWN };

ImageFormat detect_format(const uint8_t* data, size_t size) {
    if (size >= 8 &&
        data[0] == 0x89 && data[1] == 'P' && data[2] == 'N' && data[3] == 'G' &&
        data[4] == 0x0D && data[5] == 0x0A && data[6] == 0x1A && data[7] == 0x0A)
        return ImageFormat::PNG;
    if (size >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF)
        return ImageFormat::JPEG;
    if (size >= 12 &&
        data[0]=='R' && data[1]=='I' && data[2]=='F' && data[3]=='F' &&
        data[8]=='W' && data[9]=='E' && data[10]=='B' && data[11]=='P')
        return ImageFormat::WEBP;
    return ImageFormat::UNKNOWN; // reject GIF, TIFF, SVG, BMP, etc.
}
```

### 5.5 Validation Checklist for pixmask Input Stage

Every image entering the pixmask pipeline should pass the following checks before any decode or defense transform:

1. **File size guard.** Reject if `size > 64 MB`.
2. **Magic byte validation.** Detect format from bytes; reject UNKNOWN, SVG, TIFF, GIF (or restrict to PNG + JPEG + WebP only).
3. **Dimension pre-check.** Use format-specific "get info without decode" API (`WebPGetInfo`, `spng_get_image_info`, JPEG SOF0 scan). Reject if `w=0`, `h=0`, `w > 16384`, `h > 16384`, or `w*h*4 > 512 MB`.
4. **Decompression budget.** Enforce zlib/DEFLATE decompression ratio limit for PNG.
5. **Re-encode to canonical format.** Decode to raw RGBA pixel buffer → re-encode to PNG (lossless) or JPEG (if lossy acceptable). This strips all metadata, ancillary data, and polyglot payloads.
6. **Post-IEND/EOI check.** After re-encoding, verify output has no trailing data (output is clean by construction; the concern is the input stage before re-encode).
7. **SVG rejection.** If input file contains `<svg`, `<?xml`, or `<!DOCTYPE`, reject immediately.
8. **Subprocess isolation.** Perform steps 1–6 in a sandboxed child process with `RLIMIT_AS` and `RLIMIT_CPU`.

---

## 6. Summary: Risk Matrix for pixmask

| Threat | Severity | Likelihood | Mitigation |
|---|---|---|---|
| Decompression bomb (PNG IDAT) | Critical | High | Dimension pre-check + DEFLATE ratio limit |
| CVE-2023-4863 class (libwebp Huffman) | Critical | Medium | Vendor-patched lib + pre-validation via `WebPGetFeatures` |
| Integer overflow in width*height | High | High | Checked arithmetic in `uint64_t` before allocation |
| JPEG algorithmic DoS (excessive markers) | Medium | Medium | Limit APP marker count; use libjpeg-turbo >= 3.0.4 |
| stb_image GIF double-free | High | High | Disable GIF input or use different GIF parser |
| TIFF IFD chain attacks | High | Medium | Reject TIFF input entirely |
| SVG XXE / script injection | Critical | High | Reject SVG by magic bytes |
| JPEG/ZIP polyglot | Medium | Low | Re-encode strips ZIP payload |
| PNG trailing-data polyglot | Low | Medium | Check for data after IEND; re-encode strips it |
| Malformed EXIF/XMP in JPEG APP1 | Medium | Medium | Re-encode strips APP segments |
| ImageMagick delegate RCE | Critical | High | Do not use ImageMagick as trusted parser |
| Memory exhaustion (oversized ancillary chunks) | Medium | Medium | Limit ancillary chunk size in PNG; reject large APP segments |

---

## References

- CVE-2023-4863: https://nvd.nist.gov/vuln/detail/CVE-2023-4863
- isosceles.com CVE-2023-4863 analysis: https://blog.isosceles.com/the-webp-0day/
- CVE-2023-1999 (libwebp UAF): https://nvd.nist.gov/vuln/detail/CVE-2023-1999
- CVE-2023-2804 (libjpeg-turbo): https://nvd.nist.gov/vuln/detail/CVE-2023-2804
- CVE-2021-29390 (libjpeg-turbo): https://nvd.nist.gov/vuln/detail/CVE-2021-29390
- CVE-2025-28162 / CVE-2025-28164 (libpng): https://nvd.nist.gov/vuln/detail/CVE-2025-28162
- CVE-2023-40745 (libtiff integer overflow): https://nvd.nist.gov/vuln/detail/CVE-2023-40745
- CVE-2023-25433 (libtiff tiffcrop): https://nvd.nist.gov/vuln/detail/CVE-2023-25433
- ImageTragick: https://imagetragick.com/
- stb_image GIF issues: https://github.com/nothings/stb/issues/1838, /1916, /1860
- lodepng: https://github.com/lvandeve/lodepng
- libspng: https://libspng.org/
- PNG spec (RFC 2083): https://www.rfc-editor.org/rfc/rfc2083
- PNG chunk quirks: https://raw.githubusercontent.com/corkami/formats/master/image/png.md
- JPEG structure: https://raw.githubusercontent.com/corkami/formats/master/image/jpeg.md
- WebP API: https://developers.google.com/speed/webp/docs/api
- JPEG polyglot (PortSwigger): https://portswigger.net/research/bypassing-csp-using-polyglot-jpegs
- PNG trailing-data polyglot: https://github.com/DavidBuchanan314/tweetable-polyglot-png
- SVG XXE: https://portswigger.net/web-security/xxe/lab-xxe-via-file-upload
- Chromium Rule of 2: https://chromium.googlesource.com/chromium/src/+/refs/heads/main/docs/security/rule-of-2.md
- OWASP File Upload Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html
- ImageMagick security policy: https://imagemagick.org/script/security-policy.php
- libjpeg-turbo changelog: https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/ChangeLog.md
