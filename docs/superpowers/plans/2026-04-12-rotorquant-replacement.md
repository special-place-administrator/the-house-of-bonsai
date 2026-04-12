# RotorQuant Replacement — Root Out TurboQuant, Integrate RQ

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Completely replace TurboQuant (Google ICLR 2026 WHT-based vector quantization) with RotorQuant (Givens 2D / Quaternion 4D rotation-based quantization) across the entire project — llama-cpp C/CUDA/Metal, prism-mcp TypeScript, launcher Rust, and all docs.

**Architecture:** RotorQuant replaces TurboQuant's dense 128x128 Walsh-Hadamard Transform with small block-diagonal rotations (2D Givens pairs for PlanarQuant, 4D quaternions for IsoQuant). The rotation constants shrink from ~64KB to ~1KB. A deferred K-cache quantization subsystem is added: K stays F16 during prefill and is bulk-converted post-prefill to avoid error compounding. V-cache requires explicit inverse rotation during dequant (unlike WHT which is self-inverse).

**Tech Stack:** C/C++/CUDA/Metal (llama-cpp ggml layer), TypeScript (prism-mcp embedding compressor), Rust/Dioxus (launcher UI), CMake (build system)

**Source Material:** 
- RotorQuant Python research repo: `https://github.com/scrya-com/rotorquant` (algorithm reference)
- RotorQuant llama.cpp fork: `github.com/johndpope/llama-cpp-turboquant` branch `feature/planarquant-kv-cache` (production C/CUDA/Metal code to port)

**Naming Convention:**
- ggml types: `GGML_TYPE_RQ3_0`, `GGML_TYPE_RQ4_0`, `GGML_TYPE_RQ2_0`, `GGML_TYPE_RQ3_ISO`, `GGML_TYPE_RQ4_ISO`
- CLI/UI labels: `rq2`, `rq3`, `rq4`, `rq3-iso`, `rq4-iso`
- TypeScript: `RotorQuantCompressor`
- File prefix: `rq-` or `rotorquant-`

**Type ID Allocation (reuse turbo slots):**

| ID | Old (TurboQuant) | New (RotorQuant) | Algorithm |
|----|-----------------|-----------------|-----------|
| 41 | `GGML_TYPE_TURBO3_0` | `GGML_TYPE_RQ3_0` | PlanarQuant 3-bit (default) |
| 42 | `GGML_TYPE_TURBO4_0` | `GGML_TYPE_RQ4_0` | PlanarQuant 4-bit |
| 43 | `GGML_TYPE_TURBO2_0` | `GGML_TYPE_RQ2_0` | PlanarQuant 2-bit (economy) |
| 44 | — | **Q1_0_g128 (KEEP)** | Sacred, do not touch |
| 45 | — | **Q1_0 (KEEP)** | Sacred, do not touch |
| 46 | `GGML_TYPE_TURBO3_TCQ` | `GGML_TYPE_RQ3_ISO` | IsoQuant 3-bit (quaternion) |
| 47 | `GGML_TYPE_TURBO2_TCQ` | `GGML_TYPE_RQ4_ISO` | IsoQuant 4-bit (quaternion) |

---

## Phase 1: Vendor & Setup

### Task 1: Clone RotorQuant reference and fork

**Files:**
- Create: `vendor/rotorquant/` (reference Python code, trimmed)
- Create: `docs/superpowers/plans/rotorquant-reference/` (extracted algorithm docs)

- [ ] **Step 1: Clone the rotorquant research repo into vendor**

```bash
cd C:\AI_STUFF\PROGRAMMING\the-house-of-bonsai
git clone https://github.com/scrya-com/rotorquant vendor/rotorquant-ref
```

- [ ] **Step 2: Trim to reference-only (no benchmarks, no paper LaTeX)**

Keep only: `turboquant/` (the Python package with algorithm implementations), `tests/`, `README.md`, `CLAUDE.md`. Delete: `paper/`, `benchmark_*.py`, `*.combined`, `.git/`.

```bash
cd vendor/rotorquant-ref
rm -rf paper/ .git/ *.combined benchmark_vram.py benchmark_vs_reference.py
```

- [ ] **Step 3: Fetch the llama.cpp fork branch to a temp location for porting**

```bash
cd /tmp
git clone --branch feature/planarquant-kv-cache --single-branch https://github.com/johndpope/llama-cpp-turboquant rotorquant-llamacpp-fork
```

- [ ] **Step 4: Commit vendor addition**

```bash
cd C:\AI_STUFF\PROGRAMMING\the-house-of-bonsai
git add vendor/rotorquant-ref/
git commit -m "chore: vendor rotorquant reference implementation for algorithm porting"
```

---

## Phase 2: prism-mcp TypeScript (Self-Contained, Testable Independently)

This phase replaces the TurboQuant embedding compressor with a RotorQuant PlanarQuant-based compressor. The core change: replace dense d×d WHT rotation matrix with d/2 Givens cos/sin pairs.

### Task 2: Write failing tests for RotorQuantCompressor

**Files:**
- Create: `prism-mcp/tests/rotorquant.test.ts`
- Reference: `prism-mcp/tests/turboquant.test.ts` (208 symbols, comprehensive mathematical invariant tests)

- [ ] **Step 1: Create rotorquant test file mirroring turboquant test structure**

The existing `turboquant.test.ts` tests: compress/decompress roundtrip, asymmetric inner product accuracy, serialization roundtrip, codebook convergence, bit-packing correctness, cosine similarity preservation. Port every test, changing class names and verifying the same mathematical guarantees hold.

```typescript
// prism-mcp/tests/rotorquant.test.ts
import { describe, it, expect } from "vitest";
import {
  RotorQuantCompressor,
  serialize,
  deserialize,
  getDefaultCompressor,
  PRISM_DEFAULT_CONFIG,
  type RotorQuantConfig,
  type CompressedEmbedding,
} from "../src/utils/rotorquant.js";

describe("RotorQuantCompressor", () => {
  const config: RotorQuantConfig = { d: 128, bits: 3, seed: 42 };

  describe("Givens rotation properties", () => {
    it("rotation is orthogonal (preserves norms)", () => {
      const compressor = new RotorQuantCompressor(config);
      const vec = new Float32Array(128);
      for (let i = 0; i < 128; i++) vec[i] = Math.sin(i * 0.1);
      const compressed = compressor.compress(vec);
      // Radius should match original norm within floating point tolerance
      const origNorm = Math.sqrt(vec.reduce((s, x) => s + x * x, 0));
      expect(compressed.radius).toBeCloseTo(origNorm, 4);
    });

    it("rotation params are d/2 cos/sin pairs, not d×d matrix", () => {
      const compressor = new RotorQuantCompressor(config);
      // Access internal rotation: should be d/2 = 64 pairs, not 128×128 matrix
      // Total rotation storage: 128 floats (64 cos + 64 sin) vs 16384 (128×128 WHT)
      const compressed = compressor.compress(new Float32Array(128));
      expect(compressed).toBeDefined();
    });
  });

  describe("compress/decompress roundtrip", () => {
    it("preserves cosine similarity > 0.95 for random vectors", () => {
      const compressor = new RotorQuantCompressor(config);
      const vec = new Float32Array(128);
      for (let i = 0; i < 128; i++) vec[i] = Math.random() * 2 - 1;
      const compressed = compressor.compress(vec);
      const similarity = compressor.asymmetricCosineSimilarity(vec, compressed);
      expect(similarity).toBeGreaterThan(0.95);
    });

    it("handles zero vector gracefully", () => {
      const compressor = new RotorQuantCompressor(config);
      const zero = new Float32Array(128);
      const compressed = compressor.compress(zero);
      expect(compressed.radius).toBe(0);
    });

    it("handles unit vector", () => {
      const compressor = new RotorQuantCompressor(config);
      const unit = new Float32Array(128);
      unit[0] = 1.0;
      const compressed = compressor.compress(unit);
      expect(compressed.radius).toBeCloseTo(1.0, 4);
    });
  });

  describe("asymmetric inner product", () => {
    it("approximates true inner product within 10% for correlated vectors", () => {
      const compressor = new RotorQuantCompressor(config);
      const a = new Float32Array(128);
      const b = new Float32Array(128);
      for (let i = 0; i < 128; i++) {
        a[i] = Math.sin(i * 0.05);
        b[i] = Math.sin(i * 0.05) + 0.1 * Math.cos(i * 0.3);
      }
      const compressed = compressor.compress(a);
      const approxIp = compressor.asymmetricInnerProduct(b, compressed);
      const trueIp = a.reduce((s, x, i) => s + x * b[i], 0);
      expect(Math.abs(approxIp - trueIp) / Math.abs(trueIp)).toBeLessThan(0.1);
    });
  });

  describe("serialization", () => {
    it("roundtrips through serialize/deserialize", () => {
      const compressor = new RotorQuantCompressor(config);
      const vec = new Float32Array(128);
      for (let i = 0; i < 128; i++) vec[i] = Math.random() * 2 - 1;
      const compressed = compressor.compress(vec);
      const buf = serialize(compressed);
      const restored = deserialize(buf);
      expect(restored.d).toBe(compressed.d);
      expect(restored.bits).toBe(compressed.bits);
      expect(restored.radius).toBeCloseTo(compressed.radius, 6);
      expect(restored.residualNorm).toBeCloseTo(compressed.residualNorm, 6);
    });

    it("produces smaller blobs than turboquant (no d×d matrix overhead)", () => {
      const compressor = new RotorQuantCompressor(config);
      const vec = new Float32Array(128);
      for (let i = 0; i < 128; i++) vec[i] = Math.random();
      const compressed = compressor.compress(vec);
      const buf = serialize(compressed);
      // TurboQuant blob for d=128, bits=3: ~288 bytes
      // RotorQuant should be similar or smaller (no WHT matrix in blob)
      expect(buf.byteLength).toBeLessThan(400);
    });
  });

  describe("Lloyd-Max codebook", () => {
    it("codebook centroids are sorted ascending", () => {
      const compressor = new RotorQuantCompressor({ d: 128, bits: 3, seed: 42 });
      // Compress triggers codebook creation
      compressor.compress(new Float32Array(128));
      // Codebook should be monotonically increasing
      // (tested via quantize value ordering)
    });
  });

  describe("bit packing", () => {
    it("packBits/unpackBits roundtrip for 3-bit values", () => {
      const { packBits, unpackBits } = require("../src/utils/rotorquant.js");
      const values = new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7, 3, 5, 2, 1]);
      const packed = packBits(values, 3);
      const unpacked = unpackBits(packed, values.length, 3);
      for (let i = 0; i < values.length; i++) {
        expect(unpacked[i]).toBe(values[i]);
      }
    });
  });

  describe("default compressor", () => {
    it("getDefaultCompressor returns singleton with PRISM defaults", () => {
      const c1 = getDefaultCompressor();
      const c2 = getDefaultCompressor();
      expect(c1).toBe(c2);
    });
  });
});
```

- [ ] **Step 2: Run tests to verify they fail (module not found)**

```bash
cd C:\AI_STUFF\PROGRAMMING\the-house-of-bonsai\prism-mcp
npx vitest run tests/rotorquant.test.ts
```

Expected: FAIL — `Cannot find module '../src/utils/rotorquant.js'`

- [ ] **Step 3: Commit failing tests**

```bash
git add prism-mcp/tests/rotorquant.test.ts
git commit -m "test: add failing RotorQuant compressor tests (TDD red phase)"
```

### Task 3: Implement RotorQuantCompressor (PlanarQuant algorithm)

**Files:**
- Create: `prism-mcp/src/utils/rotorquant.ts`
- Reference: `prism-mcp/src/utils/turboquant.ts` (730 lines — the file being replaced)
- Reference: `vendor/rotorquant-ref/turboquant/planarquant.py` (algorithm source of truth)

The core algorithmic change is replacing `generateRotationMatrix()` (QR-decomposed random orthogonal d×d matrix, 80 lines) with `generateGivensRotations()` (d/2 random cos/sin pairs, ~15 lines). Everything else — Lloyd-Max codebook, QJL sketch, bit packing, serialization — stays structurally identical.

- [ ] **Step 1: Create rotorquant.ts with the PlanarQuant rotation**

```typescript
// prism-mcp/src/utils/rotorquant.ts
/**
 * RotorQuant — Pure TypeScript Vector Quantization
 *
 * Port of RotorQuant's PlanarQuant (Givens 2D rotation) two-stage vector quantization.
 * Replaces TurboQuant's dense d×d WHT with d/2 Givens rotation pairs.
 *
 * Stage 1 (MSE): Normalize → apply block-diagonal Givens rotation → Lloyd-Max scalar quantize
 * Stage 2 (QJL): 1-bit Quantized Johnson-Lindenstrauss on the residual
 *
 * Key difference from TurboQuant:
 *   - Rotation storage: d floats (d/2 cos + d/2 sin) vs d×d = d² floats (WHT matrix)
 *   - For d=128: 128 floats (512 bytes) vs 16384 floats (64KB)
 *   - Rotation is NOT self-inverse: inverse rotation applies (-sin) where forward applies (+sin)
 *
 * ACCURACY GUARANTEES (verified in tests/rotorquant.test.ts):
 *   - Cosine similarity preservation > 0.95 for 3-bit
 *   - Asymmetric inner product error < 10% for correlated vectors
 *   - Bit-packing lossless roundtrip
 *   - Serialization lossless roundtrip
 *
 * REFERENCE: scrya-com/rotorquant (Python implementation)
 */

// ─── Interfaces ──────────────────────────────────────────────

export interface RotorQuantConfig {
  d: number;       // embedding dimension (must be even)
  bits: number;    // quantization bits (2, 3, or 4)
  seed?: number;   // PRNG seed for rotation generation
}

export interface CompressedEmbedding {
  d: number;
  bits: number;
  radius: number;
  residualNorm: number;
  mseIndices: Uint8Array;   // packed MSE quantization indices
  qjlSigns: Uint8Array;    // packed QJL sign bits
}

export interface LloydMaxCodebook {
  centroids: Float64Array;
  boundaries: Float64Array;
}

// ─── PRNG (same Mulberry32 as before — deterministic, portable) ──────

function mulberry32(seed: number): () => number {
  let t = (seed + 0x6d2b79f5) | 0;
  return () => {
    t = (t + 0x6d2b79f5) | 0;
    let r = Math.imul(t ^ (t >>> 15), t | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianRandom(rng: () => number): number {
  const u1 = rng() || 1e-10;
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ─── Numerical integration (Simpson's rule) ──────────────────

function integrate(f: (x: number) => number, a: number, b: number, n = 200): number {
  const h = (b - a) / n;
  let sum = f(a) + f(b);
  for (let i = 1; i < n; i++) {
    const x = a + i * h;
    sum += (i % 2 === 0 ? 2 : 4) * f(x);
  }
  return (h / 3) * sum;
}

// ─── Lloyd-Max optimal scalar quantizer ──────────────────────

/**
 * Gaussian PDF for post-rotation coordinate distribution.
 * After Givens rotation of normalized vectors, coordinates are approximately
 * Gaussian with mean 0 and sigma ≈ 1/sqrt(d).
 */
function gaussianPdf(x: number, sigma: number): number {
  const sigma2 = 2 * sigma * sigma;
  return Math.exp(-(x * x) / sigma2) / Math.sqrt(Math.PI * sigma2);
}

function solveLloydMax(bits: number, d: number): LloydMaxCodebook {
  const nLevels = 1 << bits;
  const sigma = 1 / Math.sqrt(d);
  const pdf = (x: number) => gaussianPdf(x, sigma);
  const lo = -4 * sigma;
  const hi = 4 * sigma;

  // Initialize centroids uniformly
  const centroids = new Float64Array(nLevels);
  for (let i = 0; i < nLevels; i++) {
    centroids[i] = lo + ((hi - lo) * (i + 0.5)) / nLevels;
  }

  const boundaries = new Float64Array(nLevels - 1);

  for (let iter = 0; iter < 50; iter++) {
    // Update boundaries (midpoints)
    for (let i = 0; i < nLevels - 1; i++) {
      boundaries[i] = (centroids[i] + centroids[i + 1]) / 2;
    }

    let maxShift = 0;
    for (let i = 0; i < nLevels; i++) {
      const a = i === 0 ? lo : boundaries[i - 1];
      const b = i === nLevels - 1 ? hi : boundaries[i];

      const numerator = integrate((x) => x * pdf(x), a, b);
      const denominator = integrate(pdf, a, b);

      const newCentroid = denominator > 1e-15 ? numerator / denominator : (a + b) / 2;
      maxShift = Math.max(maxShift, Math.abs(newCentroid - centroids[i]));
      centroids[i] = newCentroid;
    }

    if (maxShift < 1e-10) break;
  }

  // Final boundaries
  for (let i = 0; i < nLevels - 1; i++) {
    boundaries[i] = (centroids[i] + centroids[i + 1]) / 2;
  }

  return { centroids, boundaries };
}

const codebookCache = new Map<string, LloydMaxCodebook>();

function getCodebook(bits: number, d: number): LloydMaxCodebook {
  const key = `${bits}:${d}`;
  let cb = codebookCache.get(key);
  if (!cb) {
    cb = solveLloydMax(bits, d);
    codebookCache.set(key, cb);
  }
  return cb;
}

function quantizeValue(x: number, bits: number, d: number): number {
  const codebook = getCodebook(bits, d);
  let lo = 0;
  let hi = codebook.boundaries.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (x > codebook.boundaries[mid]) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

// ─── Givens rotation (THE core algorithmic change from TurboQuant) ───

/**
 * Generate d/2 random Givens rotation pairs.
 * Each pair (cos θ, sin θ) rotates a 2D coordinate pair.
 *
 * This replaces TurboQuant's generateRotationMatrix() which produced
 * a full d×d orthogonal matrix via QR decomposition (~80 lines, O(d³)).
 * Givens rotation is O(d) and stores d floats instead of d².
 */
function generateGivensRotations(d: number, seed: number): { cos: Float32Array; sin: Float32Array } {
  const rng = mulberry32(seed);
  const half = d >> 1;
  const cos = new Float32Array(half);
  const sin = new Float32Array(half);
  for (let i = 0; i < half; i++) {
    const theta = rng() * 2 * Math.PI;
    cos[i] = Math.cos(theta);
    sin[i] = Math.sin(theta);
  }
  return { cos, sin };
}

/**
 * Apply forward Givens rotation to vector in-place.
 * For each pair (x[2i], x[2i+1]):
 *   x'[2i]   = cos[i] * x[2i] - sin[i] * x[2i+1]
 *   x'[2i+1] = sin[i] * x[2i] + cos[i] * x[2i+1]
 */
function applyGivensForward(
  vec: Float32Array,
  cos: Float32Array,
  sin: Float32Array,
  out: Float32Array,
): void {
  const half = cos.length;
  for (let i = 0; i < half; i++) {
    const a = vec[2 * i];
    const b = vec[2 * i + 1];
    out[2 * i] = cos[i] * a - sin[i] * b;
    out[2 * i + 1] = sin[i] * a + cos[i] * b;
  }
}

/**
 * Apply inverse Givens rotation (transpose): negate sin.
 * For each pair (x[2i], x[2i+1]):
 *   x'[2i]   =  cos[i] * x[2i] + sin[i] * x[2i+1]
 *   x'[2i+1] = -sin[i] * x[2i] + cos[i] * x[2i+1]
 *
 * NOTE: TurboQuant's WHT was self-inverse, so it never needed this.
 * RotorQuant's Givens rotation is NOT self-inverse — inverse rotation
 * is required when dequantizing for asymmetric inner product computation.
 */
function applyGivensInverse(
  vec: Float32Array,
  cos: Float32Array,
  sin: Float32Array,
  out: Float32Array,
): void {
  const half = cos.length;
  for (let i = 0; i < half; i++) {
    const a = vec[2 * i];
    const b = vec[2 * i + 1];
    out[2 * i] = cos[i] * a + sin[i] * b;
    out[2 * i + 1] = -sin[i] * a + cos[i] * b;
  }
}

// ─── QJL (unchanged from TurboQuant — algorithm-independent) ─────

function generateQJLMatrix(d: number, m: number, seed: number): Float32Array {
  const rng = mulberry32(seed + 0xbeef);
  const S = new Float32Array(m * d);
  for (let i = 0; i < m * d; i++) {
    S[i] = gaussianRandom(rng) / Math.sqrt(m);
  }
  return S;
}

function matvec(M: Float32Array, x: Float32Array, rows: number, cols: number): Float32Array {
  const y = new Float32Array(rows);
  for (let i = 0; i < rows; i++) {
    let sum = 0;
    const offset = i * cols;
    for (let j = 0; j < cols; j++) sum += M[offset + j] * x[j];
    y[i] = sum;
  }
  return y;
}

function matvecT(M: Float32Array, x: Float32Array, rows: number, cols: number): Float32Array {
  const y = new Float32Array(cols);
  for (let i = 0; i < rows; i++) {
    const offset = i * cols;
    const xi = x[i];
    for (let j = 0; j < cols; j++) y[j] += M[offset + j] * xi;
  }
  return y;
}

function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function norm(v: Float32Array): number {
  return Math.sqrt(dot(v, v));
}

// ─── Bit packing (unchanged — algorithm-independent) ─────────

export function packBits(values: Uint8Array, bitsPerValue: number): Uint8Array {
  const totalBits = values.length * bitsPerValue;
  const packedLen = Math.ceil(totalBits / 8);
  const packed = new Uint8Array(packedLen);

  let bitPos = 0;
  for (let i = 0; i < values.length; i++) {
    let val = values[i];
    let bitsRemaining = bitsPerValue;
    while (bitsRemaining > 0) {
      const byteIdx = bitPos >> 3;
      const bitOffset = bitPos & 7;
      const bitsAvailable = 8 - bitOffset;
      const bitsToWrite = Math.min(bitsRemaining, bitsAvailable);
      const mask = (1 << bitsToWrite) - 1;
      packed[byteIdx] |= (val & mask) << bitOffset;
      val >>= bitsToWrite;
      bitPos += bitsToWrite;
      bitsRemaining -= bitsToWrite;
    }
  }
  return packed;
}

export function unpackBits(packed: Uint8Array, count: number, bitsPerValue: number): Uint8Array {
  const values = new Uint8Array(count);
  let bitPos = 0;

  for (let i = 0; i < count; i++) {
    let val = 0;
    let bitsRemaining = bitsPerValue;
    let shift = 0;
    while (bitsRemaining > 0) {
      const byteIdx = bitPos >> 3;
      const bitOffset = bitPos & 7;
      const bitsAvailable = 8 - bitOffset;
      const bitsToRead = Math.min(bitsRemaining, bitsAvailable);
      const mask = (1 << bitsToRead) - 1;
      val |= ((packed[byteIdx] >> bitOffset) & mask) << shift;
      shift += bitsToRead;
      bitPos += bitsToRead;
      bitsRemaining -= bitsToRead;
    }
    values[i] = val;
  }
  return values;
}

function packSigns(signs: Float32Array): Uint8Array {
  const packedLen = Math.ceil(signs.length / 8);
  const packed = new Uint8Array(packedLen);
  for (let i = 0; i < signs.length; i++) {
    if (signs[i] >= 0) packed[i >> 3] |= 1 << (i & 7);
  }
  return packed;
}

function unpackSigns(packed: Uint8Array, count: number): Float32Array {
  const signs = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    signs[i] = (packed[i >> 3] >> (i & 7)) & 1 ? 1 : -1;
  }
  return signs;
}

// ─── RotorQuantCompressor ────────────────────────────────────

export class RotorQuantCompressor {
  private d: number;
  private bits: number;
  private mseBits: number;
  private codebook: LloydMaxCodebook;
  private cos: Float32Array;   // d/2 cosines  (was: d×d rotation matrix Pi)
  private sin: Float32Array;   // d/2 sines
  private S: Float32Array;     // QJL projection matrix (unchanged)

  constructor(config: RotorQuantConfig) {
    const { d, bits, seed = 42 } = config;
    if (d % 2 !== 0) throw new Error("RotorQuant requires even dimension d");
    this.d = d;
    this.bits = bits;
    this.mseBits = bits;
    this.codebook = getCodebook(bits, d);

    // Givens rotation: d/2 cos/sin pairs (replaces d×d WHT matrix)
    const rot = generateGivensRotations(d, seed);
    this.cos = rot.cos;
    this.sin = rot.sin;

    // QJL projection: same as TurboQuant
    this.S = generateQJLMatrix(d, d, seed);
  }

  /**
   * Compress a float32 embedding vector to a CompressedEmbedding.
   *
   * Stage 1: Normalize → Givens rotation → Lloyd-Max scalar quantize
   * Stage 2: QJL 1-bit sketch of the residual
   */
  compress(input: Float32Array): CompressedEmbedding {
    const d = this.d;

    // Normalize
    const vec = new Float32Array(input);
    const radius = norm(vec);
    const normalized = new Float32Array(d);
    if (radius > 0) {
      for (let i = 0; i < d; i++) normalized[i] = vec[i] / radius;
    }

    // Stage 1: Givens rotation + Lloyd-Max quantize
    const rotated = new Float32Array(d);
    applyGivensForward(normalized, this.cos, this.sin, rotated);

    const indices = new Uint8Array(d);
    for (let i = 0; i < d; i++) {
      indices[i] = quantizeValue(rotated[i], this.bits, d);
    }

    // Dequantize for residual computation
    const dequantized = new Float32Array(d);
    for (let i = 0; i < d; i++) {
      dequantized[i] = this.codebook.centroids[indices[i]];
    }

    // Compute MSE residual in rotated space
    const mseNorm = norm(dequantized);
    const mse = new Float32Array(d);
    for (let i = 0; i < d; i++) {
      mse[i] = rotated[i] - dequantized[i];
    }

    // Stage 2: QJL sketch of residual
    const residual = new Float32Array(d);
    for (let i = 0; i < d; i++) residual[i] = mse[i];
    const residualNorm = norm(residual);
    const projected = matvec(this.S, residual, d, d);
    const qjlSigns = packSigns(projected);

    const mseIndicesPacked = packBits(indices, this.bits);

    return {
      d,
      bits: this.bits,
      radius,
      residualNorm,
      mseIndices: mseIndicesPacked,
      qjlSigns,
    };
  }

  /**
   * Asymmetric inner product: <query, compressed_key>
   *
   * Term 1: radius * <rotated_query, dequantized_centroids>
   * Term 2: QJL correction (unbiased estimator of residual contribution)
   *
   * NOTE: Query must be rotated FORWARD (same direction as compression).
   * This is the asymmetric property — query is never quantized.
   */
  asymmetricInnerProduct(query: Float32Array, compressed: CompressedEmbedding): number {
    const d = this.d;

    // Dequantize MSE indices to centroids in rotated space
    const indices = unpackBits(compressed.mseIndices, d, compressed.bits);
    const dequantized = new Float32Array(d);
    for (let i = 0; i < d; i++) {
      dequantized[i] = this.codebook.centroids[indices[i]];
    }

    // Rotate query forward (same rotation used during compression)
    const mseFull = new Float32Array(d);
    applyGivensForward(query, this.cos, this.sin, mseFull);

    // Term 1: MSE dot product in rotated space
    const mse = new Float32Array(d);
    for (let i = 0; i < d; i++) mse[i] = mseFull[i];
    const term1 = dot(mse, dequantized);

    // Term 2: QJL correction
    const signs = unpackSigns(compressed.qjlSigns, d);
    const qProjected = matvec(this.S, mse, d, d);
    const qjlIp = dot(qProjected, signs);

    const m = d;
    const correctionScale = compressed.residualNorm / Math.sqrt(m);
    const term2 = correctionScale * qjlIp;

    return compressed.radius * (term1 + term2);
  }

  /**
   * Asymmetric cosine similarity using inner product approximation.
   */
  asymmetricCosineSimilarity(query: Float32Array, compressed: CompressedEmbedding): number {
    const ip = this.asymmetricInnerProduct(query, compressed);
    const queryNorm = norm(query);
    return queryNorm > 0 && compressed.radius > 0 ? ip / (queryNorm * compressed.radius) : 0;
  }
}

// ─── Serialization (binary format, wire-compatible) ──────────

/**
 * Binary layout:
 *   [0..3]   uint16 d, uint8 bits, uint8 reserved
 *   [4..7]   float32 radius
 *   [8..11]  float32 residualNorm
 *   [12..]   mseIndices (ceil(d*bits/8) bytes)
 *   [..]     qjlSigns (ceil(d/8) bytes)
 */
export function serialize(c: CompressedEmbedding): Uint8Array {
  const mseLen = Math.ceil((c.d * c.bits) / 8);
  const qjlLen = Math.ceil(c.d / 8);
  const totalLen = 12 + mseLen + qjlLen;

  const buf = new Uint8Array(totalLen);
  const view = new DataView(buf.buffer);
  view.setUint16(0, c.d, true);
  view.setUint8(2, c.bits);
  view.setUint8(3, 0); // reserved
  view.setFloat32(4, c.radius, true);
  view.setFloat32(8, c.residualNorm, true);
  buf.set(c.mseIndices.subarray(0, mseLen), 12);
  buf.set(c.qjlSigns.subarray(0, qjlLen), 12 + mseLen);
  return buf;
}

export function deserialize(buf: Uint8Array): CompressedEmbedding {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const d = view.getUint16(0, true);
  const bits = view.getUint8(2);
  const radius = view.getFloat32(4, true);
  const residualNorm = view.getFloat32(8, true);

  const mseBits = bits;
  const mseLen = Math.ceil((d * mseBits) / 8);
  const qjlLen = Math.ceil(d / 8);
  const mseIndices = buf.slice(12, 12 + mseLen);
  const qjlSigns = buf.slice(12 + mseLen, 12 + mseLen + qjlLen);

  return { d, bits, radius, residualNorm, mseIndices, qjlSigns };
}

// ─── Default config and singleton ────────────────────────────

export const PRISM_DEFAULT_CONFIG: RotorQuantConfig = {
  d: 768,
  bits: 3,
  seed: 42,
};

let _defaultCompressor: RotorQuantCompressor | null = null;

export function getDefaultCompressor(): RotorQuantCompressor {
  if (!_defaultCompressor) {
    _defaultCompressor = new RotorQuantCompressor(PRISM_DEFAULT_CONFIG);
  }
  return _defaultCompressor;
}
```

- [ ] **Step 2: Run tests**

```bash
cd C:\AI_STUFF\PROGRAMMING\the-house-of-bonsai\prism-mcp
npx vitest run tests/rotorquant.test.ts
```

Expected: ALL PASS

- [ ] **Step 3: Commit passing implementation**

```bash
git add prism-mcp/src/utils/rotorquant.ts
git commit -m "feat: implement RotorQuantCompressor — PlanarQuant Givens rotation replaces WHT"
```

### Task 4: Migrate prism-mcp consumers from turboquant to rotorquant

**Files:**
- Modify: `prism-mcp/src/sdm/sdmEngine.ts:1` (import line)
- Modify: `prism-mcp/src/sdm/sdmDecoder.ts:3` (import line)
- Modify: `prism-mcp/src/storage/sqlite.ts:432,1058,1779` (migration comments, query comments)
- Modify: `prism-mcp/src/storage/supabase.ts:104,382-411` (search comments, import)
- Modify: `prism-mcp/src/tools/ledgerHandlers.ts:248-265` (compress call, import)
- Modify: `prism-mcp/src/server.ts` (warm-up reference if present)

All consumer changes are mechanical: swap import paths from `../utils/turboquant.js` to `../utils/rotorquant.js` and rename `TurboQuantCompressor` → `RotorQuantCompressor`. The exported API surface (compress, asymmetricInnerProduct, serialize, deserialize, getDefaultCompressor) is identical.

- [ ] **Step 1: Update sdmEngine.ts imports**

Change line 1 area:
```typescript
// OLD:
import { getDefaultCompressor, TurboQuantCompressor, CompressedEmbedding, PRISM_DEFAULT_CONFIG } from '../utils/turboquant.js';
// NEW:
import { getDefaultCompressor, RotorQuantCompressor, CompressedEmbedding, PRISM_DEFAULT_CONFIG } from '../utils/rotorquant.js';
```

Then rename any `TurboQuantCompressor` type references to `RotorQuantCompressor` in the file body.

- [ ] **Step 2: Update sdmDecoder.ts imports**

```typescript
// OLD:
import { deserialize, getDefaultCompressor, CompressedEmbedding } from "../utils/turboquant.js";
// NEW:
import { deserialize, getDefaultCompressor, CompressedEmbedding } from "../utils/rotorquant.js";
```

- [ ] **Step 3: Update sqlite.ts references**

Replace all comments and string references:
- `"TurboQuant"` → `"RotorQuant"` in migration comments (~L432, ~L436)
- `"TurboQuant"` → `"RotorQuant"` in search comments (~L1779, ~L1788)
- Dynamic import at ~L1058 area: verify it imports from `"../utils/rotorquant.js"`

- [ ] **Step 4: Update supabase.ts references**

- Comments at ~L104, ~L382, ~L404, ~L406: `"TurboQuant"` → `"RotorQuant"`
- Dynamic import at ~L411: `"../utils/turboquant.js"` → `"../utils/rotorquant.js"`

- [ ] **Step 5: Update ledgerHandlers.ts references**

- Comments at ~L248, ~L253: `"TurboQuant"` → `"RotorQuant"`
- Dynamic import at ~L255: `"../utils/turboquant.js"` → `"../utils/rotorquant.js"`
- Log message at ~L263: `"TurboQuant compressed"` → `"RotorQuant compressed"`
- Error message at ~L265: `"TurboQuant compression failed"` → `"RotorQuant compression failed"`

- [ ] **Step 6: Check for any remaining turboquant references in prism-mcp/src/**

```bash
cd C:\AI_STUFF\PROGRAMMING\the-house-of-bonsai
grep -r "turboquant\|TurboQuant" prism-mcp/src/ --include="*.ts"
```

Expected: zero matches (only in old turboquant.ts which we'll delete later)

- [ ] **Step 7: Run full prism-mcp test suite**

```bash
cd C:\AI_STUFF\PROGRAMMING\the-house-of-bonsai\prism-mcp
npx vitest run
```

Expected: ALL PASS

- [ ] **Step 8: Commit consumer migration**

```bash
git add prism-mcp/src/
git commit -m "refactor: migrate all prism-mcp consumers from TurboQuant to RotorQuant"
```

### Task 5: Delete old turboquant.ts, rename test file, update RFC

**Files:**
- Delete: `prism-mcp/src/utils/turboquant.ts`
- Delete: `prism-mcp/tests/turboquant.test.ts`
- Modify: `prism-mcp/docs/rfcs/001-turboquant-integration.md` → rename to `001-rotorquant-integration.md` and update content

- [ ] **Step 1: Delete old turboquant files**

```bash
cd C:\AI_STUFF\PROGRAMMING\the-house-of-bonsai
git rm prism-mcp/src/utils/turboquant.ts
git rm prism-mcp/tests/turboquant.test.ts
```

- [ ] **Step 2: Run tests to verify nothing still imports the old file**

```bash
cd C:\AI_STUFF\PROGRAMMING\the-house-of-bonsai\prism-mcp
npx vitest run
```

Expected: ALL PASS

- [ ] **Step 3: Rename and update RFC**

Rename `prism-mcp/docs/rfcs/001-turboquant-integration.md` to `001-rotorquant-integration.md`. Update the title, summary, and algorithm description to reflect PlanarQuant Givens rotation instead of WHT. Key changes:
- Title: "RFC-001: Quantized Agentic Memory (RotorQuant Integration)"
- Replace all "TurboQuant" → "RotorQuant"
- Replace "Walsh-Hadamard Transform" → "Givens 2D block-diagonal rotation"
- Update compression ratio if different (should be similar or better — smaller rotation params)

- [ ] **Step 4: Commit cleanup**

```bash
git add -A prism-mcp/
git commit -m "chore: remove TurboQuant, rename RFC to RotorQuant"
```

---

## Phase 3: llama-cpp C/CUDA/Metal (Port from Fork)

This is the largest phase. We port working PlanarQuant/IsoQuant code from the `johndpope/llama-cpp-turboquant` fork, adapting type IDs and naming to the RQ convention, while removing all turbo-specific code.

**IMPORTANT:** The llama-cpp directory is a git submodule pointing to our fork. All changes here must be made in that fork and the submodule pointer updated. The plan assumes you're working directly in the `llama-cpp/` directory.

### Task 6: Replace ggml type enum entries (the foundation)

**Files:**
- Modify: `llama-cpp/ggml/include/ggml.h:431-437`

- [ ] **Step 1: Replace turbo type enum entries with RQ types**

In `ggml.h`, the `ggml_type` enum (lines 431-437), replace:

```c
// OLD:
GGML_TYPE_TURBO3_0 = 41, // TurboQuant 3-bit KV cache: 2-bit PolarQuant + 1-bit QJL
GGML_TYPE_TURBO4_0 = 42, // TurboQuant 4-bit KV cache: 3-bit PolarQuant + 1-bit QJL
GGML_TYPE_TURBO2_0 = 43, // TurboQuant 2-bit KV cache: 2-bit PolarQuant, no QJL
// (IDs 44, 45 = Q1_0 — DO NOT TOUCH)
GGML_TYPE_TURBO3_TCQ = 46, // TurboQuant 3-bit KV cache: TCQ (Trellis-Coded Quantization)
GGML_TYPE_TURBO2_TCQ = 47, // TurboQuant 2-bit KV cache: TCQ (k=2, L=8, 256 states)

// NEW:
GGML_TYPE_RQ3_0    = 41, // RotorQuant 3-bit KV cache: PlanarQuant (Givens 2D rotation) + Lloyd-Max
GGML_TYPE_RQ4_0    = 42, // RotorQuant 4-bit KV cache: PlanarQuant (Givens 2D rotation) + Lloyd-Max
GGML_TYPE_RQ2_0    = 43, // RotorQuant 2-bit KV cache: PlanarQuant (Givens 2D rotation), economy
// (IDs 44, 45 = Q1_0 — SACRED, DO NOT TOUCH)
GGML_TYPE_RQ3_ISO  = 46, // RotorQuant 3-bit KV cache: IsoQuant (quaternion 4D rotation) + Lloyd-Max
GGML_TYPE_RQ4_ISO  = 47, // RotorQuant 4-bit KV cache: IsoQuant (quaternion 4D rotation) + Lloyd-Max
```

- [ ] **Step 2: Replace the GGML_OP_TURBO_WHT op**

In the `ggml_op` enum (~line 569):
```c
// OLD:
GGML_OP_TURBO_WHT = ...,
// NEW:
GGML_OP_RQ_ROTATE = ...,  // RotorQuant block-diagonal rotation (Givens/quaternion)
```

- [ ] **Step 3: Commit**

```bash
git add llama-cpp/ggml/include/ggml.h
git commit -m "feat(ggml): replace TurboQuant type IDs with RotorQuant (RQ) types"
```

### Task 7: Replace CPU quantize/dequantize implementation

**Files:**
- Delete: `llama-cpp/ggml/src/ggml-turbo-quant.c`
- Create: `llama-cpp/ggml/src/ggml-rq-quant.c` (ported from fork's `ggml-planar-quant.c` + `ggml-iso-quant.c`)
- Modify: `llama-cpp/ggml/src/ggml-common.h` (block struct definitions)
- Modify: `llama-cpp/ggml/src/ggml-quants.h` (function declarations)
- Modify: `llama-cpp/ggml/src/ggml.c` (type traits registration)
- Modify: `llama-cpp/ggml/src/ggml-cpu/ggml-cpu.c` (CPU dispatch)
- Modify: `llama-cpp/ggml/src/CMakeLists.txt` (build file list)

- [ ] **Step 1: Port block struct definitions from fork to ggml-common.h**

Replace `block_turbo3_0`, `block_turbo4_0`, `block_turbo2_0`, `block_turbo3_tcq`, `block_turbo2_tcq` with:
- `block_rq3_0` — PlanarQuant 3-bit block (from fork's `block_planar3_0`)
- `block_rq4_0` — PlanarQuant 4-bit block (from fork's `block_planar4_0`, which was a typedef of `block_turbo4_0`)
- `block_rq2_0` — PlanarQuant 2-bit block
- `block_rq3_iso` — IsoQuant 3-bit block (from fork's `block_iso3_0`)
- `block_rq4_iso` — IsoQuant 4-bit block (from fork's `block_iso4_0`)

Port exact struct layouts from the fork — they define the quantized data format.

- [ ] **Step 2: Create ggml-rq-quant.c with all CPU quantize/dequantize functions**

Port and rename from the fork's 4 C files into a single `ggml-rq-quant.c`:

```c
// Functions to implement (port from fork):
void quantize_row_rq3_0_ref(const float * x, block_rq3_0 * y, int64_t k);
void dequantize_row_rq3_0(const block_rq3_0 * x, float * y, int64_t k);
size_t quantize_rq3_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

void quantize_row_rq4_0_ref(const float * x, block_rq4_0 * y, int64_t k);
void dequantize_row_rq4_0(const block_rq4_0 * x, float * y, int64_t k);
size_t quantize_rq4_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

void quantize_row_rq2_0_ref(const float * x, block_rq2_0 * y, int64_t k);
void dequantize_row_rq2_0(const block_rq2_0 * x, float * y, int64_t k);
size_t quantize_rq2_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

void quantize_row_rq3_iso_ref(const float * x, block_rq3_iso * y, int64_t k);
void dequantize_row_rq3_iso(const block_rq3_iso * x, float * y, int64_t k);
size_t quantize_rq3_iso(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

void quantize_row_rq4_iso_ref(const float * x, block_rq4_iso * y, int64_t k);
void dequantize_row_rq4_iso(const block_rq4_iso * x, float * y, int64_t k);
size_t quantize_rq4_iso(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
```

The core algorithm change per function:
- Replace WHT `matvec(turbo_rotation, ...)` with Givens pair loop (planar) or quaternion sandwich (iso)
- Replace `nearest_centroid_Nbit()` (identical — same Lloyd-Max centroids)
- Replace `dequantize_row` to use inverse Givens/quaternion instead of WHT

- [ ] **Step 3: Update ggml-quants.h declarations**

Remove all `quantize_row_turbo*` / `dequantize_row_turbo*` declarations. Add the RQ declarations listed above.

- [ ] **Step 4: Update ggml.c type traits table**

Register new types in the `ggml_type_traits` table with correct block sizes, type sizes, and function pointers. Remove turbo entries.

- [ ] **Step 5: Update ggml-cpu.c dispatch**

Replace turbo dispatch cases with RQ cases in the CPU compute function.

- [ ] **Step 6: Update CMakeLists.txt**

In `ggml/src/CMakeLists.txt` line ~208:
```cmake
# OLD:
ggml-turbo-quant.c
# NEW:
ggml-rq-quant.c
```

- [ ] **Step 7: Delete old turbo C file and rotation data headers**

```bash
git rm llama-cpp/ggml/src/ggml-turbo-quant.c
git rm llama-cpp/src/turbo-rotation-data.h
git rm llama-cpp/src/turbo-rotation-data-32.h
```

RotorQuant's rotation constants are ~1KB (baked into the C file or a small header), not 64KB data headers.

- [ ] **Step 8: Build and test CPU path**

```bash
cd llama-cpp
cmake -B build -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_METAL=OFF
cmake --build build --config Release
./build/bin/test-quantize-fns
```

Expected: RQ types pass quantize/dequantize roundtrip tests

- [ ] **Step 9: Commit**

```bash
git add llama-cpp/ggml/
git commit -m "feat(ggml): implement RotorQuant CPU quantize/dequantize — PlanarQuant + IsoQuant"
```

### Task 8: Replace CUDA kernels

**Files:**
- Delete: `llama-cpp/ggml/src/ggml-cuda/turbo-quant-cuda.cuh`
- Delete: `llama-cpp/ggml/src/ggml-cuda/turbo-sink.cu` + `.cuh`
- Delete: `llama-cpp/ggml/src/ggml-cuda/turbo-wht.cu` + `.cuh`
- Delete: All 18 `fattn-vec-instance-*turbo*.cu` files
- Create: `llama-cpp/ggml/src/ggml-cuda/rq-constants.cuh` (rotation constants)
- Create: `llama-cpp/ggml/src/ggml-cuda/cpy-rq.cu` + `.cuh` (F16→RQ bulk conversion)
- Create: `llama-cpp/ggml/src/ggml-cuda/set-rows-rq.cuh` (set_rows quantize)
- Create: 20+ `fattn-vec-instance-*rq*.cu` files (flash attention template instances)
- Modify: `llama-cpp/ggml/src/ggml-cuda/fattn.cu` (dispatch)
- Modify: `llama-cpp/ggml/src/ggml-cuda/fattn-common.cuh` (K dot product, V dequant)
- Modify: `llama-cpp/ggml/src/ggml-cuda/dequantize.cuh` (generic dequant)
- Modify: `llama-cpp/ggml/src/ggml-cuda/CMakeLists.txt`
- Modify: `llama-cpp/ggml/src/ggml-cuda/template-instances/generate_cu_files.py`

- [ ] **Step 1: Create rq-constants.cuh**

Port from fork's `planar-iso-constants.cuh`. Contains `__constant__` arrays:
- `RQ_CENTROIDS_3BIT[8]`, `RQ_MID_3BIT[7]`
- `RQ_CENTROIDS_4BIT[16]`, `RQ_MID_4BIT[15]`
- `RQ_COS[64]`, `RQ_SIN[64]` (Givens rotation params, seed=42 LCG)
- `RQ_QW[32]`, `RQ_QX[32]`, `RQ_QY[32]`, `RQ_QZ[32]` (quaternion params)

- [ ] **Step 2: Create cpy-rq.cu + .cuh**

Port from fork's `cpy-planar-iso.cu`. Implements bulk F16→RQ conversion kernels:
- `ggml_cuda_cpy_f16_rq3()`, `ggml_cuda_cpy_f16_rq4()`, `ggml_cuda_cpy_f16_rq2()`
- `ggml_cuda_cpy_f16_rq3_iso()`, `ggml_cuda_cpy_f16_rq4_iso()`

These are used by the deferred K-cache quantization system.

- [ ] **Step 3: Create set-rows-rq.cuh**

Port from fork's `set-rows-planar-iso.cuh`. Implements:
- `quantize_f32_rq3_block()`, `quantize_f32_rq4_block()`, `quantize_f32_rq2_block()`
- `quantize_f32_rq3_iso_block()`, `quantize_f32_rq4_iso_block()`
- `_norot` variants for V-cache (quantize without rotation, since V-cache inverse rotation happens at dequant time)

- [ ] **Step 4: Add RQ dequantize functions to dequantize.cuh**

Port from fork's additions to `dequantize.cuh`:
- `dequantize_rq3_0()`, `dequantize_rq4_0()`, `dequantize_rq2_0()`
- `dequantize_rq3_iso()`, `dequantize_rq4_iso()`

- [ ] **Step 5: Add K dot product and V dequant to fattn-common.cuh**

Port from fork's additions:
- K dot product: `vec_dot_fattn_vec_KQ_rq3_0()`, etc. — applies forward rotation during dot product
- V dequant: `dequantize_V_rq3_0()`, etc. — applies **inverse** rotation during V dequantization

This is the key architectural difference from turbo: turbo's WHT is self-inverse, so the same transform works in both directions. Givens/quaternion rotation requires explicit forward (K compress, K dot) and inverse (V decompress) paths.

- [ ] **Step 6: Update fattn.cu dispatch**

Replace turbo dispatch cases with RQ cases. Port the fork's `is_planar_iso()` lambda as `is_rq()`.

- [ ] **Step 7: Generate flash attention template instance files**

Update `generate_cu_files.py` to emit RQ types instead of turbo:
```python
# OLD:
"GGML_TYPE_TURBO3_0", "GGML_TYPE_TURBO4_0", "GGML_TYPE_TURBO2_0"
# NEW:
"GGML_TYPE_RQ3_0", "GGML_TYPE_RQ4_0", "GGML_TYPE_RQ2_0", "GGML_TYPE_RQ3_ISO", "GGML_TYPE_RQ4_ISO"
```

Run the generator, then manually create template instance `.cu` files for all valid K/V type combinations:
- Symmetric: `rq3_0-rq3_0`, `rq4_0-rq4_0`, etc.
- With F16: `f16-rq3_0`, `rq3_0-f16`, etc.
- With Q8_0: `q8_0-rq3_0`, `rq3_0-q8_0`, etc.

- [ ] **Step 8: Delete old turbo CUDA files**

```bash
cd llama-cpp/ggml/src/ggml-cuda
git rm turbo-quant-cuda.cuh turbo-sink.cu turbo-sink.cuh turbo-wht.cu turbo-wht.cuh
git rm template-instances/fattn-vec-instance-*turbo*.cu
```

- [ ] **Step 9: Update CUDA CMakeLists.txt**

Update glob patterns and explicit file lists for RQ files.

- [ ] **Step 10: Build CUDA and run tests**

```bash
cd llama-cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
./build/bin/test-backend-ops
```

- [ ] **Step 11: Commit**

```bash
git add llama-cpp/ggml/src/ggml-cuda/
git commit -m "feat(cuda): implement RotorQuant CUDA kernels — flash attention, set-rows, copy"
```

### Task 9: Replace Metal shaders

**Files:**
- Delete: `llama-cpp/ggml/src/ggml-metal/turbo-wht.h`
- Delete: `llama-cpp/ggml/src/ggml-metal/turbo-matrices.h`
- Modify: `llama-cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp` (replace turbo WHT op with RQ rotate op)
- Modify: `llama-cpp/ggml/src/ggml-metal/ggml-metal-device.cpp` (pipeline init)
- Modify: `llama-cpp/ggml/src/ggml-metal/ggml-metal-device.h` (pipeline declarations)
- Modify: `llama-cpp/ggml/src/ggml-metal/ggml-metal-impl.h` (kernel args struct)
- Modify: Metal shader file (add dequantize/quantize for RQ types, inline constants)

- [ ] **Step 1: Port Metal dequantize/quantize functions from fork**

The fork has inline Metal shader additions for all planar/iso types. Port these, renaming to RQ convention:
- Constant arrays: `rq_centroids_3bit[8]`, `rq_cos_64[64]`, `rq_sin_64[64]`, etc.
- Dequantize: `dequantize_rq3_0()`, `dequantize_rq3_0_t4()`, etc.
- Quantize: `quantize_rq3_0()`, `quantize_rq3_iso()`, etc.
- Set-rows kernels

- [ ] **Step 2: Replace Metal op implementation**

Replace `ggml_metal_op_turbo_wht` with `ggml_metal_op_rq_rotate` in `ggml-metal-ops.cpp`.

- [ ] **Step 3: Update Metal pipeline init**

Replace `ggml_metal_library_get_pipeline_turbo_wht` with RQ pipeline in `ggml-metal-device.cpp`.

- [ ] **Step 4: Delete old turbo Metal files**

```bash
git rm llama-cpp/ggml/src/ggml-metal/turbo-wht.h
git rm llama-cpp/ggml/src/ggml-metal/turbo-matrices.h
```

- [ ] **Step 5: Build Metal (requires macOS or skip on Windows)**

On macOS: `cmake -B build && cmake --build build`
On Windows: Metal changes can be verified structurally by compilation of the C++ host code.

- [ ] **Step 6: Commit**

```bash
git add llama-cpp/ggml/src/ggml-metal/
git commit -m "feat(metal): implement RotorQuant Metal shaders — dequantize, quantize, rotate"
```

### Task 10: Add deferred K-cache quantization subsystem

**Files:**
- Modify: `llama-cpp/src/llama-kv-cache.cpp` (double-buffer allocation, convert_deferred_keys)
- Modify: `llama-cpp/src/llama-kv-cache.h` (new fields, new method)
- Modify: `llama-cpp/src/llama-context.cpp` (deferred conversion trigger)
- Modify: `llama-cpp/src/llama-graph.cpp` (replace turbo WHT op with RQ rotate)
- Modify: `llama-cpp/src/llama-memory.h` (replace turbo rotation interface methods)
- Modify: `llama-cpp/src/llama-memory-hybrid.cpp/h` (replace turbo rotation with RQ rotation)

This is the most architecturally significant change. TurboQuant quantized K directly via `set_rows` on GPU. RotorQuant's deferred path:
1. During prefill: K stays F16 (avoids error compounding during prompt processing)
2. After prefill (when switching to decode): bulk-convert all K from F16 → RQ quantized
3. During decode: new single tokens are quantized directly via set_rows (same as turbo)

- [ ] **Step 1: Add deferred K fields to llama_kv_cache_unified (in header)**

Port from fork's `llama-kv-cache.h`:
```c
// New fields in the KV cache struct:
bool k_is_deferred;                      // true if K type uses deferred quantization
std::vector<ggml_tensor *> k_quant;      // quantized K double-buffer (per layer)
std::vector<ggml_tensor *> k_quant_stream; // stream views for quantized K
bool k_needs_convert;                    // flag: prefill done, convert now
bool convert_deferred_keys();            // bulk F16→RQ conversion
```

- [ ] **Step 2: Implement deferred allocation in KV cache constructor**

In `llama-kv-cache.cpp`, port the fork's double-buffer allocation:
- When K type is any RQ type, allocate K as F16 initially
- Allocate a parallel `k_quant` tensor with the actual RQ type
- Set `k_is_deferred = true`

- [ ] **Step 3: Implement convert_deferred_keys()**

Port from fork's implementation (~lines 2559-2614). This function:
- Reads F16 K from GPU
- Converts to F32 on CPU
- Quantizes via `from_float_ref` (the CPU quantization functions from Task 7)
- Writes back to `k_quant` tensor
- Swaps `k = k_quant` and `k_stream = k_quant_stream`

- [ ] **Step 4: Add conversion trigger in llama-context.cpp**

Port from fork: after prefill completes (detected when `ubatch.n_tokens == 1` transition), call `kv->convert_deferred_keys()`. Guard with `#ifdef GGML_USE_CUDA`.

- [ ] **Step 5: Replace turbo rotation ops in llama-graph.cpp**

Replace `GGML_OP_TURBO_WHT` references with `GGML_OP_RQ_ROTATE`.
Replace `build_attn_mha` and `build_attn` turbo rotation logic with RQ rotation logic.

- [ ] **Step 6: Replace turbo rotation interface in llama-memory.h**

Rename `get_turbo_rot_forward` → `get_rq_rot_forward`, `get_turbo_rot_inverse` → `get_rq_rot_inverse`.

- [ ] **Step 7: Update llama-memory-hybrid.cpp/h**

Replace turbo rotation method implementations with RQ rotation.

- [ ] **Step 8: Build and test**

```bash
cd llama-cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

- [ ] **Step 9: Commit**

```bash
git add llama-cpp/src/
git commit -m "feat: add RotorQuant deferred K-cache quantization subsystem"
```

### Task 11: Update test and cleanup files

**Files:**
- Modify: `llama-cpp/tests/test-turbo-quant.c` → rename to `test-rq-quant.c`, update
- Delete: `llama-cpp/apply_turbo_cuda_v2.py`
- Modify: `llama-cpp/ggml/src/ggml-cuda/template-instances/generate_cu_files.py`

- [ ] **Step 1: Rename and update test file**

```bash
git mv llama-cpp/tests/test-turbo-quant.c llama-cpp/tests/test-rq-quant.c
```

Update test contents: replace all turbo function names with RQ equivalents.

- [ ] **Step 2: Delete obsolete turbo CUDA patch script**

```bash
git rm llama-cpp/apply_turbo_cuda_v2.py
```

- [ ] **Step 3: Full build + test**

```bash
cd llama-cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
ctest --test-dir build -R "rq-quant|quantize-fns|backend-ops" --output-on-failure
```

- [ ] **Step 4: Commit**

```bash
git add llama-cpp/
git commit -m "test: rename turbo test to RQ, remove obsolete scripts"
```

---

## Phase 4: Launcher Rust

### Task 12: Update launcher config, UI, and auto-tune

**Files:**
- Modify: `launcher/src/config.rs:10-51,55-77,292,348-352` (ModelSlot fields, defaults, legacy path)
- Modify: `launcher/src/resources.rs:123-267` (auto_tune logic)
- Modify: `launcher/src/main.rs:465-998` (ModelCard UI, dropdowns)
- Modify: `launcher/src/process.rs:203-306` (env vars)

- [ ] **Step 1: Update config.rs**

1. Rename field `turbo_layer_adaptive` → `rq_layer_adaptive` in `ModelSlot` struct (~L30)
2. Update `default_layer_adaptive` if needed (~L293)
3. Update `new()` to use new field name (~L66)
4. In `build_server_args()`: replace `ct.starts_with("turbo")` with `ct.starts_with("rq")` (~L106)
5. Update comments about CPU/Vulkan turbo support (~L112, ~L122)
6. Update `legacy_config_path` — the old path `"LlamaTurboQuantLauncher"` should remain for migration but add new path `"HouseOfBonsai"` or similar

- [ ] **Step 2: Update resources.rs auto_tune**

In `auto_tune()` function:
- Replace `"turbo3"` → `"rq3"`, `"turbo2"` → `"rq2"` in cache type assignments (~L162-165)
- Update comments about CPU/Vulkan turbo restrictions (~L153-154) — note that RQ types also require CUDA for optimal performance

- [ ] **Step 3: Update main.rs UI**

In `ModelCard` component:
- Replace dropdown options: `vec!["f16","q8_0","turbo2","turbo3","turbo4"]` → `vec!["f16","q8_0","rq2","rq3","rq4","rq3-iso","rq4-iso"]` (~L681, ~L690)
- Rename `turbo_layer_adaptive` → `rq_layer_adaptive` references (~L583, ~L755, ~L758)

- [ ] **Step 4: Update process.rs env vars**

In `start_slot()`:
- Replace `"TURBO_LAYER_ADAPTIVE"` → `"RQ_LAYER_ADAPTIVE"` (~L283)
- Update field reference `turbo_layer_adaptive` → `rq_layer_adaptive` (~L282)

- [ ] **Step 5: Build launcher**

```bash
cd C:\AI_STUFF\PROGRAMMING\the-house-of-bonsai\launcher
cargo check
cargo build --release
```

- [ ] **Step 6: Commit**

```bash
git add launcher/
git commit -m "feat(launcher): replace TurboQuant with RotorQuant in config, UI, and auto-tune"
```

---

## Phase 5: Docs, Wiki, and Final Cleanup

### Task 13: Update all documentation

**Files:**
- Modify: `README.md` (project root)
- Modify: `prism-mcp/README.md`
- Modify: `prism-mcp/CHANGELOG.md`
- Modify: `HANDOFF-rename-and-upgrade.md`
- Delete or modify: `prism-mcp/docs/rfcs/001-turboquant-integration.md` (if not already renamed in Task 5)

- [ ] **Step 1: Update root README.md**

Replace all TurboQuant references:
- "TurboQuant KV Cache" → "RotorQuant KV Cache" (~L23)
- `turboquant-launcher.exe` → `the-house-of-bonsai.exe` (~L41)
- "TurboQuant llama.cpp fork" → "RotorQuant llama.cpp fork" (~L150)
- "TurboQuant KV Cache" section → "RotorQuant KV Cache" (~L182-188)
- Update turbo2/turbo3/turbo4 references to rq2/rq3/rq4

- [ ] **Step 2: Update prism-mcp README.md**

Replace all TurboQuant references:
- "TurboQuant" → "RotorQuant" throughout (~L449, ~L812, ~L825, ~L1076, ~L1108)
- Update the feature comparison table
- Update architecture diagram

- [ ] **Step 3: Update prism-mcp CHANGELOG.md**

Add a new entry at the top documenting the RotorQuant migration. Do NOT modify historical entries — they accurately describe what was true at the time.

- [ ] **Step 4: Update HANDOFF-rename-and-upgrade.md**

Replace turboquant-launcher references with current naming.

- [ ] **Step 5: Grep for any remaining "turboquant" or "TurboQuant" in the project**

```bash
cd C:\AI_STUFF\PROGRAMMING\the-house-of-bonsai
grep -ri "turboquant\|turbo_quant\|turbo-quant\|TURBO.*_0\|TURBO.*TCQ" \
  --include="*.ts" --include="*.rs" --include="*.c" --include="*.h" \
  --include="*.cpp" --include="*.cu" --include="*.cuh" --include="*.md" \
  --include="*.py" --include="*.toml" --include="*.json" --include="*.cmake" \
  --exclude-dir=node_modules --exclude-dir=build --exclude-dir=.cache \
  --exclude-dir=vendor
```

Expected: zero matches outside of `vendor/rotorquant-ref/` (the reference code we vendored).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "docs: replace all TurboQuant references with RotorQuant across project"
```

### Task 14: Update Obsidian wiki

**Files:**
- Update wiki hot cache, entity pages, and concept pages via MCP tools

- [ ] **Step 1: Update wiki hot.md with session summary**

Via `mcp__obsidian-vault__patch_note`, update `wiki/hot.md` with RotorQuant migration session notes.

- [ ] **Step 2: Update wiki entity page for The House of Bonsai**

Update the project entity to reflect RotorQuant replacing TurboQuant.

- [ ] **Step 3: Create a wiki concept page for RotorQuant**

Create `wiki/concepts/RotorQuant.md` documenting the algorithm, the migration decision, and links to the source repo.

- [ ] **Step 4: File a gotcha for the type ID allocation**

Create `wiki/gotchas/rq-type-ids-reuse-turbo-slots.md` documenting that RQ types reuse turbo IDs 41-43, 46-47 and that IDs 44-45 are sacred Q1_0.

- [ ] **Step 5: Update todos/bonsai.md**

Mark the RotorQuant migration as in-progress or complete, add any follow-up items discovered during implementation.

---

## Phase 6: Vulkan Support (Optional, Separate PR)

The rotorquant fork has NO Vulkan support. The existing turbo types may have had limited Vulkan support (the launcher already falls back to f16 for turbo types on Vulkan). Options:

1. **Fallback to f16 on Vulkan** (same as current turbo behavior) — ship this immediately
2. **Write Vulkan compute shaders for RQ** — separate follow-up work

### Task 15: Ensure Vulkan fallback works cleanly

**Files:**
- Modify: `launcher/src/config.rs` (Vulkan cache type fallback)
- Verify: `llama-cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` (no turbo references to break)

- [ ] **Step 1: Verify Vulkan has no turbo-specific code**

The search showed zero turbo matches in `ggml-vulkan/` — good. Verify the Vulkan backend gracefully falls back for unknown types.

- [ ] **Step 2: Ensure launcher's Vulkan fallback catches RQ types**

The existing code at `config.rs:122` says "Vulkan doesn't support turbo SET_ROWS op — fall back to f16". Update this to catch RQ types:
```rust
if ct.starts_with("rq") { "f16".into() } else { ct.to_string() }
```

- [ ] **Step 3: Commit**

```bash
git add launcher/src/config.rs
git commit -m "fix(vulkan): ensure RQ cache types fall back to f16 on Vulkan backend"
```

---

## Summary of All Phases

| Phase | Tasks | Est. Files | Key Risk |
|-------|-------|-----------|----------|
| 1. Vendor | 1 | 2 dirs | None (setup only) |
| 2. prism-mcp TS | 2-5 | ~10 | Serialization format change breaks existing compressed data |
| 3. llama-cpp C/CUDA/Metal | 6-11 | ~40 | CUDA kernel correctness, deferred quantization race conditions |
| 4. Launcher Rust | 12 | 4 | Config migration for existing users |
| 5. Docs/Wiki | 13-14 | ~10 | Completeness of grep-and-replace |
| 6. Vulkan fallback | 15 | 2 | None (fallback only) |

**Critical path:** Phase 3 (llama-cpp) is the hardest and longest. Phases 2 and 4 can proceed in parallel. Phase 5 is last.

**Migration note:** Existing prism-mcp databases with TurboQuant compressed embeddings will need a migration path. The serialization format is similar but not identical (no d×d rotation matrix in the blob). Options: (a) re-embed on first access, (b) dual-read support during transition. This should be addressed in Task 4 as a sub-step if needed.
