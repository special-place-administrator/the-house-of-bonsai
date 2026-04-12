/**
 * RotorQuant — Pure TypeScript Vector Quantization with PlanarQuant
 * ═══════════════════════════════════════════════════════════════════════════
 * Replacement for TurboQuant using Givens 2D rotation (PlanarQuant) instead
 * of full QR-decomposed d*d orthogonal matrix.
 *
 * KEY CHANGE FROM TURBOQUANT:
 *   - TurboQuant: O(d^3) QR decomposition -> d*d rotation matrix (4.7 MB at d=768)
 *   - RotorQuant: O(d) Givens rotations -> d/2 (cos, sin) pairs (~6 KB at d=768)
 *
 *   The Givens rotation decorrelates pairs of adjacent coordinates independently.
 *   Each pair (v[2i], v[2i+1]) is rotated by a random angle theta_i.
 *   This provides less decorrelation than full QR rotation (no cross-pair mixing),
 *   but is dramatically cheaper to compute and store.
 *
 * PIPELINE (Two-Stage Compression):
 *   Stage 1: Givens Rotation -> Per-Coordinate Lloyd-Max Quantization (MSE)
 *   Stage 2: 1-bit QJL (Quantized Johnson-Lindenstrauss) Residual Correction
 *
 * The serialization format is WIRE-COMPATIBLE with TurboQuant:
 *   same fields: d, bits, radius, residualNorm, mseIndices, qjlSigns
 *
 * @module rotorquant
 */

// ─── Types ───────────────────────────────────────────────────────

export interface RotorQuantConfig {
  /** Vector dimension (must be even). Prism default: 768 */
  d: number;
  /** Total bits per coordinate: MSE uses (bits-1), QJL uses 1 bit. Min: 2 */
  bits: number;
  /** Random seed for reproducibility (default: 42) */
  seed?: number;
}

export interface CompressedEmbedding {
  /** Bit-packed MSE codebook indices: ceil(d * mseBits / 8) bytes */
  mseIndices: Uint8Array;
  /** Bit-packed QJL sign bits: ceil(d / 8) bytes */
  qjlSigns: Uint8Array;
  /** L2 norm of quantization residual */
  residualNorm: number;
  /** Original vector L2 norm (magnitude) */
  radius: number;
  /** Config used for compression */
  config: { d: number; bits: number };
}

export interface LloydMaxCodebook {
  /** Optimal centroid values, sorted ascending */
  centroids: Float64Array;
  /** Decision boundaries between adjacent centroids */
  boundaries: Float64Array;
  /** Number of quantization levels = 2^bits */
  nLevels: number;
  /** Bits per coordinate */
  bits: number;
}

// ─── Seeded PRNG (Mulberry32) ────────────────────────────────────

function mulberry32(seed: number): () => number {
  let t = seed | 0;
  return () => {
    t = (t + 0x6d2b79f5) | 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

/** Generate N(0,1) using Box-Muller transform */
function gaussianRandom(rng: () => number): number {
  const u1 = rng();
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1 + 1e-15)) * Math.cos(2 * Math.PI * u2);
}

// ─── Numerical Integration (Simpson's Rule) ──────────────────────

function integrate(f: (x: number) => number, a: number, b: number, n = 1000): number {
  if (n % 2 !== 0) n++;
  const h = (b - a) / n;
  let sum = f(a) + f(b);
  for (let i = 1; i < n; i++) {
    const x = a + i * h;
    sum += (i % 2 === 0 ? 2 : 4) * f(x);
  }
  return (h / 3) * sum;
}

// ─── Lloyd-Max Codebook Solver ───────────────────────────────────

function gaussianPdf(x: number, d: number): number {
  const sigma2 = 1.0 / d;
  return (1.0 / Math.sqrt(2 * Math.PI * sigma2)) * Math.exp(-x * x / (2 * sigma2));
}

export function solveLloydMax(d: number, bits: number): LloydMaxCodebook {
  const nLevels = 1 << bits;
  const sigma = 1.0 / Math.sqrt(d);
  const pdf = (x: number) => gaussianPdf(x, d);

  const lo = -3.5 * sigma;
  const hi = 3.5 * sigma;

  const centroids = new Float64Array(nLevels);
  for (let i = 0; i < nLevels; i++) {
    centroids[i] = lo + (hi - lo) * (i + 0.5) / nLevels;
  }

  const boundaries = new Float64Array(nLevels - 1);

  for (let iter = 0; iter < 200; iter++) {
    for (let i = 0; i < nLevels - 1; i++) {
      boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
    }

    let maxShift = 0;
    for (let i = 0; i < nLevels; i++) {
      const a = i === 0 ? lo * 3 : boundaries[i - 1];
      const b = i === nLevels - 1 ? hi * 3 : boundaries[i];

      const numerator = integrate((x) => x * pdf(x), a, b);
      const denominator = integrate(pdf, a, b);

      const newCentroid = denominator > 1e-15 ? numerator / denominator : centroids[i];
      maxShift = Math.max(maxShift, Math.abs(newCentroid - centroids[i]));
      centroids[i] = newCentroid;
    }

    if (maxShift < 1e-10) break;
  }

  for (let i = 0; i < nLevels - 1; i++) {
    boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
  }

  return { centroids, boundaries, nLevels, bits };
}

// ─── Codebook Cache ──────────────────────────────────────────────

const codebookCache = new Map<string, LloydMaxCodebook>();

function getCodebook(d: number, bits: number): LloydMaxCodebook {
  const key = `${d}:${bits}`;
  let cb = codebookCache.get(key);
  if (!cb) {
    cb = solveLloydMax(d, bits);
    codebookCache.set(key, cb);
  }
  return cb;
}

function quantizeValue(value: number, codebook: LloydMaxCodebook): number {
  const { boundaries, nLevels } = codebook;
  let lo = 0;
  let hi = nLevels - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (mid < boundaries.length && value > boundaries[mid]) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// ─── Givens 2D Rotation ─────────────────────────────────────────
//
// Replaces TurboQuant's d*d QR-decomposed orthogonal matrix with d/2
// independent 2D Givens rotations. Each pair (v[2i], v[2i+1]) is
// rotated by a random angle theta_i.
//
// Memory: 2 * d/2 * 4 bytes = d * 4 bytes (e.g., 3 KB at d=768)
// vs TurboQuant: d*d * 8 bytes = 4.7 MB at d=768
//
// Computation: O(d) per rotation vs O(d^2) for matvec.

/**
 * Generate d/2 random Givens rotation parameters (cos theta, sin theta).
 * Uses deterministic PRNG seeded from the config seed.
 */
function generateGivensRotations(d: number, seed: number): { cos: Float32Array; sin: Float32Array } {
  const nPairs = d / 2;
  const rng = mulberry32(seed);
  const cos = new Float32Array(nPairs);
  const sin = new Float32Array(nPairs);

  for (let i = 0; i < nPairs; i++) {
    const angle = rng() * 2 * Math.PI;
    cos[i] = Math.cos(angle);
    sin[i] = Math.sin(angle);
  }

  return { cos, sin };
}

/**
 * Apply forward Givens rotation to a vector.
 * For each pair i: out[2i]   = cos[i]*vec[2i] - sin[i]*vec[2i+1]
 *                  out[2i+1] = sin[i]*vec[2i] + cos[i]*vec[2i+1]
 */
function applyGivensForward(
  vec: Float64Array,
  cos: Float32Array,
  sin: Float32Array,
  out: Float64Array,
): void {
  const nPairs = cos.length;
  for (let i = 0; i < nPairs; i++) {
    const c = cos[i];
    const s = sin[i];
    const v0 = vec[2 * i];
    const v1 = vec[2 * i + 1];
    out[2 * i] = c * v0 - s * v1;
    out[2 * i + 1] = s * v0 + c * v1;
  }
}

/**
 * Apply inverse Givens rotation (transpose = negate sin).
 * For each pair i: out[2i]   = cos[i]*vec[2i] + sin[i]*vec[2i+1]
 *                  out[2i+1] = -sin[i]*vec[2i] + cos[i]*vec[2i+1]
 */
function applyGivensInverse(
  vec: Float64Array,
  cos: Float32Array,
  sin: Float32Array,
  out: Float64Array,
): void {
  const nPairs = cos.length;
  for (let i = 0; i < nPairs; i++) {
    const c = cos[i];
    const s = sin[i];
    const v0 = vec[2 * i];
    const v1 = vec[2 * i + 1];
    out[2 * i] = c * v0 + s * v1;
    out[2 * i + 1] = -s * v0 + c * v1;
  }
}

// ─── QJL Random Projection Matrix ────────────────────────────────

function generateQJLMatrix(d: number, seed: number, m?: number): Float64Array {
  m = m ?? d;
  const rng = mulberry32(seed);
  const S = new Float64Array(m * d);
  for (let i = 0; i < m * d; i++) {
    S[i] = gaussianRandom(rng);
  }
  return S;
}

// ─── Matrix-Vector Operations ────────────────────────────────────

function matvec(M: Float64Array, x: Float64Array | number[], rows: number, cols: number): Float64Array {
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    let sum = 0;
    const offset = i * cols;
    for (let j = 0; j < cols; j++) {
      sum += M[offset + j] * x[j];
    }
    y[i] = sum;
  }
  return y;
}

function dot(a: Float64Array | number[], b: Float64Array | number[], len: number): number {
  let sum = 0;
  for (let i = 0; i < len; i++) sum += a[i] * b[i];
  return sum;
}

function norm(a: Float64Array, len: number): number {
  return Math.sqrt(dot(a, a, len));
}

// ─── Bit Packing ─────────────────────────────────────────────────

export function packBits(values: Uint16Array, bits: number): Uint8Array {
  const totalBits = values.length * bits;
  const packedLen = Math.ceil(totalBits / 8);
  const packed = new Uint8Array(packedLen);

  let bitPos = 0;
  for (let i = 0; i < values.length; i++) {
    let val = values[i];
    let bitsRemaining = bits;
    while (bitsRemaining > 0) {
      const byteIdx = bitPos >> 3;
      const bitOffset = bitPos & 7;
      const bitsAvailable = 8 - bitOffset;
      const bitsToWrite = Math.min(bitsRemaining, bitsAvailable);
      const mask = (1 << bitsToWrite) - 1;
      packed[byteIdx] |= (val & mask) << bitOffset;
      val >>= bitsToWrite;
      bitsRemaining -= bitsToWrite;
      bitPos += bitsToWrite;
    }
  }

  return packed;
}

export function unpackBits(packed: Uint8Array, bits: number, count: number): Uint16Array {
  const values = new Uint16Array(count);
  let bitPos = 0;

  for (let i = 0; i < count; i++) {
    let val = 0;
    let bitsRemaining = bits;
    let shift = 0;
    while (bitsRemaining > 0) {
      const byteIdx = bitPos >> 3;
      const bitOffset = bitPos & 7;
      const bitsAvailable = 8 - bitOffset;
      const bitsToRead = Math.min(bitsRemaining, bitsAvailable);
      const mask = (1 << bitsToRead) - 1;
      val |= ((packed[byteIdx] >> bitOffset) & mask) << shift;
      shift += bitsToRead;
      bitsRemaining -= bitsToRead;
      bitPos += bitsToRead;
    }
    values[i] = val;
  }

  return values;
}

function packSigns(signs: Float64Array): Uint8Array {
  const packedLen = Math.ceil(signs.length / 8);
  const packed = new Uint8Array(packedLen);
  for (let i = 0; i < signs.length; i++) {
    if (signs[i] >= 0) {
      packed[i >> 3] |= 1 << (i & 7);
    }
  }
  return packed;
}

function unpackSigns(packed: Uint8Array, count: number): Float64Array {
  const signs = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    signs[i] = (packed[i >> 3] >> (i & 7)) & 1 ? 1.0 : -1.0;
  }
  return signs;
}

// ─── RotorQuant Compressor ───────────────────────────────────────

export class RotorQuantCompressor {
  readonly d: number;
  readonly bits: number;
  readonly mseBits: number;
  readonly codebook: LloydMaxCodebook;
  readonly cos: Float32Array;   // d/2 cosine values
  readonly sin: Float32Array;   // d/2 sine values
  readonly S: Float64Array;     // d*d QJL projection matrix

  constructor(config: RotorQuantConfig) {
    if (config.d % 2 !== 0) {
      throw new Error(`RotorQuantConfig.d must be even; got ${config.d}`);
    }
    if (config.bits < 2 || config.bits > 6) {
      throw new Error(`RotorQuantConfig.bits must be in [2, 6]; got ${config.bits}`);
    }
    this.d = config.d;
    this.bits = config.bits;
    this.mseBits = Math.max(config.bits - 1, 1);
    this.codebook = getCodebook(config.d, this.mseBits);

    const seed = config.seed ?? 42;
    const givens = generateGivensRotations(config.d, seed);
    this.cos = givens.cos;
    this.sin = givens.sin;

    this.S = generateQJLMatrix(config.d, seed + 1);
  }

  /**
   * Compress a float32/float64 embedding vector.
   *
   * Pipeline:
   *   1. Normalize to unit vector (store radius/magnitude)
   *   2. Apply Givens forward rotation
   *   3. Per-coordinate Lloyd-Max quantization -> indices
   *   4. Dequantize -> inverse rotate -> compute MSE reconstruction
   *   5. Compute residual = original - MSE reconstruction
   *   6. QJL: project residual through S, keep sign bits
   */
  compress(embedding: number[]): CompressedEmbedding {
    const d = this.d;
    if (embedding.length !== d) {
      throw new Error(`Expected ${d}-dim vector, got ${embedding.length}`);
    }

    // Step 1: Normalize
    const vec = new Float64Array(embedding);
    const radius = norm(vec, d);
    const normalized = new Float64Array(d);
    if (radius > 1e-15) {
      for (let i = 0; i < d; i++) normalized[i] = vec[i] / radius;
    }

    // Step 2: Forward Givens rotation
    const rotated = new Float64Array(d);
    applyGivensForward(normalized, this.cos, this.sin, rotated);

    // Step 3: Per-coordinate quantization
    const indices = new Uint16Array(d);
    for (let i = 0; i < d; i++) {
      indices[i] = quantizeValue(rotated[i], this.codebook);
    }

    // Step 4: Dequantize -> inverse rotate -> MSE reconstruction
    const dequantized = new Float64Array(d);
    for (let i = 0; i < d; i++) {
      dequantized[i] = this.codebook.centroids[indices[i]];
    }
    // Inverse rotate back to original space
    const mseNorm = new Float64Array(d);
    applyGivensInverse(dequantized, this.cos, this.sin, mseNorm);
    // Scale back
    const mse = new Float64Array(d);
    for (let i = 0; i < d; i++) mse[i] = mseNorm[i] * radius;

    // Step 5: Residual
    const residual = new Float64Array(d);
    for (let i = 0; i < d; i++) residual[i] = vec[i] - mse[i];
    const residualNorm = norm(residual, d);

    // Step 6: QJL sign bits
    const projected = matvec(this.S, residual, d, d);
    const qjlSigns = packSigns(projected);

    // Pack MSE indices
    const mseIndicesPacked = packBits(indices, this.mseBits);

    return {
      mseIndices: mseIndicesPacked,
      qjlSigns,
      residualNorm,
      radius,
      config: { d, bits: this.bits },
    };
  }

  /**
   * Compute unbiased inner product estimate <query, compressed_vec>.
   *
   * Term 1: <query, x_mse> — dequantize, inverse rotate, scale, dot with query
   * Term 2: <query, residual> — QJL sign-bit correction
   */
  asymmetricInnerProduct(query: number[], compressed: CompressedEmbedding): number {
    const d = this.d;

    // Reconstruct MSE vector: unpack -> dequantize -> inverse rotate -> scale
    const indices = unpackBits(compressed.mseIndices, this.mseBits, d);
    const dequantized = new Float64Array(d);
    for (let i = 0; i < d; i++) {
      dequantized[i] = this.codebook.centroids[indices[i]];
    }
    const mseUnit = new Float64Array(d);
    applyGivensInverse(dequantized, this.cos, this.sin, mseUnit);

    const mse = new Float64Array(d);
    for (let i = 0; i < d; i++) mse[i] = mseUnit[i] * compressed.radius;

    // Term 1: <query, x_mse>
    const term1 = dot(query, mse, d);

    // Term 2: QJL correction
    const signs = unpackSigns(compressed.qjlSigns, d);
    const qProjected = matvec(this.S, new Float64Array(query), d, d);
    const qjlIp = dot(qProjected, signs, d);

    const m = d;
    const correctionScale = Math.sqrt(Math.PI / 2) / m;
    const term2 = compressed.residualNorm * correctionScale * qjlIp;

    return term1 + term2;
  }

  /**
   * Compute cosine similarity between a query (float) and compressed vector.
   */
  asymmetricCosineSimilarity(query: number[], compressed: CompressedEmbedding): number {
    const ip = this.asymmetricInnerProduct(query, compressed);
    const queryNorm = Math.sqrt(dot(query, query, query.length));
    if (queryNorm < 1e-15 || compressed.radius < 1e-15) return 0;
    return ip / (queryNorm * compressed.radius);
  }
}

// ─── Serialization ───────────────────────────────────────────────
// Wire-compatible with TurboQuant binary format.

export function serialize(compressed: CompressedEmbedding): Buffer {
  const mseLen = compressed.mseIndices.length;
  const qjlLen = compressed.qjlSigns.length;
  const totalLen = 16 + mseLen + qjlLen;

  const buf = Buffer.alloc(totalLen);
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);

  view.setUint16(0, compressed.config.d, true);
  view.setUint8(2, compressed.config.bits);
  // bytes 3-7: reserved
  view.setFloat32(8, compressed.radius, true);
  view.setFloat32(12, compressed.residualNorm, true);

  buf.set(compressed.mseIndices, 16);
  buf.set(compressed.qjlSigns, 16 + mseLen);

  return buf;
}

export function deserialize(buf: Buffer): CompressedEmbedding {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);

  const d = view.getUint16(0, true);
  const bits = view.getUint8(2);
  const radius = view.getFloat32(8, true);
  const residualNorm = view.getFloat32(12, true);

  const mseBits = Math.max(bits - 1, 1);
  const mseLen = Math.ceil(d * mseBits / 8);
  const qjlLen = Math.ceil(d / 8);

  const mseIndices = new Uint8Array(buf.slice(16, 16 + mseLen));
  const qjlSigns = new Uint8Array(buf.slice(16 + mseLen, 16 + mseLen + qjlLen));

  return {
    mseIndices,
    qjlSigns,
    residualNorm,
    radius,
    config: { d, bits },
  };
}

// ─── Default Prism Compressor ────────────────────────────────────

/** Default config for Prism's 768-dim Gemini embeddings at 3-bit quantization */
export const PRISM_DEFAULT_CONFIG: RotorQuantConfig = {
  d: 768,
  bits: 3,
  seed: 42,
};

let _defaultCompressor: RotorQuantCompressor | null = null;

/** Get or create the default Prism compressor (lazy singleton) */
export function getDefaultCompressor(): RotorQuantCompressor {
  if (!_defaultCompressor) {
    _defaultCompressor = new RotorQuantCompressor(PRISM_DEFAULT_CONFIG);
  }
  return _defaultCompressor;
}
