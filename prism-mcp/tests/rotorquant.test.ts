/**
 * RotorQuant — Test Suite (TDD Red Phase)
 * ═════════════════════════════════════════════════════════════════════════
 *
 * Tests for the PlanarQuant (Givens 2D rotation) replacement of TurboQuant.
 * RotorQuant uses d/2 independent Givens rotations instead of a d*d
 * orthogonal matrix, achieving O(d) rotation cost instead of O(d^3).
 *
 * Run: npx vitest run tests/rotorquant.test.ts
 */

import { describe, it, expect, beforeAll } from "vitest";
import {
  RotorQuantCompressor,
  solveLloydMax,
  serialize,
  deserialize,
  packBits,
  unpackBits,
  getDefaultCompressor,
  PRISM_DEFAULT_CONFIG,
  type RotorQuantConfig,
} from "../src/utils/rotorquant.js";

// ─── Helpers ─────────────────────────────────────────────────────

function mulberry32(seed: number): () => number {
  let t = seed | 0;
  return () => {
    t = (t + 0x6d2b79f5) | 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianRandom(rng: () => number): number {
  const u1 = rng();
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1 + 1e-15)) * Math.cos(2 * Math.PI * u2);
}

function randomUnitVector(d: number, rng: () => number): number[] {
  const v = Array.from({ length: d }, () => gaussianRandom(rng));
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return v.map((x) => x / norm);
}

function randomVector(d: number, rng: () => number): number[] {
  return Array.from({ length: d }, () => gaussianRandom(rng));
}

function cosineSim(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function dotProduct(a: number[], b: number[]): number {
  return a.reduce((s, v, i) => s + v * b[i], 0);
}

// ─── Test Constants ──────────────────────────────────────────────

// Use d=128 for fast tests; bits=4 for quality-sensitive tests
const FAST_CONFIG: RotorQuantConfig = { d: 128, bits: 4, seed: 42 };
const FAST_3BIT_CONFIG: RotorQuantConfig = { d: 128, bits: 3, seed: 42 };

// ─── 1. Givens Rotation Preserves Norms ──────────────────────────

describe("Givens Rotation Norm Preservation", () => {
  it("rotating a vector preserves its L2 norm", () => {
    const compressor = new RotorQuantCompressor(FAST_CONFIG);
    const rng = mulberry32(123);

    for (let trial = 0; trial < 20; trial++) {
      const v = randomUnitVector(128, rng);
      const vNorm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
      // Unit vectors should have norm ~1.0
      expect(Math.abs(vNorm - 1.0)).toBeLessThan(1e-10);

      // Compress and check that radius matches original norm
      const compressed = compressor.compress(v);
      expect(Math.abs(compressed.radius - 1.0)).toBeLessThan(0.01);
    }
  });

  it("non-unit vector radius matches original L2 norm", () => {
    const compressor = new RotorQuantCompressor(FAST_CONFIG);
    const rng = mulberry32(42);

    for (let trial = 0; trial < 10; trial++) {
      const v = randomVector(128, rng);
      const vNorm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
      const compressed = compressor.compress(v);
      expect(Math.abs(compressed.radius - vNorm)).toBeLessThan(1e-4);
    }
  });
});

// ─── 2. Compress/Decompress Roundtrip ────────────────────────────

describe("Compress/Decompress Roundtrip", () => {
  it("self-similarity > 0.95 (vector vs its own compressed form)", () => {
    const compressor = new RotorQuantCompressor(FAST_CONFIG);
    const rng = mulberry32(42);

    for (let trial = 0; trial < 20; trial++) {
      const v = randomUnitVector(128, rng);
      const compressed = compressor.compress(v);
      const selfSim = compressor.asymmetricCosineSimilarity(v, compressed);
      expect(selfSim).toBeGreaterThan(0.95);
    }
  });

  it("self-similarity > 0.95 at 3-bit config", () => {
    const compressor = new RotorQuantCompressor(FAST_3BIT_CONFIG);
    const rng = mulberry32(42);

    for (let trial = 0; trial < 20; trial++) {
      const v = randomUnitVector(128, rng);
      const compressed = compressor.compress(v);
      const selfSim = compressor.asymmetricCosineSimilarity(v, compressed);
      expect(selfSim).toBeGreaterThan(0.95);
    }
  });
});

// ─── 3. Zero Vector Handling ─────────────────────────────────────

describe("Zero Vector Handling", () => {
  it("handles zero vector gracefully", () => {
    const compressor = new RotorQuantCompressor(FAST_CONFIG);
    const zero = new Array(128).fill(0);
    const compressed = compressor.compress(zero);
    expect(compressed.radius).toBeLessThan(1e-10);
    expect(compressed.residualNorm).toBeLessThan(1e-10);
  });

  it("cosine similarity with zero vector returns 0", () => {
    const compressor = new RotorQuantCompressor(FAST_CONFIG);
    const zero = new Array(128).fill(0);
    const compressed = compressor.compress(zero);
    const query = randomUnitVector(128, mulberry32(42));
    const sim = compressor.asymmetricCosineSimilarity(query, compressed);
    expect(sim).toBe(0);
  });
});

// ─── 4. Unit Vector Handling ─────────────────────────────────────

describe("Unit Vector Handling", () => {
  it("unit vectors have radius ~1.0", () => {
    const compressor = new RotorQuantCompressor(FAST_CONFIG);
    const rng = mulberry32(42);

    for (let trial = 0; trial < 10; trial++) {
      const v = randomUnitVector(128, rng);
      const compressed = compressor.compress(v);
      expect(compressed.radius).toBeGreaterThan(0.99);
      expect(compressed.radius).toBeLessThan(1.01);
    }
  });

  it("throws on wrong dimension", () => {
    const compressor = new RotorQuantCompressor(FAST_CONFIG);
    expect(() => compressor.compress(new Array(64).fill(0))).toThrow("Expected 128-dim");
  });

  it("rejects odd dimensions in constructor", () => {
    expect(() => new RotorQuantCompressor({ d: 127, bits: 4, seed: 42 })).toThrow();
  });
});

// ─── 5. Asymmetric Inner Product Approximation ───────────────────

describe("Asymmetric Inner Product", () => {
  it("mean bias of asymmetric estimator < 0.05 across 200 random pairs", () => {
    const compressor = new RotorQuantCompressor(FAST_CONFIG);
    const rng = mulberry32(42);
    const nPairs = 200;
    let totalBias = 0;

    for (let i = 0; i < nPairs; i++) {
      const a = randomUnitVector(128, rng);
      const b = randomUnitVector(128, rng);
      const compressed = compressor.compress(b);

      const trueIP = dotProduct(a, b);
      const estIP = compressor.asymmetricInnerProduct(a, compressed);
      totalBias += estIP - trueIP;
    }

    const meanBias = Math.abs(totalBias / nPairs);
    expect(meanBias).toBeLessThan(0.05);
  });

  it("asymmetric cosine similarity correlates >0.70 with true similarity (4-bit)", () => {
    const compressor = new RotorQuantCompressor(FAST_CONFIG);
    const rng = mulberry32(42);
    const nPairs = 100;
    const trueSims: number[] = [];
    const estSims: number[] = [];

    for (let i = 0; i < nPairs; i++) {
      const a = randomUnitVector(128, rng);
      const b = randomUnitVector(128, rng);
      const compressed = compressor.compress(b);

      trueSims.push(cosineSim(a, b));
      estSims.push(compressor.asymmetricCosineSimilarity(a, compressed));
    }

    const meanTrue = trueSims.reduce((s, x) => s + x, 0) / nPairs;
    const meanEst = estSims.reduce((s, x) => s + x, 0) / nPairs;
    let cov = 0, varTrue = 0, varEst = 0;
    for (let i = 0; i < nPairs; i++) {
      const dt = trueSims[i] - meanTrue;
      const de = estSims[i] - meanEst;
      cov += dt * de;
      varTrue += dt * dt;
      varEst += de * de;
    }
    const correlation = cov / (Math.sqrt(varTrue) * Math.sqrt(varEst));
    expect(correlation).toBeGreaterThan(0.70);
  });

  it("works with non-normalized query", () => {
    const compressor = new RotorQuantCompressor(FAST_CONFIG);
    const rng = mulberry32(42);
    const target = randomUnitVector(128, rng);
    const query = randomVector(128, rng);

    const compressed = compressor.compress(target);
    const sim = compressor.asymmetricCosineSimilarity(query, compressed);
    expect(sim).toBeGreaterThan(-1.5);
    expect(sim).toBeLessThan(1.5);
  });
});

// ─── 6. Serialize/Deserialize Roundtrip ──────────────────────────

describe("Serialize/Deserialize Roundtrip", () => {
  let compressor: RotorQuantCompressor;

  beforeAll(() => {
    compressor = new RotorQuantCompressor(FAST_CONFIG);
  });

  it("serialize -> deserialize preserves all fields", () => {
    const rng = mulberry32(42);
    const vec = randomUnitVector(128, rng);
    const compressed = compressor.compress(vec);

    const buf = serialize(compressed);
    const restored = deserialize(buf);

    expect(restored.config.d).toBe(compressed.config.d);
    expect(restored.config.bits).toBe(compressed.config.bits);
    expect(Math.abs(restored.radius - compressed.radius)).toBeLessThan(1e-4);
    expect(Math.abs(restored.residualNorm - compressed.residualNorm)).toBeLessThan(1e-4);

    // Byte-level equality of packed data
    for (let i = 0; i < compressed.mseIndices.length; i++) {
      expect(restored.mseIndices[i]).toBe(compressed.mseIndices[i]);
    }
    for (let i = 0; i < compressed.qjlSigns.length; i++) {
      expect(restored.qjlSigns[i]).toBe(compressed.qjlSigns[i]);
    }
  });

  it("similarity is preserved through serialize/deserialize cycle", () => {
    const rng = mulberry32(77);
    const query = randomUnitVector(128, rng);
    const target = randomUnitVector(128, rng);

    const compressed = compressor.compress(target);
    const sim1 = compressor.asymmetricCosineSimilarity(query, compressed);

    const buf = serialize(compressed);
    const restored = deserialize(buf);
    const sim2 = compressor.asymmetricCosineSimilarity(query, restored);

    expect(Math.abs(sim1 - sim2)).toBeLessThan(1e-4);
  });

  it("deterministic: same input + same seed -> identical bytes", () => {
    const vec = randomUnitVector(128, mulberry32(42));
    const c1 = compressor.compress(vec);
    const c2 = compressor.compress(vec);

    const buf1 = serialize(c1);
    const buf2 = serialize(c2);
    expect(Buffer.compare(buf1, buf2)).toBe(0);
  });
});

// ─── 7. packBits/unpackBits Roundtrip ────────────────────────────

describe("packBits / unpackBits Roundtrip", () => {
  it("3-bit values roundtrip correctly", () => {
    const rng = mulberry32(42);
    const count = 128;
    const values = new Uint16Array(count);
    for (let i = 0; i < count; i++) {
      values[i] = Math.floor(rng() * 8); // 0-7 for 3-bit
    }

    const packed = packBits(values, 3);
    const unpacked = unpackBits(packed, 3, count);

    for (let i = 0; i < count; i++) {
      expect(unpacked[i]).toBe(values[i]);
    }
  });

  it("2-bit values roundtrip correctly", () => {
    const rng = mulberry32(99);
    const count = 256;
    const values = new Uint16Array(count);
    for (let i = 0; i < count; i++) {
      values[i] = Math.floor(rng() * 4); // 0-3 for 2-bit
    }

    const packed = packBits(values, 2);
    const unpacked = unpackBits(packed, 2, count);

    for (let i = 0; i < count; i++) {
      expect(unpacked[i]).toBe(values[i]);
    }
  });

  it("packed size is correct", () => {
    const values = new Uint16Array(768);
    const packed = packBits(values, 3);
    expect(packed.length).toBe(Math.ceil(768 * 3 / 8));
  });
});

// ─── 8. getDefaultCompressor Singleton ───────────────────────────

describe("getDefaultCompressor", () => {
  it("returns a singleton instance", () => {
    const a = getDefaultCompressor();
    const b = getDefaultCompressor();
    expect(a).toBe(b);
  });

  it("uses PRISM_DEFAULT_CONFIG", () => {
    const compressor = getDefaultCompressor();
    expect(compressor.d).toBe(PRISM_DEFAULT_CONFIG.d);
    expect(compressor.bits).toBe(PRISM_DEFAULT_CONFIG.bits);
  });

  it("PRISM_DEFAULT_CONFIG is { d: 768, bits: 3, seed: 42 }", () => {
    expect(PRISM_DEFAULT_CONFIG.d).toBe(768);
    expect(PRISM_DEFAULT_CONFIG.bits).toBe(3);
    expect(PRISM_DEFAULT_CONFIG.seed).toBe(42);
  });
});

// ─── 9. Compression Ratio ────────────────────────────────────────

describe("Compression Ratio", () => {
  it("serialized 4-bit d=128 has correct size", () => {
    const compressor = new RotorQuantCompressor(FAST_CONFIG);
    const vec = randomUnitVector(128, mulberry32(42));
    const compressed = compressor.compress(vec);
    const buf = serialize(compressed);

    // Header(16) + ceil(128 * 3 / 8) MSE + ceil(128/8) QJL
    // = 16 + 48 + 16 = 80 bytes
    const mseBits = FAST_CONFIG.bits - 1; // 3
    const expectedMse = Math.ceil(128 * mseBits / 8);
    const expectedQjl = Math.ceil(128 / 8);
    expect(buf.length).toBe(16 + expectedMse + expectedQjl);
  });

  it("serialized 3-bit d=768 < 350 bytes", () => {
    const config: RotorQuantConfig = { d: 768, bits: 3, seed: 42 };
    const compressor = new RotorQuantCompressor(config);
    const rng = mulberry32(42);
    const vec = randomUnitVector(768, rng);

    const compressed = compressor.compress(vec);
    const buf = serialize(compressed);

    expect(buf.length).toBeLessThan(350);
    const float32Size = 768 * 4;
    expect(float32Size / buf.length).toBeGreaterThan(8);
  });
});
