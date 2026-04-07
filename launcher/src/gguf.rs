//! GGUF file header reader.
//!
//! Parses only the key-value metadata section of a `.gguf` file (the first
//! few KB). Tensor data is never touched, so even multi-GB files are fast.

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

// ---------------------------------------------------------------------------
// Value types stored in the GGUF KV header
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    /// Arrays are skipped during parsing; this variant records only the count.
    ArraySkipped { element_type: u32, count: u64 },
}

impl fmt::Display for GgufValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uint8(v)  => write!(f, "{v}"),
            Self::Int8(v)   => write!(f, "{v}"),
            Self::Uint16(v) => write!(f, "{v}"),
            Self::Int16(v)  => write!(f, "{v}"),
            Self::Uint32(v) => write!(f, "{v}"),
            Self::Int32(v)  => write!(f, "{v}"),
            Self::Float32(v) => write!(f, "{v}"),
            Self::Bool(v)   => write!(f, "{v}"),
            Self::String(v) => write!(f, "{v}"),
            Self::Uint64(v) => write!(f, "{v}"),
            Self::Int64(v)  => write!(f, "{v}"),
            Self::Float64(v) => write!(f, "{v}"),
            Self::ArraySkipped { count, .. } => write!(f, "[array; {count} elements]"),
        }
    }
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint32(v) => Some(*v),
            Self::Uint8(v)  => Some(*v as u32),
            Self::Uint16(v) => Some(*v as u32),
            Self::Int32(v)  => Some(*v as u32),
            Self::Uint64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Self::Int32(v)  => Some(*v),
            Self::Int8(v)   => Some(*v as i32),
            Self::Int16(v)  => Some(*v as i32),
            Self::Uint32(v) => Some(*v as i32),
            Self::Uint64(v) => Some(*v as i32),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            Self::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Raw GGUF metadata (HashMap of all KV pairs)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub version: u32,
    pub n_tensors: u64,
    pub kv: HashMap<String, GgufValue>,
}

// ---------------------------------------------------------------------------
// High-level model metadata extracted from the raw KV pairs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct ModelMetadata {
    pub name: Option<String>,
    pub size_label: Option<String>,
    pub architecture: Option<String>,
    pub context_length: Option<u32>,
    pub n_layers: Option<u32>,
    pub n_head_kv: Option<u32>,
    pub n_head: Option<u32>,
    pub embedding_length: Option<u32>,
    pub recommended_top_k: Option<i32>,
    pub recommended_top_p: Option<f32>,
    pub recommended_temp: Option<f32>,
    pub recommended_min_p: Option<f32>,
    pub file_type: Option<u32>,
}

impl ModelMetadata {
    /// Read and extract model metadata from a GGUF file.
    pub fn from_file(path: &str) -> Option<Self> {
        let raw = read_gguf_header(Path::new(path)).ok()?;
        Some(Self::from_raw(&raw))
    }

    /// Build from already-parsed raw metadata.
    pub fn from_raw(raw: &GgufMetadata) -> Self {
        let kv = &raw.kv;

        // Detect architecture prefix (e.g. "llama", "qwen2", "gemma")
        let arch = kv
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let arch_prefix = arch.as_deref().unwrap_or("").to_string();

        // Helper: look up "<arch>.suffix" key
        let arch_u32 = |suffix: &str| -> Option<u32> {
            let key = format!("{arch_prefix}.{suffix}");
            kv.get(&key).and_then(|v| v.as_u32())
        };

        Self {
            name: kv.get("general.name").and_then(|v| v.as_str()).map(String::from),
            size_label: kv.get("general.size_label").and_then(|v| v.as_str()).map(String::from),
            architecture: arch,
            context_length: arch_u32("context_length"),
            n_layers: arch_u32("block_count"),
            n_head_kv: arch_u32("attention.head_count_kv"),
            n_head: arch_u32("attention.head_count"),
            embedding_length: arch_u32("embedding_length"),
            recommended_top_k: kv.get("general.sampling.top_k").and_then(|v| v.as_i32()),
            recommended_top_p: kv.get("general.sampling.top_p").and_then(|v| v.as_f32()),
            recommended_temp: kv.get("general.sampling.temp").and_then(|v| v.as_f32()),
            recommended_min_p: kv.get("general.sampling.min_p").and_then(|v| v.as_f32()),
            file_type: kv.get("general.file_type").and_then(|v| v.as_u32()),
        }
    }

    /// Head dimension derived from embedding_length / n_head (typically 128).
    pub fn head_dim(&self) -> Option<u32> {
        match (self.embedding_length, self.n_head) {
            (Some(emb), Some(h)) if h > 0 => Some(emb / h),
            _ => None,
        }
    }

    /// Bytes per token per layer for KV cache:
    /// `n_kv_heads * head_dim * 2 (K+V)`
    ///
    /// Returns `None` when the required architecture fields are missing.
    pub fn kv_bytes_per_token_per_layer(&self) -> Option<u64> {
        let kv_heads = self.n_head_kv? as u64;
        let hdim = self.head_dim()? as u64;
        Some(kv_heads * hdim * 2)
    }
}

// ---------------------------------------------------------------------------
// Binary reader helpers (little-endian, no external crate)
// ---------------------------------------------------------------------------

fn read_u8(r: &mut impl Read) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i8(r: &mut impl Read) -> io::Result<i8> {
    Ok(read_u8(r)? as i8)
}

fn read_u16(r: &mut impl Read) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16(r: &mut impl Read) -> io::Result<i16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(r: &mut impl Read) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64(r: &mut impl Read) -> io::Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32(r: &mut impl Read) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64(r: &mut impl Read) -> io::Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_string(r: &mut impl Read) -> io::Result<String> {
    let len = read_u64(r)? as usize;
    // Sanity cap: reject strings > 1 MB
    if len > 1_000_000 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "GGUF string too long"));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn read_bool(r: &mut impl Read) -> io::Result<bool> {
    Ok(read_u8(r)? != 0)
}

// ---------------------------------------------------------------------------
// GGUF value type IDs
// ---------------------------------------------------------------------------

const TYPE_UINT8: u32   = 0;
const TYPE_INT8: u32    = 1;
const TYPE_UINT16: u32  = 2;
const TYPE_INT16: u32   = 3;
const TYPE_UINT32: u32  = 4;
const TYPE_INT32: u32   = 5;
const TYPE_FLOAT32: u32 = 6;
const TYPE_BOOL: u32    = 7;
const TYPE_STRING: u32  = 8;
const TYPE_ARRAY: u32   = 9;
const TYPE_UINT64: u32  = 10;
const TYPE_INT64: u32   = 11;
const TYPE_FLOAT64: u32 = 12;

/// Size in bytes of a single element of the given scalar type.
fn scalar_size(type_id: u32) -> Option<u64> {
    match type_id {
        TYPE_UINT8 | TYPE_INT8 | TYPE_BOOL => Some(1),
        TYPE_UINT16 | TYPE_INT16 => Some(2),
        TYPE_UINT32 | TYPE_INT32 | TYPE_FLOAT32 => Some(4),
        TYPE_UINT64 | TYPE_INT64 | TYPE_FLOAT64 => Some(8),
        _ => None,
    }
}

/// Read a single GGUF value.  Arrays are skipped (seeked past) because the
/// only arrays in practice are huge token lists we don't need.
fn read_value<R: Read + Seek>(r: &mut R) -> io::Result<GgufValue> {
    let vtype = read_u32(r)?;
    match vtype {
        TYPE_UINT8   => Ok(GgufValue::Uint8(read_u8(r)?)),
        TYPE_INT8    => Ok(GgufValue::Int8(read_i8(r)?)),
        TYPE_UINT16  => Ok(GgufValue::Uint16(read_u16(r)?)),
        TYPE_INT16   => Ok(GgufValue::Int16(read_i16(r)?)),
        TYPE_UINT32  => Ok(GgufValue::Uint32(read_u32(r)?)),
        TYPE_INT32   => Ok(GgufValue::Int32(read_i32(r)?)),
        TYPE_FLOAT32 => Ok(GgufValue::Float32(read_f32(r)?)),
        TYPE_BOOL    => Ok(GgufValue::Bool(read_bool(r)?)),
        TYPE_STRING  => Ok(GgufValue::String(read_string(r)?)),
        TYPE_UINT64  => Ok(GgufValue::Uint64(read_u64(r)?)),
        TYPE_INT64   => Ok(GgufValue::Int64(read_i64(r)?)),
        TYPE_FLOAT64 => Ok(GgufValue::Float64(read_f64(r)?)),
        TYPE_ARRAY   => {
            let elem_type = read_u32(r)?;
            let count = read_u64(r)?;
            // Skip over array elements
            if elem_type == TYPE_STRING {
                // String arrays: must read each element to know its length
                for _ in 0..count {
                    let len = read_u64(r)? as i64;
                    r.seek(SeekFrom::Current(len))?;
                }
            } else if let Some(elem_size) = scalar_size(elem_type) {
                let total = elem_size.saturating_mul(count) as i64;
                r.seek(SeekFrom::Current(total))?;
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("GGUF: unsupported array element type {elem_type}"),
                ));
            }
            Ok(GgufValue::ArraySkipped { element_type: elem_type, count })
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("GGUF: unknown value type {vtype}"),
        )),
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Read only the KV metadata header of a GGUF file.
///
/// The function opens the file, validates the magic bytes and version, reads
/// all key-value pairs into a `HashMap`, and returns immediately without
/// touching any tensor data.
pub fn read_gguf_header(path: &Path) -> io::Result<GgufMetadata> {
    let file = File::open(path)?;
    let mut r = BufReader::new(file);

    // -- Magic: "GGUF" (4 bytes) --
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Not a GGUF file (magic: {:?})", magic),
        ));
    }

    // -- Version (u32 LE) --
    let version = read_u32(&mut r)?;
    if version < 2 || version > 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported GGUF version {version} (expected 2 or 3)"),
        ));
    }

    // -- Tensor count and KV count --
    let n_tensors = read_u64(&mut r)?;
    let n_kv = read_u64(&mut r)?;

    // Sanity: metadata shouldn't have millions of keys
    if n_kv > 100_000 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("GGUF: suspiciously large n_kv = {n_kv}"),
        ));
    }

    // -- Read KV pairs --
    let mut kv = HashMap::with_capacity(n_kv as usize);
    for _ in 0..n_kv {
        let key = read_string(&mut r)?;
        let value = read_value(&mut r)?;
        kv.insert(key, value);
    }

    Ok(GgufMetadata { version, n_tensors, kv })
}
