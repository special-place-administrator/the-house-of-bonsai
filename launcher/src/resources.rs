use std::path::Path;
use crate::config::ModelSlot;
use crate::gguf::ModelMetadata;

const RESERVE_FRACTION: f64 = 0.10; // Reserve 10% of each resource

// ---------------------------------------------------------------------------
// GPU vendor / backend detection
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Unknown,
    None, // CPU only / no GPU
}

impl GpuVendor {
        /// Return the compute backends available for this GPU vendor.
        /// Also checks if the required runtime DLLs are actually present on the system.
        pub fn available_backends(&self) -> Vec<&'static str> {
            let mut backends = vec!["cpu"]; // CPU always available

            // Check Vulkan runtime (vulkan-1.dll in System32 — comes with GPU drivers)
            let has_vulkan_runtime = std::path::Path::new(r"C:\Windows\System32\vulkan-1.dll").exists();

            // Check CUDA runtime (nvcuda.dll in System32 — comes with NVIDIA drivers)
            let has_cuda_runtime = std::path::Path::new(r"C:\Windows\System32\nvcuda.dll").exists();

            match self {
                GpuVendor::Nvidia => {
                    if has_vulkan_runtime { backends.push("vulkan"); }
                    if has_cuda_runtime { backends.push("cuda"); }
                }
                GpuVendor::Amd | GpuVendor::Intel | GpuVendor::Unknown => {
                    if has_vulkan_runtime { backends.push("vulkan"); }
                }
                GpuVendor::None => {}
            }

            if backends.len() > 1 {
                backends.insert(0, "auto");
            }
            backends
        }

    /// Human-readable label for a backend value.
    pub fn backend_label(backend: &str) -> &'static str {
        match backend {
            "auto"   => "Auto (detect)",
            "cuda"   => "CUDA (NVIDIA)",
            "vulkan" => "Vulkan (AMD/Intel/NVIDIA)",
            "cpu"    => "CPU Only",
            _        => "Unknown",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub vendor: GpuVendor,
    pub name: String,
    pub vram_total_mb: u64,
    pub vram_free_mb: u64,
}

#[derive(Debug, Clone)]
pub struct SystemResources {
    pub gpu_vram_total_mb: u64,
    pub gpu_vram_free_mb: u64,
    pub gpu_name: String,
    pub gpu_vendor: GpuVendor,
    pub ram_total_mb: u64,
    pub ram_free_mb: u64,
    pub cpu_cores: u32,
    pub cpu_threads: u32,
}

impl SystemResources {
    pub fn detect() -> Self {
        let gpu = detect_gpu();
        let (ram_total, ram_free) = detect_ram();
        let (cores, threads) = detect_cpu();

        Self {
            gpu_vram_total_mb: gpu.vram_total_mb,
            gpu_vram_free_mb: gpu.vram_free_mb,
            gpu_name: gpu.name,
            gpu_vendor: gpu.vendor,
            ram_total_mb: ram_total,
            ram_free_mb: ram_free,
            cpu_cores: cores,
            cpu_threads: threads,
        }
    }

    /// Available compute backends based on detected GPU hardware.
    pub fn available_backends(&self) -> Vec<&'static str> {
        self.gpu_vendor.available_backends()
    }

    /// Available VRAM after 10% reserve
    pub fn usable_vram_mb(&self) -> u64 {
        let reserved = (self.gpu_vram_free_mb as f64 * RESERVE_FRACTION) as u64;
        self.gpu_vram_free_mb.saturating_sub(reserved)
    }

    /// Available RAM after 10% reserve
    pub fn usable_ram_mb(&self) -> u64 {
        let reserved = (self.ram_free_mb as f64 * RESERVE_FRACTION) as u64;
        self.ram_free_mb.saturating_sub(reserved)
    }

    /// CPU threads to use (reserve ~10%)
    pub fn usable_threads(&self) -> u32 {
        let reserved = std::cmp::max(2, (self.cpu_threads as f64 * RESERVE_FRACTION) as u32);
        self.cpu_threads.saturating_sub(reserved)
    }

    /// Auto-tune a single model slot based on detected resources and model size.
    ///
    /// When `meta` is provided the actual model architecture (KV heads,
    /// head dim, layer count, native context length) is used for accurate
    /// KV-cache sizing.  Without it, a generic 8B-class default is assumed.
    ///
    /// Returns the recommended CPU threads so the caller can apply them to
    /// the shared config.
    pub fn auto_tune(
        &self,
        slot: &mut ModelSlot,
        model_path: &str,
        meta: Option<&ModelMetadata>,
    ) -> (u32, u32) {
        let model_size_mb = if !model_path.is_empty() {
            Path::new(model_path)
                .metadata()
                .map(|m| m.len() / (1024 * 1024))
                .unwrap_or(0)
        } else {
            0
        };

        let vram = self.usable_vram_mb();
        let threads = self.usable_threads();

        // Estimate VRAM budget after model loading
        let model_vram = (model_size_mb as f64 * 1.1) as u64;
        let remaining_vram = vram.saturating_sub(model_vram);

        // -- Context size --
        let bytes_per_elem = match slot.cache_type_k.as_str() {
            "f16" => 2.0_f64,
            "q8_0" => 1.0,
            "turbo2" => 0.25,
            "turbo3" => 0.375,
            "turbo4" => 0.5,
            _ => 2.0,
        };

        let (kv_bytes_per_token, native_ctx) = if let Some(m) = meta {
            let bpt = m.kv_bytes_per_token_per_layer().unwrap_or(8 * 128 * 2);
            let layers = m.n_layers.unwrap_or(36) as u64;
            let ctx_cap = m.context_length.unwrap_or(65536) as u64;
            (bpt * layers, ctx_cap)
        } else {
            (589_824_u64, 65_536_u64)
        };

        // Per 1K context in MB
        let kv_per_1k_ctx = (kv_bytes_per_token as f64 * 1024.0 * bytes_per_elem / (1024.0 * 1024.0)) as u64;
        let max_ctx_by_vram = if kv_per_1k_ctx > 0 {
            let kv_budget = remaining_vram.saturating_sub(512);
            (kv_budget / kv_per_1k_ctx) * 1024
        } else {
            32768
        };
        let ctx = std::cmp::min(max_ctx_by_vram, native_ctx);
        let ctx = (ctx / 4096) * 4096;
        let ctx = std::cmp::max(ctx, 4096);
        slot.context_size = ctx.to_string();

        // -- Batch size --
        if remaining_vram > 8000 {
            slot.batch_size = "2048".into();
            slot.ubatch_size = "512".into();
        } else if remaining_vram > 4000 {
            slot.batch_size = "1024".into();
            slot.ubatch_size = "256".into();
        } else {
            slot.batch_size = "512".into();
            slot.ubatch_size = "128".into();
        }

        // -- Flash attention --
        // On by default — nearly all modern architectures support it.
        // Only disable if model metadata explicitly indicates issues.
        slot.flash_attention = "on".into();

        // -- Cache type --
        // Derive from model size vs available VRAM:
        //   - If model leaves plenty of VRAM headroom → turbo3 (best quality/compression)
        //   - If tight on VRAM → turbo2 (more compression)
        //   - Embedding models with small embedding dim → turbo3 is fine
        if remaining_vram > model_vram * 2 {
            // Plenty of room — turbo3 gives good quality with 3-bit KV
            slot.cache_type_k = "turbo3".into();
            slot.cache_type_v = "turbo3".into();
        } else if remaining_vram > model_vram {
            // Moderate room — turbo3 still fine
            slot.cache_type_k = "turbo3".into();
            slot.cache_type_v = "turbo3".into();
        } else {
            // Tight — use turbo2 for max compression
            slot.cache_type_k = "turbo2".into();
            slot.cache_type_v = "turbo2".into();
        }

        // -- Parallel slots --
        // Embedding models: lower parallel (2) since each request is cheap
        // Text models: compute from VRAM budget
        let is_embedding = meta.map(|m| m.capabilities.embedding).unwrap_or(false);
        let kv_per_slot = (ctx / 1024) * kv_per_1k_ctx;
        let max_slots = if is_embedding {
            2
        } else if kv_per_slot > 0 {
            let kv_budget = remaining_vram.saturating_sub(512);
            std::cmp::min(kv_budget / kv_per_slot, 8)
        } else {
            1
        };
        slot.parallel = std::cmp::max(max_slots, 1).to_string();

        // -- Layer adaptive --
        // Off for embedding models (no KV cache compression benefit)
        // On (1) for text models with turbo cache
        if is_embedding {
            slot.turbo_layer_adaptive = "off".into();
        } else {
            slot.turbo_layer_adaptive = "1".into();
        }

        // -- GPU layers --
        slot.gpu_layers = "auto".into();

        // Return recommended threads so the caller can set them on the shared config
        let http_threads = std::cmp::max(threads / 4, 2);
        (threads, http_threads)
    }

    /// Generate a human-readable summary
    pub fn summary(&self) -> String {
        let vendor_str = match &self.gpu_vendor {
            GpuVendor::Nvidia  => "NVIDIA",
            GpuVendor::Amd     => "AMD",
            GpuVendor::Intel   => "Intel",
            GpuVendor::Unknown => "Unknown",
            GpuVendor::None    => "None",
        };
        let backends = self.available_backends().join(", ");
        format!(
            "GPU: {} ({} MB total, {} MB free, {} MB usable)\n\
             Vendor: {} | Backends: [{}]\n\
             RAM: {} MB total, {} MB free, {} MB usable\n\
             CPU: {} cores / {} threads ({} usable)",
            self.gpu_name,
            self.gpu_vram_total_mb,
            self.gpu_vram_free_mb,
            self.usable_vram_mb(),
            vendor_str,
            backends,
            self.ram_total_mb,
            self.ram_free_mb,
            self.usable_ram_mb(),
            self.cpu_cores,
            self.cpu_threads,
            self.usable_threads(),
        )
    }
}

fn detect_gpu() -> GpuInfo {
    // Try NVIDIA first via nvidia-smi
    let nvidia = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total,memory.free,name", "--format=csv,noheader,nounits"])
        .output();

    if let Ok(out) = nvidia {
        if out.status.success() {
            let text = String::from_utf8_lossy(&out.stdout);
            let line = text.lines().next().unwrap_or("");
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() >= 3 {
                let total = parts[0].parse().unwrap_or(0);
                let free = parts[1].parse().unwrap_or(0);
                let name = parts[2].to_string();
                return GpuInfo {
                    vendor: GpuVendor::Nvidia,
                    name,
                    vram_total_mb: total,
                    vram_free_mb: free,
                };
            }
        }
    }

    // Fallback: detect GPU vendor via WMI (AMD / Intel / Unknown)
    let vendor = detect_gpu_vendor_wmi();
    let name = match &vendor {
        GpuVendor::Amd => "AMD GPU (VRAM unknown)".into(),
        GpuVendor::Intel => "Intel GPU (VRAM unknown)".into(),
        GpuVendor::Unknown => "Unknown GPU".into(),
        GpuVendor::None => "No GPU detected".into(),
        GpuVendor::Nvidia => unreachable!(), // handled above
    };

    GpuInfo {
        vendor,
        name,
        vram_total_mb: 0,
        vram_free_mb: 0,
    }
}

/// Detect GPU vendor from WMI VideoController names.
fn detect_gpu_vendor_wmi() -> GpuVendor {
    let output = std::process::Command::new("wmic")
        .args(["path", "win32_VideoController", "get", "Name"])
        .output();

    let text = match output {
        Ok(out) if out.status.success() => {
            String::from_utf8_lossy(&out.stdout).to_string()
        }
        _ => return GpuVendor::None,
    };

    let upper = text.to_uppercase();

    // Check for discrete GPU vendors (skip integrated if discrete found)
    if upper.contains("NVIDIA") || upper.contains("GEFORCE") || upper.contains("QUADRO")
        || upper.contains("RTX") || upper.contains("GTX")
    {
        return GpuVendor::Nvidia;
    }
    if upper.contains("RADEON") || upper.contains("AMD") {
        return GpuVendor::Amd;
    }
    if upper.contains("ARC") || upper.contains("BATTLEMAGE") {
        return GpuVendor::Intel;
    }
    // Intel integrated (UHD/Iris) is not useful for LLM offload — treat as None
    // unless it's the only GPU
    if upper.contains("INTEL") {
        // Could be Intel integrated only — still report as Intel so Vulkan is offered
        return GpuVendor::Intel;
    }

    GpuVendor::None
}

fn detect_ram() -> (u64, u64) {
    let total = run_wmic_query("ComputerSystem", "TotalPhysicalMemory")
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|b| b / (1024 * 1024))
        .unwrap_or(0);

    let free = run_wmic_query("OS", "FreePhysicalMemory")
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|kb| kb / 1024)
        .unwrap_or(0);

    (total, free)
}

fn run_wmic_query(class: &str, property: &str) -> Option<String> {
    let output = std::process::Command::new("wmic")
        .args([class, "get", property, "/value"])
        .output()
        .ok()?;

    if !output.status.success() { return None; }

    let text = String::from_utf8_lossy(&output.stdout);
    for line in text.lines() {
        if let Some(val) = line.strip_prefix(&format!("{}=", property)) {
            return Some(val.trim().to_string());
        }
    }
    None
}

fn detect_cpu() -> (u32, u32) {
    let cores = std::thread::available_parallelism()
        .map(|p| p.get() as u32)
        .unwrap_or(4);

    let threads = std::env::var("NUMBER_OF_PROCESSORS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(cores);

    let physical_cores = if threads > cores { cores } else { threads / 2 };
    let physical_cores = std::cmp::max(physical_cores, 1);

    (physical_cores, threads)
}
