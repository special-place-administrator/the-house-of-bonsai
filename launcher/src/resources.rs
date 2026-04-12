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

            let is_cpu = slot.backend == "cpu";
            let threads = self.usable_threads();
            let is_embedding = meta.map(|m| m.capabilities.embedding).unwrap_or(false);

            // Memory budget: use RAM for CPU, VRAM for GPU backends
            let budget_mb = if is_cpu {
                self.usable_ram_mb()
            } else {
                self.usable_vram_mb()
            };

            let model_mem = (model_size_mb as f64 * 1.1) as u64;
            let remaining = budget_mb.saturating_sub(model_mem);

            // -- Cache type --
            // CPU/Vulkan: RQ cache types require CUDA kernels, fall back to f16
            // CUDA: rq3 for quality, rq2 if tight on VRAM
            if is_cpu {
                slot.cache_type_k = "f16".into();
                slot.cache_type_v = "f16".into();
            } else if slot.backend == "vulkan" {
                slot.cache_type_k = "f16".into();
                slot.cache_type_v = "f16".into();
            } else if remaining > model_mem * 2 {
                slot.cache_type_k = "rq3".into();
                slot.cache_type_v = "rq3".into();
            } else if remaining > model_mem {
                slot.cache_type_k = "rq3".into();
                slot.cache_type_v = "rq3".into();
            } else {
                slot.cache_type_k = "rq2".into();
                slot.cache_type_v = "rq2".into();
            }

            // -- Context size --
            let bytes_per_elem = match slot.cache_type_k.as_str() {
                "f16" => 2.0_f64,
                "q8_0" => 1.0,
                "rq2" => 0.25,
                "rq3" => 0.375,
                "rq4" => 0.5,
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

            let kv_per_1k_ctx = (kv_bytes_per_token as f64 * 1024.0 * bytes_per_elem / (1024.0 * 1024.0)) as u64;
            let max_ctx_by_mem = if kv_per_1k_ctx > 0 {
                let kv_budget = remaining.saturating_sub(512);
                (kv_budget / kv_per_1k_ctx) * 1024
            } else {
                32768
            };

            // CPU: cap context more aggressively since inference is slow
            let ctx_cap = if is_cpu {
                std::cmp::min(native_ctx, 8192)
            } else {
                native_ctx
            };
            let ctx = std::cmp::min(max_ctx_by_mem, ctx_cap);
            let ctx = (ctx / 4096) * 4096;
            let ctx = std::cmp::max(ctx, 4096);
            slot.context_size = ctx.to_string();

            // -- Batch size --
            if is_cpu {
                // CPU: smaller batches to avoid memory pressure and long stalls
                slot.batch_size = "512".into();
                slot.ubatch_size = "128".into();
            } else if remaining > 8000 {
                slot.batch_size = "2048".into();
                slot.ubatch_size = "512".into();
            } else if remaining > 4000 {
                slot.batch_size = "1024".into();
                slot.ubatch_size = "256".into();
            } else {
                slot.batch_size = "512".into();
                slot.ubatch_size = "128".into();
            }

            // -- Flash attention --
            // CPU: not supported. GPU: on by default.
            if is_cpu {
                slot.flash_attention = "off".into();
            } else {
                slot.flash_attention = "on".into();
            }

            // -- Parallel slots --
            let kv_per_slot = (ctx / 1024) * kv_per_1k_ctx;
            let max_slots = if is_embedding {
                2
            } else if is_cpu {
                1 // CPU: single slot to avoid contention
            } else if kv_per_slot > 0 {
                let kv_budget = remaining.saturating_sub(512);
                std::cmp::min(kv_budget / kv_per_slot, 8)
            } else {
                1
            };
            slot.parallel = std::cmp::max(max_slots, 1).to_string();

            // -- Layer adaptive --
            // Off for CPU/embedding (no KV cache compression benefit)
            // On (1) for GPU text models with RQ cache
            if is_embedding || is_cpu {
                slot.rq_layer_adaptive = "off".into();
            } else {
                slot.rq_layer_adaptive = "1".into();
            }

            // -- GPU layers --
            if is_cpu {
                slot.gpu_layers = "0".into();
            } else {
                slot.gpu_layers = "auto".into();
            }

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
