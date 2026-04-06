use std::path::Path;
use crate::config::LauncherConfig;

const RESERVE_FRACTION: f64 = 0.10; // Reserve 10% of each resource

#[derive(Debug, Clone)]
pub struct SystemResources {
    pub gpu_vram_total_mb: u64,
    pub gpu_vram_free_mb: u64,
    pub gpu_name: String,
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
            gpu_vram_total_mb: gpu.0,
            gpu_vram_free_mb: gpu.1,
            gpu_name: gpu.2,
            ram_total_mb: ram_total,
            ram_free_mb: ram_free,
            cpu_cores: cores,
            cpu_threads: threads,
        }
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

    /// Auto-tune config based on detected resources and model size
    pub fn auto_tune(&self, config: &mut LauncherConfig, model_path: &str) {
        let model_size_mb = if !model_path.is_empty() {
            Path::new(model_path)
                .metadata()
                .map(|m| m.len() / (1024 * 1024))
                .unwrap_or(0)
        } else {
            0
        };

        let vram = self.usable_vram_mb();
        let ram = self.usable_ram_mb();
        let threads = self.usable_threads();

        // Estimate VRAM budget after model loading
        // Model VRAM ≈ file size * 1.1 (overhead for buffers)
        let model_vram = (model_size_mb as f64 * 1.1) as u64;
        let remaining_vram = vram.saturating_sub(model_vram);

        // -- Context size --
        // KV cache formula: n_kv_heads * head_dim * 2(K+V) * n_layers * ctx * bytes_per_elem / 1MB
        // For Qwen3-8B: 8 kv_heads, 128 head_dim, 36 layers
        // bytes_per_elem by cache type: f16=2.0, q8_0=1.0, turbo2=0.25, turbo3=0.375, turbo4=0.5
        let bytes_per_elem = match config.cache_type_k.as_str() {
            "f16" => 2.0_f64,
            "q8_0" => 1.0,
            "turbo2" => 0.25,
            "turbo3" => 0.375,
            "turbo4" => 0.5,
            _ => 2.0,
        };
        // Generic 8B model: 8 kv_heads * 128 head_dim * 2 * 36 layers = 589824 per token
        // Per 1K context in MB: 589824 * 1024 * bytes_per_elem / (1024*1024)
        let kv_per_1k_ctx = (589824.0 * 1024.0 * bytes_per_elem / (1024.0 * 1024.0)) as u64;
        let max_ctx_by_vram = if kv_per_1k_ctx > 0 {
            // Leave 512 MB for compute buffers
            let kv_budget = remaining_vram.saturating_sub(512);
            (kv_budget / kv_per_1k_ctx) * 1024
        } else {
            32768
        };
        // Cap at model's native context
        let ctx = std::cmp::min(max_ctx_by_vram, 65536);
        // Round down to nearest 4096
        let ctx = (ctx / 4096) * 4096;
        let ctx = std::cmp::max(ctx, 4096); // minimum 4K
        config.context_size = ctx.to_string();

        // -- Batch size --
        // Higher batch = faster prompt processing, more VRAM
        if remaining_vram > 8000 {
            config.batch_size = "2048".into();
            config.ubatch_size = "512".into();
        } else if remaining_vram > 4000 {
            config.batch_size = "1024".into();
            config.ubatch_size = "256".into();
        } else {
            config.batch_size = "512".into();
            config.ubatch_size = "128".into();
        }

        // -- Parallel slots --
        // Each slot reserves context_size worth of KV cache
        // With turbo3, can fit more slots
        let ctx_val = ctx;
        let kv_per_slot = (ctx_val / 1024) * kv_per_1k_ctx;
        let max_slots = if kv_per_slot > 0 {
            let kv_budget = remaining_vram.saturating_sub(512);
            std::cmp::min(kv_budget / kv_per_slot, 8)
        } else {
            1
        };
        config.parallel = std::cmp::max(max_slots, 1).to_string();

        // -- CPU threads --
        config.threads = threads.to_string();
        // HTTP threads: use a fraction of CPU threads
        let http_threads = std::cmp::max(threads / 4, 2);
        config.threads_http = http_threads.to_string();

        // -- GPU layers --
        config.gpu_layers = "auto".into();
    }

    /// Generate a human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "GPU: {} ({} MB total, {} MB free, {} MB usable)\n\
             RAM: {} MB total, {} MB free, {} MB usable\n\
             CPU: {} cores / {} threads ({} usable)",
            self.gpu_name,
            self.gpu_vram_total_mb,
            self.gpu_vram_free_mb,
            self.usable_vram_mb(),
            self.ram_total_mb,
            self.ram_free_mb,
            self.usable_ram_mb(),
            self.cpu_cores,
            self.cpu_threads,
            self.usable_threads(),
        )
    }
}

fn detect_gpu() -> (u64, u64, String) {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total,memory.free,name", "--format=csv,noheader,nounits"])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            let line = text.lines().next().unwrap_or("");
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() >= 3 {
                let total = parts[0].parse().unwrap_or(0);
                let free = parts[1].parse().unwrap_or(0);
                let name = parts[2].to_string();
                return (total, free, name);
            }
            (0, 0, "Unknown GPU".into())
        }
        _ => (0, 0, "No NVIDIA GPU detected".into()),
    }
}

fn detect_ram() -> (u64, u64) {
    // Use wmic for quick RAM detection on Windows
    let total = run_wmic_query("ComputerSystem", "TotalPhysicalMemory")
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|b| b / (1024 * 1024))
        .unwrap_or(0);

    let free = run_wmic_query("OS", "FreePhysicalMemory")
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|kb| kb / 1024) // wmic returns KB for this
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

    // On Windows, NUMBER_OF_PROCESSORS gives logical processors (threads)
    let threads = std::env::var("NUMBER_OF_PROCESSORS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(cores);

    // Rough core estimate: threads / 2 for SMT
    let physical_cores = if threads > cores { cores } else { threads / 2 };
    let physical_cores = std::cmp::max(physical_cores, 1);

    (physical_cores, threads)
}
