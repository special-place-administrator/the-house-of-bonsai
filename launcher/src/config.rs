use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Per-model-slot settings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelSlot {
    #[serde(default)]
    pub model_path: String,
    #[serde(default)]
    pub alias: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub embedding_mode: bool,
    #[serde(default = "default_context_size")]
    pub context_size: String,
    #[serde(default = "default_gpu_layers")]
    pub gpu_layers: String,
    #[serde(default = "default_flash_attention")]
    pub flash_attention: String,
    #[serde(default = "default_cache_type")]
    pub cache_type_k: String,
    #[serde(default = "default_cache_type")]
    pub cache_type_v: String,
    #[serde(default = "default_layer_adaptive")]
    pub turbo_layer_adaptive: String,
    #[serde(default = "default_parallel")]
    pub parallel: String,
    #[serde(default = "default_batch_size")]
    pub batch_size: String,
    #[serde(default = "default_ubatch_size")]
    pub ubatch_size: String,
    #[serde(default = "default_backend")]
    pub backend: String, // "auto", "cuda", "vulkan", "cpu"

    // -- sampling params --
    #[serde(default = "default_temp")]
    pub temp: String,
    #[serde(default = "default_top_p")]
    pub top_p: String,
    #[serde(default = "default_top_k")]
    pub top_k: String,
    #[serde(default = "default_min_p")]
    pub min_p: String,
    #[serde(default = "default_reasoning_budget")]
    pub reasoning_budget: String,
}

impl ModelSlot {
    /// Create a new default slot on the given port.
    pub fn new(port: u16) -> Self {
        Self {
            model_path: String::new(),
            alias: format!("Model (port {})", port),
            port,
            embedding_mode: false,
            context_size: default_context_size(),
            gpu_layers: default_gpu_layers(),
            flash_attention: default_flash_attention(),
            cache_type_k: default_cache_type(),
            cache_type_v: default_cache_type(),
            turbo_layer_adaptive: default_layer_adaptive(),
            parallel: default_parallel(),
            batch_size: default_batch_size(),
            ubatch_size: default_ubatch_size(),
            backend: default_backend(),
            temp: default_temp(),
            top_p: default_top_p(),
            top_k: default_top_k(),
            min_p: default_min_p(),
            reasoning_budget: default_reasoning_budget(),
        }
    }

    /// Build the argument list for llama-server for this slot.
    ///
    /// Per-slot args come from `self`; shared / global args come from `config`.
    pub fn build_server_args(&self, config: &LauncherConfig) -> Vec<String> {
        let mut args = Vec::new();

        args.extend(["--host".into(), config.host.clone()]);
        args.extend(["--port".into(), self.port.to_string()]);

        if !self.model_path.is_empty() {
            args.extend(["-m".into(), self.model_path.clone()]);
        }
        if !self.alias.is_empty() {
            args.extend(["-a".into(), self.alias.clone()]);
        }

        if self.embedding_mode {
            args.push("--embedding".into());
        }

        // Backend-specific adjustments
        let effective_gpu_layers: String;
        let effective_cache_k: String;
        let effective_cache_v: String;
        let mut extra_flags: Vec<String> = Vec::new();

        let safe_cache = |ct: &str| -> String {
            if ct.starts_with("turbo") { "f16".into() } else { ct.to_string() }
        };

        match self.backend.as_str() {
            "cpu" => {
                effective_gpu_layers = "0".into();
                effective_cache_k = self.cache_type_k.clone(); // turbo3 works on CPU
                effective_cache_v = self.cache_type_v.clone();
                extra_flags.push("--no-warmup".into()); // CPU warmup is very slow
            }
            "vulkan" => {
                effective_gpu_layers = self.gpu_layers.clone();
                // Vulkan doesn't support turbo SET_ROWS op — fall back to f16
                effective_cache_k = safe_cache(&self.cache_type_k);
                effective_cache_v = safe_cache(&self.cache_type_v);
            }
            _ => {
                // "auto" or "cuda" — all cache types supported
                effective_gpu_layers = self.gpu_layers.clone();
                effective_cache_k = self.cache_type_k.clone();
                effective_cache_v = self.cache_type_v.clone();
            }
        }

        let pairs: &[(&str, &str)] = &[
            ("-c",  &self.context_size),
            ("-ngl", &effective_gpu_layers),
            ("-fa", &self.flash_attention),
            ("-ctk", &effective_cache_k),
            ("-ctv", &effective_cache_v),
            ("-np", &self.parallel),
            ("-b",  &self.batch_size),
            ("-ub", &self.ubatch_size),
            ("-t",  &config.threads),
            ("--threads-http", &config.threads_http),
            ("-lv", &config.log_verbosity),
            ("--temp", &self.temp),
            ("--top-p", &self.top_p),
            ("--top-k", &self.top_k),
            ("--min-p", &self.min_p),
            ("--reasoning-budget", &self.reasoning_budget),
        ];

        for (flag, value) in pairs {
            if !value.is_empty() {
                args.extend([flag.to_string(), value.to_string()]);
            }
        }

        if !config.api_key.is_empty() {
            args.extend(["--api-key".into(), config.api_key.clone()]);
        }

        if config.no_mmap {
            args.push("--no-mmap".into());
        } else {
            args.push("--mmap".into());
        }

        if config.disable_web_ui {
            args.push("--no-webui".into());
        } else {
            args.push("--webui".into());
        }

        args.extend(extra_flags);

        args
    }

    /// Build the full command preview string for this slot.
    pub fn command_preview(&self, server_exe: &Path, config: &LauncherConfig) -> String {
        let exe = server_exe.display().to_string();
        let mut parts = vec![format!("\"{}\"", exe)];
        for arg in self.build_server_args(config) {
            if arg.contains(' ') || arg.contains('"') {
                parts.push(format!("\"{}\"", arg.replace('"', "\\\"")));
            } else {
                parts.push(arg);
            }
        }
        if !config.extra_args.is_empty() {
            parts.push(config.extra_args.trim().to_string());
        }
        parts.join(" ")
    }
}

// ---------------------------------------------------------------------------
// Global / shared launcher settings
// ---------------------------------------------------------------------------

/// Helper struct used only for deserializing legacy (pre-slots) config files.
/// When the JSON has flat `modelPath`, `port`, etc. we migrate them into a
/// single `ModelSlot`.
#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct LegacyFields {
    #[serde(default)]
    model_selection: String,
    #[serde(default)]
    model_path: String,
    #[serde(default = "default_port")]
    port: u16,
    #[serde(default)]
    alias: String,
    #[serde(default = "default_context_size")]
    context_size: String,
    #[serde(default = "default_gpu_layers")]
    gpu_layers: String,
    #[serde(default = "default_flash_attention")]
    flash_attention: String,
    #[serde(default = "default_cache_type")]
    cache_type_k: String,
    #[serde(default = "default_cache_type")]
    cache_type_v: String,
    #[serde(default = "default_layer_adaptive")]
    turbo_layer_adaptive: String,
    #[serde(default = "default_parallel")]
    parallel: String,
    #[serde(default = "default_batch_size")]
    batch_size: String,
    #[serde(default = "default_ubatch_size")]
    ubatch_size: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LauncherConfig {
    #[serde(default = "default_slots")]
    pub slots: Vec<ModelSlot>,

    // -- shared across all slots --
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default)]
    pub threads: String,
    #[serde(default)]
    pub threads_http: String,
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub no_mmap: bool,
    #[serde(default)]
    pub disable_web_ui: bool,
    #[serde(default = "default_log_verbosity")]
    pub log_verbosity: String,
    #[serde(default)]
    pub extra_args: String,

    // -- management API --
    #[serde(default = "default_api_port")]
    pub api_port: u16,
    #[serde(default)]
    pub api_secret: String,

    // -- bookkeeping --
    #[serde(default)]
    pub recent_models: Vec<String>,
    #[serde(default)]
    pub model_root: String,
}

fn default_slots() -> Vec<ModelSlot> {
    vec![ModelSlot::new(default_port())]
}

fn default_host() -> String { "127.0.0.1".into() }
fn default_port() -> u16 { 8080 }
fn default_context_size() -> String { "32768".into() }
fn default_gpu_layers() -> String { "auto".into() }
fn default_flash_attention() -> String { "on".into() }
fn default_cache_type() -> String { "f16".into() }
fn default_layer_adaptive() -> String { "1".into() }
fn default_parallel() -> String { "1".into() }
fn default_batch_size() -> String { "2048".into() }
fn default_ubatch_size() -> String { "512".into() }
fn default_backend() -> String { "auto".into() }
fn default_log_verbosity() -> String { "3".into() }

fn default_temp() -> String { "0.5".into() }
fn default_top_p() -> String { "0.85".into() }
fn default_top_k() -> String { "20".into() }
fn default_min_p() -> String { "0".into() }
fn default_reasoning_budget() -> String { "0".into() }

fn default_api_port() -> u16 { 9876 }

impl Default for LauncherConfig {
    fn default() -> Self {
        Self {
            slots: default_slots(),
            host: default_host(),
            threads: String::new(),
            threads_http: String::new(),
            api_key: String::new(),
            no_mmap: false,
            disable_web_ui: false,
            log_verbosity: default_log_verbosity(),
            extra_args: String::new(),
            api_port: default_api_port(),
            api_secret: String::new(),
            recent_models: Vec::new(),
            model_root: String::new(),
        }
    }
}

impl LauncherConfig {
    pub fn config_dir() -> PathBuf {
        let local_app_data = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."));
        local_app_data.join("BonsaiLauncher")
    }

    pub fn config_path() -> PathBuf {
        Self::config_dir().join("launcher-config.json")
    }

    pub fn runtime_state_path() -> PathBuf {
        Self::config_dir().join("runtime-state.json")
    }

    pub fn log_dir() -> PathBuf {
        Self::config_dir().join("logs")
    }

    /// Legacy config directory (pre-rename).
    fn legacy_config_path() -> PathBuf {
        let local_app_data = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."));
        local_app_data.join("LlamaTurboQuantLauncher").join("launcher-config.json")
    }

    pub fn load() -> Self {
        let path = Self::config_path();

        // Migrate from legacy config dir if new one doesn't exist yet
        if !path.exists() {
            let legacy = Self::legacy_config_path();
            if legacy.exists() {
                let _ = std::fs::create_dir_all(Self::config_dir());
                let _ = std::fs::copy(&legacy, &path);
            }
        }

        if !path.exists() {
            return Self::default();
        }
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => return Self::default(),
        };

        // Try normal deserialization first.
        let mut cfg: LauncherConfig = match serde_json::from_str(&content) {
            Ok(c) => c,
            Err(_) => Self::default(),
        };

        // Backward compat: if slots is empty (legacy config with flat fields),
        // migrate into a single slot.
        if cfg.slots.is_empty() {
            if let Ok(legacy) = serde_json::from_str::<LegacyFields>(&content) {
                let slot = ModelSlot {
                    model_path: legacy.model_path,
                    alias: if legacy.alias.is_empty() { "Bonsai Local".into() } else { legacy.alias },
                    port: legacy.port,
                    embedding_mode: false,
                    context_size: legacy.context_size,
                    gpu_layers: legacy.gpu_layers,
                    flash_attention: legacy.flash_attention,
                    cache_type_k: legacy.cache_type_k,
                    cache_type_v: legacy.cache_type_v,
                    turbo_layer_adaptive: legacy.turbo_layer_adaptive,
                    parallel: legacy.parallel,
                    batch_size: legacy.batch_size,
                    ubatch_size: legacy.ubatch_size,
                    backend: default_backend(),
                    temp: default_temp(),
                    top_p: default_top_p(),
                    top_k: default_top_k(),
                    min_p: default_min_p(),
                    reasoning_budget: default_reasoning_budget(),
                };
                cfg.slots = vec![slot];
            } else {
                cfg.slots = default_slots();
            }
        }

        // Re-detect capabilities on every load — ensures new detection
        // logic applies to previously saved configs.
        cfg.redetect_capabilities();

        cfg
    }

    /// Re-read GGUF metadata for each slot.
    /// - Always syncs: embedding_mode, sampling params (from GGUF author recommendations)
    /// - Fresh slots only: alias from model name
    /// Infrastructure params (context, cache, FA, parallel) are set by auto_tune(),
    /// not here — auto_tune uses VRAM + model architecture to compute them.
    pub fn redetect_capabilities(&mut self) {
        for slot in &mut self.slots {
            if slot.model_path.is_empty() { continue; }
            if let Some(meta) = crate::gguf::ModelMetadata::from_file(&slot.model_path) {
                // Always sync embedding_mode from GGUF metadata
                slot.embedding_mode = meta.capabilities.embedding;

                // Apply GGUF-embedded sampling params when present (author's recommendations)
                if let Some(t) = meta.recommended_temp { slot.temp = format!("{t}"); }
                if let Some(p) = meta.recommended_top_p { slot.top_p = format!("{p}"); }
                if let Some(k) = meta.recommended_top_k { slot.top_k = format!("{k}"); }
                if let Some(m) = meta.recommended_min_p { slot.min_p = format!("{m}"); }

                // Auto-name fresh slots from GGUF model name
                let is_fresh = slot.alias.starts_with("Model (port")
                    || slot.alias == "Bonsai Local";
                if is_fresh {
                    if let Some(ref name) = meta.name {
                        slot.alias = name.clone();
                    }
                }
            }
        }
    }

    pub fn save(&self) -> Result<(), String> {
        let dir = Self::config_dir();
        std::fs::create_dir_all(&dir).map_err(|e| format!("Cannot create config dir: {e}"))?;
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Serialize error: {e}"))?;
        std::fs::write(Self::config_path(), json)
            .map_err(|e| format!("Write error: {e}"))?;
        Ok(())
    }

    pub fn add_recent_model(&mut self, path: &str) {
        let normalized = path.trim().to_string();
        if normalized.is_empty() { return; }

        self.recent_models.retain(|p| !p.eq_ignore_ascii_case(&normalized));
        self.recent_models.insert(0, normalized);
        self.recent_models.truncate(12);
    }

    /// Add a new model slot with the next available port.
    pub fn add_slot(&mut self) {
        let max_port = self.slots.iter().map(|s| s.port).max().unwrap_or(default_port());
        self.slots.push(ModelSlot::new(max_port + 1));
    }

    /// Remove a slot by index. Keeps at least 1 slot.
    pub fn remove_slot(&mut self, index: usize) {
        if self.slots.len() > 1 && index < self.slots.len() {
            self.slots.remove(index);
        }
    }
}
