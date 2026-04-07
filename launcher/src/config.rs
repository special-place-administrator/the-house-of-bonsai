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

        // Backend "cpu" forces gpu_layers to 0 regardless of config
        let effective_gpu_layers: String;
        if self.backend == "cpu" {
            effective_gpu_layers = "0".into();
        } else {
            effective_gpu_layers = self.gpu_layers.clone();
        }

        let pairs: &[(&str, &str)] = &[
            ("-c",  &self.context_size),
            ("-ngl", &effective_gpu_layers),
            ("-fa", &self.flash_attention),
            ("-ctk", &self.cache_type_k),
            ("-ctv", &self.cache_type_v),
            ("-np", &self.parallel),
            ("-b",  &self.batch_size),
            ("-ub", &self.ubatch_size),
            ("-t",  &config.threads),
            ("--threads-http", &config.threads_http),
            ("-lv", &config.log_verbosity),
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
fn default_cache_type() -> String { "turbo3".into() }
fn default_layer_adaptive() -> String { "1".into() }
fn default_parallel() -> String { "1".into() }
fn default_batch_size() -> String { "2048".into() }
fn default_ubatch_size() -> String { "512".into() }
fn default_backend() -> String { "auto".into() }
fn default_log_verbosity() -> String { "3".into() }

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
            recent_models: Vec::new(),
            model_root: String::new(),
        }
    }
}

impl LauncherConfig {
    pub fn config_dir() -> PathBuf {
        let local_app_data = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."));
        local_app_data.join("LlamaTurboQuantLauncher")
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

    pub fn load() -> Self {
        let path = Self::config_path();
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
                    alias: if legacy.alias.is_empty() { "TurboQuant Local".into() } else { legacy.alias },
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

    /// Re-read GGUF metadata for each slot and apply capability-based settings.
    pub fn redetect_capabilities(&mut self) {
        for slot in &mut self.slots {
            if slot.model_path.is_empty() { continue; }
            if let Some(meta) = crate::gguf::ModelMetadata::from_file(&slot.model_path) {
                if meta.capabilities.embedding {
                    slot.embedding_mode = true;
                    slot.cache_type_k = "f16".into();
                    slot.cache_type_v = "f16".into();
                    slot.flash_attention = "off".into();
                    slot.turbo_layer_adaptive = "off".into();
                    if slot.parallel == "1" { slot.parallel = "4".into(); }
                } else {
                    slot.embedding_mode = false;
                }
                if let Some(ref name) = meta.name {
                    if slot.alias.starts_with("Model (port") {
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
