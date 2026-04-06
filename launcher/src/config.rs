use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LauncherConfig {
    #[serde(default)]
    pub model_selection: String,
    #[serde(default)]
    pub model_path: String,
    #[serde(default)]
    pub recent_models: Vec<String>,
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub alias: String,
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
fn default_log_verbosity() -> String { "3".into() }

impl Default for LauncherConfig {
    fn default() -> Self {
        Self {
            model_selection: String::new(),
            model_path: String::new(),
            recent_models: Vec::new(),
            host: default_host(),
            port: default_port(),
            alias: "TurboQuant Local".into(),
            context_size: default_context_size(),
            gpu_layers: default_gpu_layers(),
            flash_attention: default_flash_attention(),
            cache_type_k: default_cache_type(),
            cache_type_v: default_cache_type(),
            turbo_layer_adaptive: default_layer_adaptive(),
            parallel: default_parallel(),
            batch_size: default_batch_size(),
            ubatch_size: default_ubatch_size(),
            threads: String::new(),
            threads_http: String::new(),
            api_key: String::new(),
            no_mmap: false,
            disable_web_ui: false,
            log_verbosity: default_log_verbosity(),
            extra_args: String::new(),
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
        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
                Err(_) => Self::default(),
            }
        } else {
            Self::default()
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

    /// Build the argument list for llama-server
    pub fn build_server_args(&self) -> Vec<String> {
        let mut args = Vec::new();

        args.extend(["--host".into(), self.host.clone()]);
        args.extend(["--port".into(), self.port.to_string()]);

        if !self.model_path.is_empty() {
            args.extend(["-m".into(), self.model_path.clone()]);
        }
        if !self.alias.is_empty() {
            args.extend(["-a".into(), self.alias.clone()]);
        }

        let pairs: &[(&str, &str)] = &[
            ("-c", &self.context_size),
            ("-ngl", &self.gpu_layers),
            ("-fa", &self.flash_attention),
            ("-ctk", &self.cache_type_k),
            ("-ctv", &self.cache_type_v),
            ("-np", &self.parallel),
            ("-b", &self.batch_size),
            ("-ub", &self.ubatch_size),
            ("-t", &self.threads),
            ("--threads-http", &self.threads_http),
            ("-lv", &self.log_verbosity),
        ];

        for (flag, value) in pairs {
            if !value.is_empty() {
                args.extend([flag.to_string(), value.to_string()]);
            }
        }

        if !self.api_key.is_empty() {
            args.extend(["--api-key".into(), self.api_key.clone()]);
        }

        if self.no_mmap {
            args.push("--no-mmap".into());
        } else {
            args.push("--mmap".into());
        }

        if self.disable_web_ui {
            args.push("--no-webui".into());
        } else {
            args.push("--webui".into());
        }

        args
    }

    /// Build the full command preview string
    pub fn command_preview(&self, server_exe: &Path) -> String {
        let exe = server_exe.display().to_string();
        let mut parts = vec![format!("\"{}\"", exe)];
        for arg in self.build_server_args() {
            if arg.contains(' ') || arg.contains('"') {
                parts.push(format!("\"{}\"", arg.replace('"', "\\\"")));
            } else {
                parts.push(arg);
            }
        }
        if !self.extra_args.is_empty() {
            parts.push(self.extra_args.trim().to_string());
        }
        parts.join(" ")
    }
}
