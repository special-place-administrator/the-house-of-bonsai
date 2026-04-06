use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{RwLock, broadcast};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

use crate::config::LauncherConfig;

#[derive(Debug, Clone, PartialEq)]
pub enum ServerStatus {
    Stopped,
    Starting,
    Running,
    Unhealthy,
    Crashed(String),
}

impl std::fmt::Display for ServerStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stopped => write!(f, "Stopped"),
            Self::Starting => write!(f, "Starting..."),
            Self::Running => write!(f, "Running"),
            Self::Unhealthy => write!(f, "Unhealthy"),
            Self::Crashed(msg) => write!(f, "Crashed: {}", msg),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeInfo {
    pub pid: u32,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub model_path: String,
    pub host: String,
    pub port: u16,
}

pub struct ProcessManager {
    server_exe: PathBuf,
    child: Option<Child>,
    child_pid: Option<u32>,
    pub status: ServerStatus,
    pub runtime: Option<RuntimeInfo>,
    pub log_tx: broadcast::Sender<String>,
    pub stdout_lines: Vec<String>,
    pub stderr_lines: Vec<String>,
}

const MAX_LOG_LINES: usize = 5000;

impl ProcessManager {
    pub fn new(repo_root: &Path) -> Self {
        let server_exe = repo_root.join("build").join("bin").join("llama-server.exe");
        let (log_tx, _) = broadcast::channel(256);
        Self {
            server_exe,
            child: None,
            child_pid: None,
            status: ServerStatus::Stopped,
            runtime: None,
            log_tx,
            stdout_lines: Vec::new(),
            stderr_lines: Vec::new(),
        }
    }

    pub fn server_exe_exists(&self) -> bool {
        self.server_exe.exists()
    }

    pub fn server_exe_path(&self) -> &Path {
        &self.server_exe
    }

    pub async fn start(&mut self, config: &LauncherConfig) -> Result<(), String> {
        if self.child.is_some() {
            return Err("Server is already running. Stop it first.".into());
        }
        if !self.server_exe.exists() {
            return Err(format!("llama-server.exe not found at: {}", self.server_exe.display()));
        }
        if config.model_path.is_empty() || !Path::new(&config.model_path).exists() {
            if config.extra_args.is_empty() {
                return Err("No model selected. Pick a GGUF file first.".into());
            }
        }

        self.stdout_lines.clear();
        self.stderr_lines.clear();
        self.status = ServerStatus::Starting;

        let mut args = config.build_server_args();
        if !config.extra_args.is_empty() {
            // Split extra args on whitespace (simple — doesn't handle quotes)
            for part in config.extra_args.split_whitespace() {
                args.push(part.to_string());
            }
        }

        let mut cmd = Command::new(&self.server_exe);
        cmd.args(&args)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        // Set TURBO_LAYER_ADAPTIVE env
        if config.turbo_layer_adaptive != "off" {
            cmd.env("TURBO_LAYER_ADAPTIVE", &config.turbo_layer_adaptive);
        }

        // CREATE_NO_WINDOW on Windows
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }

        let mut child = cmd.spawn()
            .map_err(|e| format!("Failed to spawn llama-server: {e}"))?;

        let pid = child.id().unwrap_or(0);
        self.child_pid = Some(pid);
        self.runtime = Some(RuntimeInfo {
            pid,
            started_at: chrono::Utc::now(),
            model_path: config.model_path.clone(),
            host: config.host.clone(),
            port: config.port,
        });

        // Save runtime state for external tools
        let _ = self.save_runtime_state();

        // Capture stdout
        if let Some(stdout) = child.stdout.take() {
            let tx = self.log_tx.clone();
            tokio::spawn(async move {
                let reader = BufReader::new(stdout);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let _ = tx.send(format!("[stdout] {}", line));
                }
            });
        }

        // Capture stderr
        if let Some(stderr) = child.stderr.take() {
            let tx = self.log_tx.clone();
            tokio::spawn(async move {
                let reader = BufReader::new(stderr);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let _ = tx.send(format!("[stderr] {}", line));
                }
            });
        }

        self.child = Some(child);
        Ok(())
    }

    pub async fn stop(&mut self) -> Result<(), String> {
        // Always use taskkill with stored PID — most reliable on Windows
        if let Some(pid) = self.child_pid.take() {
            if pid > 0 {
                let mut cmd = std::process::Command::new("taskkill");
                cmd.args(["/F", "/PID", &pid.to_string()]);
                #[cfg(target_os = "windows")]
                cmd.creation_flags(0x08000000);
                let _ = cmd.output();
            }
        }

        // Clean up tokio child handle
        if let Some(mut child) = self.child.take() {
            let _ = child.wait().await;
        }

        self.status = ServerStatus::Stopped;
        self.runtime = None;
        self.clear_runtime_state();
        Ok(())
    }

    pub fn is_running(&self) -> bool {
        matches!(self.status, ServerStatus::Starting | ServerStatus::Running | ServerStatus::Unhealthy)
    }

    pub async fn check_alive(&mut self) -> bool {
        if let Some(child) = &mut self.child {
            match child.try_wait() {
                Ok(Some(exit_status)) => {
                    // Process has exited
                    let msg = format!("exit code: {}", exit_status);
                    self.status = ServerStatus::Crashed(msg);
                    self.child = None;
                    self.clear_runtime_state();
                    false
                }
                Ok(None) => true, // Still running
                Err(_) => false,
            }
        } else {
            false
        }
    }

    pub async fn health_check(&mut self) {
        if !self.check_alive().await {
            return;
        }

        let Some(info) = &self.runtime else { return };
        let url = format!("http://{}:{}/health", info.host, info.port);

        match reqwest::get(&url).await {
            Ok(resp) if resp.status().is_success() => {
                self.status = ServerStatus::Running;
            }
            _ => {
                if self.status != ServerStatus::Starting {
                    self.status = ServerStatus::Unhealthy;
                }
            }
        }
    }

    fn save_runtime_state(&self) -> Result<(), String> {
        let Some(info) = &self.runtime else { return Ok(()) };
        let state = serde_json::json!({
            "pid": info.pid,
            "host": info.host,
            "port": info.port,
            "startedAt": info.started_at.to_rfc3339(),
            "modelPath": info.model_path,
        });
        let path = LauncherConfig::runtime_state_path();
        let _ = std::fs::create_dir_all(path.parent().unwrap());
        std::fs::write(&path, serde_json::to_string_pretty(&state).unwrap_or_default())
            .map_err(|e| format!("Failed to save runtime state: {e}"))
    }

    fn clear_runtime_state(&self) {
        let path = LauncherConfig::runtime_state_path();
        let _ = std::fs::remove_file(path);
    }
}

/// Shared process manager wrapped for concurrent access
pub type SharedProcessManager = Arc<RwLock<ProcessManager>>;

/// Spawn a background health-check loop
pub fn spawn_health_loop(pm: SharedProcessManager) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            let mut mgr = pm.write().await;
            mgr.health_check().await;
        }
    });
}

/// Spawn a background log collector that feeds lines into the ProcessManager's buffers
pub fn spawn_log_collector(pm: SharedProcessManager) {
    let pm2 = pm.clone();
    tokio::spawn(async move {
        let mut rx = {
            let mgr = pm2.read().await;
            mgr.log_tx.subscribe()
        };
        loop {
            match rx.recv().await {
                Ok(line) => {
                    let mut mgr = pm2.write().await;
                    if line.starts_with("[stderr]") {
                        mgr.stderr_lines.push(line.clone());
                        if mgr.stderr_lines.len() > MAX_LOG_LINES {
                            mgr.stderr_lines.remove(0);
                        }
                    } else {
                        mgr.stdout_lines.push(line.clone());
                        if mgr.stdout_lines.len() > MAX_LOG_LINES {
                            mgr.stdout_lines.remove(0);
                        }
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    // Re-subscribe if sender was dropped and recreated
                    rx = {
                        let mgr = pm2.read().await;
                        mgr.log_tx.subscribe()
                    };
                }
            }
        }
    });
}
