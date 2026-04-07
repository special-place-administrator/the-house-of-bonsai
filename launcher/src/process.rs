use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{RwLock, broadcast};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

use crate::config::{LauncherConfig, ModelSlot};

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

// ---------------------------------------------------------------------------
// Per-slot process state
// ---------------------------------------------------------------------------

pub struct ProcessSlot {
    child: Option<Child>,
    child_pid: Option<u32>,
    pub status: ServerStatus,
    pub runtime: Option<RuntimeInfo>,
}

impl ProcessSlot {
    fn new() -> Self {
        Self {
            child: None,
            child_pid: None,
            status: ServerStatus::Stopped,
            runtime: None,
        }
    }

    pub fn is_running(&self) -> bool {
        matches!(self.status, ServerStatus::Starting | ServerStatus::Running | ServerStatus::Unhealthy)
    }
}

// ---------------------------------------------------------------------------
// Multi-process manager
// ---------------------------------------------------------------------------

pub struct ProcessManager {
    server_exe: PathBuf,
    pub slots: Vec<ProcessSlot>,
    pub log_tx: broadcast::Sender<String>,
    pub stdout_lines: Vec<String>,
    pub stderr_lines: Vec<String>,
}

const MAX_LOG_LINES: usize = 5000;

impl ProcessManager {
    pub fn new(repo_root: &Path) -> Self {
            // Look for llama-server.exe in multiple locations:
            // 1. Next to the launcher exe (release/portable layout)
            // 2. In the build output directory (development layout)
            let server_exe = if let Ok(exe_path) = std::env::current_exe() {
                let exe_dir = exe_path.parent().unwrap_or(Path::new("."));
                let portable = exe_dir.join("llama-server.exe");
                if portable.exists() {
                    portable
                } else {
                    repo_root.join("build").join("bin").join("llama-server.exe")
                }
            } else {
                repo_root.join("build").join("bin").join("llama-server.exe")
            };

            let (log_tx, _) = broadcast::channel(256);
            Self {
                server_exe,
                slots: Vec::new(),
                log_tx,
                stdout_lines: Vec::new(),
                stderr_lines: Vec::new(),
            }
        }

    /// Ensure we have at least `n` ProcessSlot entries, adding empty ones as
    /// needed.  Called before start to keep slots in sync with config.
    pub fn ensure_slots(&mut self, n: usize) {
        while self.slots.len() < n {
            self.slots.push(ProcessSlot::new());
        }
        // Trim excess if config shrank
        self.slots.truncate(n);
    }

    pub fn server_exe_exists(&self) -> bool {
        self.server_exe.exists()
    }

    pub fn server_exe_path(&self) -> &Path {
        &self.server_exe
    }

    // -- single-slot operations -----------------------------------------------

    pub async fn start_slot(
        &mut self,
        index: usize,
        slot_config: &ModelSlot,
        shared_config: &LauncherConfig,
    ) -> Result<(), String> {
        self.ensure_slots(index + 1);
        let ps = &self.slots[index];

        if ps.child.is_some() {
            return Err(format!("Slot {} is already running. Stop it first.", index));
        }
        if !self.server_exe.exists() {
            return Err(format!("llama-server.exe not found at: {}", self.server_exe.display()));
        }
        if slot_config.model_path.is_empty() || !Path::new(&slot_config.model_path).exists() {
            if shared_config.extra_args.is_empty() {
                return Err(format!("Slot {}: No model selected. Pick a GGUF file first.", index));
            }
        }

        // Reset slot state
        let ps = &mut self.slots[index];
        ps.status = ServerStatus::Starting;

        let mut args = slot_config.build_server_args(shared_config);
        if !shared_config.extra_args.is_empty() {
            for part in shared_config.extra_args.split_whitespace() {
                args.push(part.to_string());
            }
        }

        let mut cmd = Command::new(&self.server_exe);
        cmd.args(&args)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        // Set TURBO_LAYER_ADAPTIVE env
        if slot_config.turbo_layer_adaptive != "off" {
            cmd.env("TURBO_LAYER_ADAPTIVE", &slot_config.turbo_layer_adaptive);
        }

        #[cfg(target_os = "windows")]
        {
            // Suppress missing DLL error dialogs (e.g. cublas64_13.dll when CUDA
            // runtime isn't installed). SEM_FAILCRITICALERRORS lets llama-server
            // fail gracefully instead of showing a Windows popup.
            unsafe {
                #[link(name = "kernel32")]
                unsafe extern "system" { fn SetErrorMode(mode: u32) -> u32; }
                SetErrorMode(0x0001); // SEM_FAILCRITICALERRORS
            }
            cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }

        let mut child = cmd.spawn()
            .map_err(|e| format!("Slot {}: Failed to spawn llama-server: {e}", index))?;

        let pid = child.id().unwrap_or(0);
        let ps = &mut self.slots[index];
        ps.child_pid = Some(pid);
        ps.runtime = Some(RuntimeInfo {
            pid,
            started_at: chrono::Utc::now(),
            model_path: slot_config.model_path.clone(),
            host: shared_config.host.clone(),
            port: slot_config.port,
        });

        let slot_prefix = format!("[slot-{}]", index);

        // Capture stdout
        if let Some(stdout) = child.stdout.take() {
            let tx = self.log_tx.clone();
            let prefix = slot_prefix.clone();
            tokio::spawn(async move {
                let reader = BufReader::new(stdout);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let _ = tx.send(format!("{} [stdout] {}", prefix, line));
                }
            });
        }

        // Capture stderr
        if let Some(stderr) = child.stderr.take() {
            let tx = self.log_tx.clone();
            let prefix = slot_prefix;
            tokio::spawn(async move {
                let reader = BufReader::new(stderr);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let _ = tx.send(format!("{} [stderr] {}", prefix, line));
                }
            });
        }

        self.slots[index].child = Some(child);

        // Save combined runtime state
        let _ = self.save_runtime_state();
        Ok(())
    }

    pub async fn stop_slot(&mut self, index: usize) -> Result<(), String> {
        if index >= self.slots.len() { return Ok(()); }
        let ps = &mut self.slots[index];

        if let Some(pid) = ps.child_pid.take() {
            if pid > 0 {
                let mut cmd = std::process::Command::new("taskkill");
                cmd.args(["/F", "/PID", &pid.to_string()]);
                #[cfg(target_os = "windows")]
                cmd.creation_flags(0x08000000);
                let _ = cmd.output();
            }
        }

        if let Some(mut child) = ps.child.take() {
            let _ = child.wait().await;
        }

        ps.status = ServerStatus::Stopped;
        ps.runtime = None;

        // Update persisted state
        if self.is_any_running() {
            let _ = self.save_runtime_state();
        } else {
            self.clear_runtime_state();
        }
        Ok(())
    }

    // -- batch operations -----------------------------------------------------

    pub async fn start_all(&mut self, config: &LauncherConfig) -> Vec<Result<(), String>> {
        self.ensure_slots(config.slots.len());
        let mut results = Vec::new();
        for i in 0..config.slots.len() {
            let r = self.start_slot(i, &config.slots[i].clone(), config).await;
            results.push(r);
        }
        results
    }

    pub async fn stop_all(&mut self) {
        let len = self.slots.len();
        for i in 0..len {
            let _ = self.stop_slot(i).await;
        }
    }

    // -- status queries -------------------------------------------------------

    pub fn is_any_running(&self) -> bool {
        self.slots.iter().any(|s| s.is_running())
    }

    pub fn is_all_running(&self) -> bool {
        !self.slots.is_empty() && self.slots.iter().all(|s| s.is_running())
    }

    pub fn running_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_running()).count()
    }

    /// Aggregate status label for the header badge.
    pub fn aggregate_status(&self) -> (String, String) {
        let total = self.slots.len();
        let running = self.running_count();
        if running == 0 {
            // Check for crashes
            let crashed = self.slots.iter().any(|s| matches!(s.status, ServerStatus::Crashed(_)));
            if crashed {
                let msg = self.slots.iter()
                    .filter_map(|s| if let ServerStatus::Crashed(m) = &s.status { Some(m.as_str()) } else { None })
                    .next()
                    .unwrap_or("unknown");
                (format!("Crashed: {}", msg), "#f44336".into())
            } else {
                ("Stopped".into(), "#888".into())
            }
        } else if running == total {
            ("Running".into(), "#4caf50".into())
        } else {
            (format!("Partial ({}/{})", running, total), "#e6a817".into())
        }
    }

    // -- health checks --------------------------------------------------------

    pub async fn check_alive_slot(&mut self, index: usize) -> bool {
        if index >= self.slots.len() { return false; }
        let ps = &mut self.slots[index];
        if let Some(child) = &mut ps.child {
            match child.try_wait() {
                Ok(Some(exit_status)) => {
                    let msg = format!("exit code: {}", exit_status);
                    ps.status = ServerStatus::Crashed(msg);
                    ps.child = None;
                    false
                }
                Ok(None) => true,
                Err(_) => false,
            }
        } else {
            false
        }
    }

    pub async fn health_check_slot(&mut self, index: usize) {
        if !self.check_alive_slot(index).await {
            return;
        }
        let ps = &self.slots[index];
        let Some(info) = &ps.runtime else { return };
        let url = format!("http://{}:{}/health", info.host, info.port);

        let healthy = match reqwest::get(&url).await {
            Ok(resp) if resp.status().is_success() => true,
            _ => false,
        };

        let ps = &mut self.slots[index];
        if healthy {
            ps.status = ServerStatus::Running;
        } else if ps.status != ServerStatus::Starting {
            ps.status = ServerStatus::Unhealthy;
        }
    }

    pub async fn health_check_all(&mut self) {
        let len = self.slots.len();
        for i in 0..len {
            if self.slots[i].is_running() {
                self.health_check_slot(i).await;
            }
        }
    }

    // -- runtime state persistence --------------------------------------------

    fn save_runtime_state(&self) -> Result<(), String> {
        let active: Vec<_> = self.slots.iter()
            .filter_map(|ps| ps.runtime.as_ref())
            .map(|info| serde_json::json!({
                "pid": info.pid,
                "host": info.host,
                "port": info.port,
                "startedAt": info.started_at.to_rfc3339(),
                "modelPath": info.model_path,
            }))
            .collect();

        let state = serde_json::json!({ "slots": active });
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
            mgr.health_check_all().await;
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
                    if line.contains("[stderr]") {
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
                    rx = {
                        let mgr = pm2.read().await;
                        mgr.log_tx.subscribe()
                    };
                }
            }
        }
    });
}


/// Kill ALL llama-server.exe processes on the system.
/// Called on launcher exit to prevent zombie processes.
/// Tries graceful SIGTERM first, waits up to 3 seconds, then force-kills.
pub fn kill_all_llama_servers() {
    #[cfg(target_os = "windows")]
    {
        // First pass: taskkill (graceful)
        let mut cmd = std::process::Command::new("taskkill");
        cmd.args(["/IM", "llama-server.exe"]);
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
        let _ = cmd.output();

        // Wait up to 3 seconds for graceful shutdown
        std::thread::sleep(std::time::Duration::from_secs(3));

        // Second pass: force kill any survivors
        let mut cmd = std::process::Command::new("taskkill");
        cmd.args(["/F", "/IM", "llama-server.exe"]);
        cmd.creation_flags(0x08000000);
        let _ = cmd.output();

        // Third pass: verify they're dead
        let check = std::process::Command::new("tasklist")
            .args(["/FI", "IMAGENAME eq llama-server.exe"])
            .creation_flags(0x08000000)
            .output();

        if let Ok(output) = check {
            let text = String::from_utf8_lossy(&output.stdout);
            if text.contains("llama-server.exe") {
                // Nuclear option: wmic
                let mut cmd = std::process::Command::new("wmic");
                cmd.args(["process", "where", "name='llama-server.exe'", "delete"]);
                cmd.creation_flags(0x08000000);
                let _ = cmd.output();
            }
        }
    }
}
