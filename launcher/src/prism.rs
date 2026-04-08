// ─── Prism MCP Deployment ─────────────────────────────────────────────────────
// Handles detection of AI coding harnesses, pre-flight validation of the
// inference stack, building prism-mcp from source, and writing MCP config
// entries to each harness's configuration.

use std::path::PathBuf;
use std::process::Command;

// ─── Types ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Harness {
    ClaudeDesktop,
    ClaudeCode,
    Cursor,
    Windsurf,
    VSCodeContinue,
}

impl Harness {
    pub fn label(&self) -> &'static str {
        match self {
            Harness::ClaudeDesktop => "Claude Desktop",
            Harness::ClaudeCode => "Claude Code (CLI)",
            Harness::Cursor => "Cursor",
            Harness::Windsurf => "Windsurf",
            Harness::VSCodeContinue => "VS Code + Continue",
        }
    }

    pub fn all() -> Vec<Harness> {
        vec![
            Harness::ClaudeDesktop,
            Harness::ClaudeCode,
            Harness::Cursor,
            Harness::Windsurf,
            Harness::VSCodeContinue,
        ]
    }
}

#[derive(Debug, Clone)]
pub enum HarnessStatus {
    Checking,
    Found(PathBuf),
    NotFound(String),
    ManualPath(PathBuf),
}

#[derive(Debug, Clone)]
pub struct HarnessProbe {
    pub harness: Harness,
    pub status: HarnessStatus,
    pub selected: bool,
}

impl HarnessProbe {
    pub fn is_available(&self) -> bool {
        matches!(self.status, HarnessStatus::Found(_) | HarnessStatus::ManualPath(_))
    }

    pub fn config_path(&self) -> Option<&PathBuf> {
        match &self.status {
            HarnessStatus::Found(p) | HarnessStatus::ManualPath(p) => Some(p),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PrismReadiness {
    pub text_ok: bool,
    pub embedding_ok: bool,
    pub embedding_dims_ok: bool,
    pub node_available: bool,
    pub npm_available: bool,
    pub prism_built: bool,
    pub dist_path: PathBuf,
    pub node_path: PathBuf,
}

impl PrismReadiness {
    pub fn all_ok(&self) -> bool {
        self.text_ok
            && self.embedding_ok
            && self.embedding_dims_ok
            && self.node_available
            && self.npm_available
            && self.prism_built
    }
}

// ─── Endpoint Discovery ──────────────────────────────────────────────────────

pub fn discover_endpoints(config: &crate::config::LauncherConfig) -> (u16, u16) {
    let mut text_port = 8080u16;
    let mut embed_port = 8081u16;
    for slot in &config.slots {
        if slot.embedding_mode {
            embed_port = slot.port;
        } else {
            text_port = slot.port;
        }
    }
    (text_port, embed_port)
}

// ─── Node / npm Detection ────────────────────────────────────────────────────

pub fn find_node_exe() -> Option<PathBuf> {
    // Try PATH first
    if let Ok(output) = Command::new("node").arg("--version").output() {
        if output.status.success() {
            return Some(PathBuf::from("node"));
        }
    }
    // Common Windows locations
    for candidate in &[
        r"C:\Program Files\nodejs\node.exe",
        r"C:\Program Files (x86)\nodejs\node.exe",
    ] {
        let p = PathBuf::from(candidate);
        if p.exists() {
            return Some(p);
        }
    }
    // nvm-windows: check APPDATA\nvm\*\node.exe
    if let Some(appdata) = dirs::data_dir() {
        let nvm_dir = appdata
            .parent()
            .unwrap_or(&appdata)
            .join("Roaming")
            .join("nvm");
        if nvm_dir.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&nvm_dir) {
                for entry in entries.flatten() {
                    let node = entry.path().join("node.exe");
                    if node.exists() {
                        return Some(node);
                    }
                }
            }
        }
    }
    None
}

pub fn find_npm_exe() -> Option<PathBuf> {
    // 1. Try PATH (works when launched from terminal)
    if let Ok(output) = Command::new("npm").arg("--version").output() {
        if output.status.success() {
            return Some(PathBuf::from("npm"));
        }
    }
    // 2. Next to node.exe (covers nvm shims, direct installs)
    if let Some(node) = find_node_exe() {
        if let Some(dir) = node.parent() {
            let npm_cmd = dir.join("npm.cmd");
            if npm_cmd.exists() {
                return Some(npm_cmd);
            }
        }
    }
    // 3. Use `where` command — finds executables across the full system PATH
    //    (more reliable than inherited process PATH for GUI apps)
    if let Ok(output) = Command::new("cmd").args(["/c", "where npm.cmd"]).output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().lines().next()
                .map(|s| PathBuf::from(s.trim()));
            if let Some(p) = path {
                if p.exists() { return Some(p); }
            }
        }
    }
    None
}

// ─── Pre-flight Checks ──────────────────────────────────────────────────────

pub async fn check_prism_readiness(
    config: &crate::config::LauncherConfig,
    repo_root: &std::path::Path,
) -> PrismReadiness {
    let (text_port, embed_port) = discover_endpoints(config);
    let host = &config.host;

    // Search for prism-mcp in multiple locations
    let prism_dir = {
        let mut found = repo_root.join("prism-mcp");
        if let Ok(exe) = std::env::current_exe() {
            if let Some(exe_dir) = exe.parent() {
                let next_to_exe = exe_dir.join("prism-mcp");
                if next_to_exe.join("package.json").exists() { found = next_to_exe; }
            }
        }
        found
    };
    let dist_path = prism_dir.join("dist").join("server.js");
    let node_path = find_node_exe().unwrap_or_default();
    let npm_available = find_npm_exe().is_some();

    // Health checks
    let text_url = format!("http://{}:{}/health", host, text_port);
    let embed_url = format!("http://{}:{}/health", host, embed_port);

    let text_ok = reqwest::get(&text_url)
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);

    let embedding_ok = reqwest::get(&embed_url)
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);

    // Embedding dimension test
    let embedding_dims_ok = if embedding_ok {
        test_embedding_dims(host, embed_port).await
    } else {
        false
    };

    PrismReadiness {
        text_ok,
        embedding_ok,
        embedding_dims_ok,
        node_available: !node_path.as_os_str().is_empty(),
        npm_available,
        prism_built: dist_path.exists(),
        dist_path,
        node_path,
    }
}

async fn test_embedding_dims(host: &str, port: u16) -> bool {
    let url = format!("http://{}:{}/v1/embeddings", host, port);
    let body = serde_json::json!({
        "model": "nomic-embed-text-v2-moe",
        "input": "test embedding dimension check"
    });

    let client = reqwest::Client::new();
    match client.post(&url).json(&body).send().await {
        Ok(resp) => {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                if let Some(dims) = json["data"][0]["embedding"].as_array() {
                    return dims.len() == 768;
                }
            }
            false
        }
        Err(_) => false,
    }
}

// ─── Build Prism ─────────────────────────────────────────────────────────────

pub fn build_prism(repo_root: &std::path::Path) -> Result<String, String> {
    // Search for prism-mcp in multiple locations:
    // 1. Next to launcher exe (package layout)
    // 2. Repo root (dev layout)
    // 3. Sibling of exe parent (e.g. C:\TOOLS\Bonzai\prism-mcp)
    let candidates = {
        let mut c = vec![repo_root.join("prism-mcp")];
        if let Ok(exe) = std::env::current_exe() {
            if let Some(exe_dir) = exe.parent() {
                c.push(exe_dir.join("prism-mcp"));
                // Also check parent of exe dir (if exe is in bin/ or system/)
                if let Some(parent) = exe_dir.parent() {
                    c.push(parent.join("prism-mcp"));
                }
            }
        }
        c
    };
    let prism_dir = candidates.iter().find(|p| p.join("package.json").exists())
        .ok_or("prism-mcp directory not found. Place it next to the launcher or run git submodule update --init")?
        .clone();

    let npm = find_npm_exe().ok_or("npm not found. Install Node.js from nodejs.org")?;

    // npm install
    let install = Command::new(&npm)
        .arg("install")
        .arg("--production")
        .current_dir(&prism_dir)
        .output()
        .map_err(|e| format!("Failed to run npm install: {}", e))?;

    if !install.status.success() {
        let stderr = String::from_utf8_lossy(&install.stderr);
        return Err(format!("npm install failed: {}", stderr));
    }

    // npm run build
    let build = Command::new(&npm)
        .arg("run")
        .arg("build")
        .current_dir(&prism_dir)
        .output()
        .map_err(|e| format!("Failed to run npm run build: {}", e))?;

    if !build.status.success() {
        let stderr = String::from_utf8_lossy(&build.stderr);
        return Err(format!("npm run build failed: {}", stderr));
    }

    let dist = prism_dir.join("dist").join("server.js");
    if dist.exists() {
        Ok(format!("Build complete: {}", dist.display()))
    } else {
        Err("Build completed but dist/server.js not found".into())
    }
}

// ─── Harness Detection ──────────────────────────────────────────────────────

pub fn detect_harness(harness: &Harness) -> HarnessProbe {
    let status = match harness {
        Harness::ClaudeDesktop => detect_claude_desktop(),
        Harness::ClaudeCode => detect_claude_code(),
        Harness::Cursor => detect_cursor(),
        Harness::Windsurf => detect_windsurf(),
        Harness::VSCodeContinue => detect_vscode_continue(),
    };
    HarnessProbe {
        harness: harness.clone(),
        status,
        selected: false,
    }
}

pub fn detect_all_harnesses() -> Vec<HarnessProbe> {
    Harness::all().iter().map(|h| detect_harness(h)).collect()
}

fn detect_claude_desktop() -> HarnessStatus {
    if let Some(appdata) = dirs::config_dir() {
        let config = appdata.join("Claude").join("claude_desktop_config.json");
        if config.exists() {
            return HarnessStatus::Found(config);
        }
        // Directory exists but no config file yet — still valid target
        let dir = appdata.join("Claude");
        if dir.exists() {
            return HarnessStatus::Found(config);
        }
    }
    HarnessStatus::NotFound("Claude Desktop not found in %APPDATA%".into())
}

fn detect_claude_code() -> HarnessStatus {
    // Claude Code uses CLI, not a config file we write directly
    if let Ok(output) = Command::new("claude").arg("--version").output() {
        if output.status.success() {
            return HarnessStatus::Found(PathBuf::from("claude-cli"));
        }
    }
    HarnessStatus::NotFound("Claude Code CLI not on PATH".into())
}

fn detect_cursor() -> HarnessStatus {
    if let Some(appdata) = dirs::config_dir() {
        let cursor_dir = appdata.join("Cursor");
        if cursor_dir.exists() {
            // Cursor uses .cursor/mcp.json in project or global config
            let config = cursor_dir
                .join("User")
                .join("globalStorage")
                .join("anysphere.cursor-mcp")
                .join("mcp.json");
            return HarnessStatus::Found(config);
        }
    }
    HarnessStatus::NotFound("Cursor not found in %APPDATA%".into())
}

fn detect_windsurf() -> HarnessStatus {
    if let Some(appdata) = dirs::config_dir() {
        let windsurf_dir = appdata.join("Windsurf");
        if windsurf_dir.exists() {
            let config = windsurf_dir
                .join("User")
                .join("globalStorage")
                .join("codeium.windsurf-mcp")
                .join("mcp.json");
            return HarnessStatus::Found(config);
        }
    }
    HarnessStatus::NotFound("Windsurf not found in %APPDATA%".into())
}

fn detect_vscode_continue() -> HarnessStatus {
    if let Some(appdata) = dirs::config_dir() {
        let continue_dir = appdata
            .join("Code")
            .join("User")
            .join("globalStorage")
            .join("continuedev.continue");
        if continue_dir.exists() {
            let config = continue_dir.join("config.json");
            return HarnessStatus::Found(config);
        }
    }
    HarnessStatus::NotFound("VS Code + Continue not found".into())
}

// ─── Deployment ─────────────────────────────────────────────────────────────

pub fn build_mcp_entry(
    readiness: &PrismReadiness,
    config: &crate::config::LauncherConfig,
    dashboard_port: u16,
) -> serde_json::Value {
    let (text_port, embed_port) = discover_endpoints(config);
    let host = &config.host;

    // Find text model alias
    let text_alias = config
        .slots
        .iter()
        .find(|s| !s.embedding_mode)
        .map(|s| s.alias.as_str())
        .unwrap_or("Bonsai-8B");

    // Find embedding model alias
    let embed_alias = config
        .slots
        .iter()
        .find(|s| s.embedding_mode)
        .map(|s| s.alias.as_str())
        .unwrap_or("nomic-embed-text-v2-moe");

    let node_path = readiness.node_path.display().to_string();
    let dist_path = readiness.dist_path.display().to_string();

    serde_json::json!({
        "command": node_path,
        "args": [dist_path],
        "env": {
            "TEXT_PROVIDER": "llamacpp",
            "EMBEDDING_PROVIDER": "llamacpp",
            "PRISM_STORAGE": "local",
            "PRISM_DASHBOARD_PORT": dashboard_port.to_string(),
            "LLAMACPP_TEXT_URL": format!("http://{}:{}/v1", host, text_port),
            "LLAMACPP_EMBEDDING_URL": format!("http://{}:{}/v1", host, embed_port),
            "LLAMACPP_TEXT_MODEL": text_alias,
            "LLAMACPP_EMBEDDING_MODEL": embed_alias,
        }
    })
}

pub fn build_bonsai_entry(
    readiness: &PrismReadiness,
    repo_root: &std::path::Path,
    api_port: u16,
    api_host: &str,
) -> serde_json::Value {
    let node_path = readiness.node_path.display().to_string();
    let dist_path = repo_root
        .join("bonsai-mcp")
        .join("dist")
        .join("server.js")
        .display()
        .to_string();

    serde_json::json!({
        "command": node_path,
        "args": [dist_path],
        "env": {
            "BONSAI_API_PORT": api_port.to_string(),
            "BONSAI_API_HOST": api_host,
        }
    })
}

pub fn build_bonsai_mcp(repo_root: &std::path::Path) -> Result<String, String> {
    let bonsai_dir = repo_root.join("bonsai-mcp");
    if !bonsai_dir.exists() {
        return Err("bonsai-mcp directory not found".into());
    }

    let npm = find_npm_exe().ok_or("npm not found. Install Node.js from nodejs.org")?;

    // npm install
    let install = Command::new(&npm)
        .arg("install")
        .current_dir(&bonsai_dir)
        .output()
        .map_err(|e| format!("Failed to run npm install: {}", e))?;

    if !install.status.success() {
        let stderr = String::from_utf8_lossy(&install.stderr);
        return Err(format!("npm install failed in bonsai-mcp: {}", stderr));
    }

    // npm run build
    let build = Command::new(&npm)
        .arg("run")
        .arg("build")
        .current_dir(&bonsai_dir)
        .output()
        .map_err(|e| format!("Failed to run npm run build: {}", e))?;

    if !build.status.success() {
        let stderr = String::from_utf8_lossy(&build.stderr);
        return Err(format!("npm run build failed in bonsai-mcp: {}", stderr));
    }

    let dist = bonsai_dir.join("dist").join("server.js");
    if dist.exists() {
        Ok(format!("bonsai-mcp build complete: {}", dist.display()))
    } else {
        Err("bonsai-mcp build completed but dist/server.js not found".into())
    }
}

pub fn deploy_to_harness(
    probe: &HarnessProbe,
    entries: &[(&str, &serde_json::Value)],
) -> Result<Vec<String>, String> {
    let mut results = Vec::new();
    for (name, entry) in entries {
        let result = match probe.harness {
            Harness::ClaudeCode => deploy_claude_code(name, entry),
            _ => {
                let path = probe
                    .config_path()
                    .ok_or_else(|| format!("{}: no config path", probe.harness.label()))?;
                write_mcp_config_json(path, name, entry)
            }
        }?;
        results.push(result);
    }
    Ok(results)
}

fn deploy_claude_code(name: &str, entry: &serde_json::Value) -> Result<String, String> {
    let command = entry["command"].as_str().unwrap_or("node");
    let args: Vec<&str> = entry["args"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    let env = entry["env"].as_object();

    let mut cmd = Command::new("claude");
    cmd.arg("mcp").arg("add").arg("-s").arg("user").arg(name);

    // Add env vars after the server name
    if let Some(env_map) = env {
        for (key, val) in env_map {
            if let Some(v) = val.as_str() {
                cmd.arg("-e").arg(format!("{}={}", key, v));
            }
        }
    }

    cmd.arg("--").arg(command);
    for a in &args {
        cmd.arg(a);
    }

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to run claude mcp add: {}", e))?;

    if output.status.success() {
        Ok(format!("{}: registered via `claude mcp add` (user scope)", name))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("claude mcp add {} failed: {}", name, stderr))
    }
}

fn write_mcp_config_json(config_path: &std::path::Path, name: &str, entry: &serde_json::Value) -> Result<String, String> {
    // Ensure parent directory exists
    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create directory {}: {}", parent.display(), e))?;
    }

    // Read existing config or start fresh
    let mut root: serde_json::Value = if config_path.exists() {
        let content = std::fs::read_to_string(config_path).map_err(|e| {
            format!(
                "Cannot read {}: {} — is the app running? Close it and retry.",
                config_path.display(),
                e
            )
        })?;
        match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(_) => {
                // Backup invalid JSON
                let bak = config_path.with_extension("json.bak");
                let _ = std::fs::copy(config_path, &bak);
                serde_json::json!({})
            }
        }
    } else {
        serde_json::json!({})
    };

    // Ensure mcpServers object exists
    if !root.get("mcpServers").is_some() {
        root["mcpServers"] = serde_json::json!({});
    }

    // Insert/update entry — preserves all other servers
    root["mcpServers"][name] = entry.clone();

    // Write back pretty-printed
    let output = serde_json::to_string_pretty(&root)
        .map_err(|e| format!("JSON serialization failed: {}", e))?;

    std::fs::write(config_path, &output).map_err(|e| {
        format!(
            "Cannot write {}: {} — check file permissions or close the app.",
            config_path.display(),
            e
        )
    })?;

    Ok(format!("{}: written to {}", name, config_path.display()))
}
