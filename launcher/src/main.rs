#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod config;
mod models;
mod process;
mod resources;

use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use dioxus::prelude::*;

use config::LauncherConfig;
use models::{ModelCatalog, ModelEntry, format_size};
use process::{ProcessManager, ServerStatus, SharedProcessManager};
use resources::SystemResources;

fn detect_repo_root() -> PathBuf {
    // Walk up from the executable to find the project root
    // Looks for markers: models/ dir, launcher/ dir, or llama-cpp/ dir
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().map(|p| p.to_path_buf());
        for _ in 0..6 {
            if let Some(d) = &dir {
                if d.join("launcher").is_dir() || d.join("models").is_dir() || d.join("llama-cpp").is_dir() {
                    return d.clone();
                }
                dir = d.parent().map(|p| p.to_path_buf());
            }
        }
    }
    // Fallback: current working directory
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn detect_model_root(repo_root: &Path) -> PathBuf {
    // Project-local models/ first
    let local = repo_root.join("models");
    if local.is_dir() { return local; }
    // Fallback to shared model root
    let shared = PathBuf::from(r"C:\AI_STUFF\LLM_MODEL");
    if shared.is_dir() { return shared; }
    local
}

fn detect_llama_root(repo_root: &Path) -> PathBuf {
    // Project-local llama-cpp/ first
    let local = repo_root.join("llama-cpp");
    if local.is_dir() { return local; }
    // Fallback to external llama.cpp
    let external = PathBuf::from(r"C:\AI_STUFF\PROGRAMMING\LLAMA\llama-cpp-turboquant-cuda");
    if external.is_dir() { return external; }
    local
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("turboquant_launcher=info")
        .init();

    let _ = std::fs::create_dir_all(LauncherConfig::config_dir());
    let _ = std::fs::create_dir_all(LauncherConfig::log_dir());

    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    let repo_root = use_signal(|| detect_repo_root());
    let llama_root = use_signal(|| {
        let root = detect_repo_root();
        detect_llama_root(&root)
    });
    let mut config = use_signal(|| LauncherConfig::load());
    let shared_root = use_signal(|| {
        let root = detect_repo_root();
        detect_model_root(&root)
    });

    let mut model_list = use_signal(|| {
        let cfg = LauncherConfig::load();
        let root = detect_repo_root();
        let model_root = detect_model_root(&root);
        ModelCatalog::scan(&model_root, &cfg.recent_models).entries
    });

    let mut pm: Signal<Option<SharedProcessManager>> = use_signal(|| None);
    let mut status_text = use_signal(|| "Stopped".to_string());
    let mut status_color = use_signal(|| "#888".to_string());
    let mut log_lines: Signal<Vec<String>> = use_signal(Vec::new);
    let mut server_running = use_signal(|| false);
    let mut active_tab = use_signal(|| "launch".to_string());
    let mut message: Signal<Option<(String, bool)>> = use_signal(|| None);

    // Initialize process manager once — uses llama-cpp root for server exe
    use_effect(move || {
        let root = llama_root.read().clone();
        let new_pm = Arc::new(RwLock::new(ProcessManager::new(&root)));
        process::spawn_health_loop(new_pm.clone());
        process::spawn_log_collector(new_pm.clone());
        pm.set(Some(new_pm));
    });

    // Poll status every 2s
    use_future(move || async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            if let Some(pm_ref) = pm.read().as_ref() {
                let mgr = pm_ref.read().await;
                let s = mgr.status.clone();
                let running = mgr.is_running();
                let lines: Vec<String> = mgr
                    .stderr_lines
                    .iter()
                    .chain(mgr.stdout_lines.iter())
                    .rev()
                    .take(200)
                    .cloned()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect();
                drop(mgr);

                let (text, color) = match &s {
                    ServerStatus::Stopped => ("Stopped".into(), "#888".into()),
                    ServerStatus::Starting => ("Starting...".into(), "#e6a817".into()),
                    ServerStatus::Running => ("Running".into(), "#4caf50".into()),
                    ServerStatus::Unhealthy => ("Unhealthy".into(), "#ff9800".into()),
                    ServerStatus::Crashed(msg) => (format!("Crashed: {msg}"), "#f44336".into()),
                };
                server_running.set(running);
                status_text.set(text);
                status_color.set(color);
                log_lines.set(lines);
            }
        }
    });

    // Toggle start/stop — single button
    let on_toggle_server = move |_| {
        let running = *server_running.read();
        spawn(async move {
            let pm_clone = pm.read().clone();
            if let Some(pm_ref) = pm_clone.as_ref() {
                let mut mgr = pm_ref.write().await;
                if running {
                    if let Err(e) = mgr.stop().await {
                        message.set(Some((e, true)));
                    }
                } else {
                    let cfg = config.read().clone();
                    match mgr.start(&cfg).await {
                        Ok(()) => status_text.set("Starting...".into()),
                        Err(e) => message.set(Some((e, true))),
                    }
                }
            }
        });
    };

    let on_save = move |_| {
        let cfg = config.read().clone();
        match cfg.save() {
            Ok(()) => message.set(Some(("Settings saved.".into(), false))),
            Err(e) => message.set(Some((e, true))),
        }
    };

    let on_refresh_models = move |_| {
        let cfg = config.read().clone();
        let root = shared_root.read().clone();
        let entries = ModelCatalog::scan(&root, &cfg.recent_models).entries;
        model_list.set(entries);
        message.set(Some(("Models refreshed.".into(), false)));
    };

    let on_open_chat = move |_| {
        let cfg = config.read();
        let _ = open::that(format!("http://{}:{}/", cfg.host, cfg.port));
    };

    let on_auto_tune = move |_| {
        let res = SystemResources::detect();
        let mut cfg = config.write();
        let model_path = cfg.model_path.clone();
        res.auto_tune(&mut cfg, &model_path);
        drop(cfg);
        message.set(Some((format!("Auto-tuned!\n{}", res.summary()), false)));
    };

    let on_build = move |_| {
        let root = llama_root.read().clone();
        let script = root.join("windows").join("Build-TurboQuant.ps1");
        if script.exists() {
            let _ = std::process::Command::new("powershell.exe")
                .args(["-NoLogo", "-ExecutionPolicy", "Bypass", "-NoExit", "-File"])
                .arg(&script)
                .current_dir(&root)
                .spawn();
        } else {
            message.set(Some(("Build script not found.".into(), true)));
        }
    };

    let cmd_preview = {
        let cfg = config.read();
        let root = repo_root.read();
        let exe = root.join("build").join("bin").join("llama-server.exe");
        cfg.command_preview(&exe)
    };

    rsx! {
        style { {include_str!("../assets/style.css")} }

        div { class: "app",
            // Header — background changes with server state
            div {
                class: if *server_running.read() { "header header-running" } else { "header header-stopped" },
                if *server_running.read() {
                    button { class: "btn btn-stop", onclick: on_toggle_server, "■ Stop" }
                } else {
                    button { class: "btn btn-start", onclick: on_toggle_server, "▶ Start" }
                }
                button { class: "btn", onclick: on_open_chat, "💬 Chat" }
                button { class: "btn", onclick: on_build, "🔧 Build" }
                button { class: "btn", onclick: on_save, "💾 Save" }
                button { class: "btn", onclick: on_refresh_models, "🔄 Models" }
                button { class: "btn btn-tune", onclick: on_auto_tune, "⚡ Auto-Tune" }
                div { class: "status-badge", style: "color: {status_color};",
                    "● {status_text}"
                }
            }

            // Message bar
            if let Some((msg, is_err)) = message.read().as_ref() {
                div {
                    class: if *is_err { "msg msg-error" } else { "msg msg-info" },
                    onclick: move |_| message.set(None),
                    "{msg} ✕"
                }
            }

            // Tabs
            div { class: "tabs",
                button {
                    class: if *active_tab.read() == "launch" { "tab active" } else { "tab" },
                    onclick: move |_| active_tab.set("launch".into()),
                    "Launch"
                }
                button {
                    class: if *active_tab.read() == "advanced" { "tab active" } else { "tab" },
                    onclick: move |_| active_tab.set("advanced".into()),
                    "Advanced"
                }
                button {
                    class: if *active_tab.read() == "logs" { "tab active" } else { "tab" },
                    onclick: move |_| active_tab.set("logs".into()),
                    "Logs"
                }
            }

            div { class: "tab-content",
                match active_tab.read().as_str() {
                    "launch" => rsx! {
                        LaunchTab { config, model_list }
                    },
                    "advanced" => rsx! {
                        AdvancedTab { config, cmd_preview: cmd_preview.clone() }
                    },
                    "logs" => rsx! {
                        LogsTab { log_lines, status_text: status_text.read().clone() }
                    },
                    _ => rsx! { div { "Unknown tab" } },
                }
            }
        }
    }
}

#[component]
fn LaunchTab(config: Signal<LauncherConfig>, model_list: Signal<Vec<ModelEntry>>) -> Element {
    let cfg = config.read().clone();

    rsx! {
        div { class: "panel-row",
            div { class: "panel",
                h3 { "Model" }
                div { class: "field",
                    label { "Model" }
                    select {
                        value: "{cfg.model_path}",
                        onchange: move |e: Event<FormData>| {
                            let val = e.value();
                            let mut c = config.write();
                            c.model_path = val.clone();
                            c.model_selection = val;
                        },
                        option { value: "", "-- Select a model --" }
                        for entry in model_list.read().iter() {
                            option {
                                value: "{entry.path.display()}",
                                selected: entry.path.to_string_lossy() == cfg.model_path,
                                "{entry.display_name} ({format_size(entry.size_bytes)})"
                            }
                        }
                    }
                }
                div { class: "field",
                    label { "Alias" }
                    input {
                        r#type: "text",
                        value: "{cfg.alias}",
                        onchange: move |e: Event<FormData>| config.write().alias = e.value(),
                    }
                }
            }

            div { class: "panel",
                h3 { "TurboQuant" }
                SelectField { label: "Cache Type K", value: cfg.cache_type_k.clone(),
                    options: vec!["f16","q8_0","turbo2","turbo3","turbo4"],
                    on_change: move |v: String| config.write().cache_type_k = v }
                SelectField { label: "Cache Type V", value: cfg.cache_type_v.clone(),
                    options: vec!["f16","q8_0","turbo2","turbo3","turbo4"],
                    on_change: move |v: String| config.write().cache_type_v = v }
                SelectField { label: "Layer Adaptive", value: cfg.turbo_layer_adaptive.clone(),
                    options: vec!["off","1","5"],
                    on_change: move |v: String| config.write().turbo_layer_adaptive = v }
                SelectField { label: "Flash Attention", value: cfg.flash_attention.clone(),
                    options: vec!["auto","on","off"],
                    on_change: move |v: String| config.write().flash_attention = v }
                div { class: "field",
                    label { "Context Size" }
                    input { r#type: "text", value: "{cfg.context_size}",
                        onchange: move |e: Event<FormData>| config.write().context_size = e.value() }
                }
                div { class: "field",
                    label { "GPU Layers" }
                    input { r#type: "text", value: "{cfg.gpu_layers}",
                        onchange: move |e: Event<FormData>| config.write().gpu_layers = e.value() }
                }
            }

            div { class: "panel",
                h3 { "Server" }
                div { class: "field",
                    label { "Host" }
                    input { r#type: "text", value: "{cfg.host}",
                        onchange: move |e: Event<FormData>| config.write().host = e.value() }
                }
                div { class: "field",
                    label { "Port" }
                    input { r#type: "text", value: "{cfg.port}",
                        onchange: move |e: Event<FormData>| {
                            if let Ok(p) = e.value().parse() { config.write().port = p; }
                        }
                    }
                }
                div { class: "field",
                    label { "Parallel Slots" }
                    input { r#type: "text", value: "{cfg.parallel}",
                        onchange: move |e: Event<FormData>| config.write().parallel = e.value() }
                }
                div { class: "field",
                    label { "Batch Size" }
                    input { r#type: "text", value: "{cfg.batch_size}",
                        onchange: move |e: Event<FormData>| config.write().batch_size = e.value() }
                }
                div { class: "field",
                    label { "Ubatch Size" }
                    input { r#type: "text", value: "{cfg.ubatch_size}",
                        onchange: move |e: Event<FormData>| config.write().ubatch_size = e.value() }
                }
                div { class: "field",
                    label { "Threads" }
                    input { r#type: "text", value: "{cfg.threads}",
                        onchange: move |e: Event<FormData>| config.write().threads = e.value() }
                }
            }
        }
    }
}

#[component]
fn AdvancedTab(config: Signal<LauncherConfig>, cmd_preview: String) -> Element {
    let cfg = config.read().clone();

    rsx! {
        div { class: "panel-row",
            div { class: "panel",
                h3 { "Advanced Options" }
                div { class: "field",
                    label { "API Key" }
                    input { r#type: "password", value: "{cfg.api_key}",
                        onchange: move |e: Event<FormData>| config.write().api_key = e.value() }
                }
                div { class: "field",
                    label { "Log Verbosity" }
                    input { r#type: "text", value: "{cfg.log_verbosity}",
                        onchange: move |e: Event<FormData>| config.write().log_verbosity = e.value() }
                }
                div { class: "field",
                    label { "HTTP Threads" }
                    input { r#type: "text", value: "{cfg.threads_http}",
                        onchange: move |e: Event<FormData>| config.write().threads_http = e.value() }
                }
                div { class: "field",
                    label { "Extra Args" }
                    textarea { value: "{cfg.extra_args}", rows: "3",
                        onchange: move |e: Event<FormData>| config.write().extra_args = e.value() }
                }
                div { class: "field-row",
                    label {
                        input { r#type: "checkbox", checked: cfg.no_mmap,
                            onchange: move |e: Event<FormData>| config.write().no_mmap = e.value() == "true" }
                        " Disable mmap"
                    }
                    label {
                        input { r#type: "checkbox", checked: cfg.disable_web_ui,
                            onchange: move |e: Event<FormData>| config.write().disable_web_ui = e.value() == "true" }
                        " Disable WebUI"
                    }
                }
            }
            div { class: "panel",
                h3 { "Command Preview" }
                pre { class: "cmd-preview", "{cmd_preview}" }
            }
        }
    }
}

#[component]
fn LogsTab(log_lines: Signal<Vec<String>>, status_text: String) -> Element {
    rsx! {
        div { class: "panel",
            h3 { "Server Status: {status_text}" }
            div { class: "log-viewer",
                pre {
                    for line in log_lines.read().iter() {
                        "{line}\n"
                    }
                }
            }
        }
    }
}

#[component]
fn SelectField(
    label: &'static str,
    value: String,
    options: Vec<&'static str>,
    on_change: EventHandler<String>,
) -> Element {
    rsx! {
        div { class: "field",
            label { "{label}" }
            select {
                value: "{value}",
                onchange: move |e: Event<FormData>| on_change.call(e.value()),
                for opt in options.iter() {
                    option {
                        value: "{opt}",
                        selected: *opt == value,
                        "{opt}"
                    }
                }
            }
        }
    }
}
