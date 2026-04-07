#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod config;
mod gguf;
mod models;
mod process;
mod resources;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

use dioxus::prelude::*;

use config::LauncherConfig;
use models::{ModelCatalog, ModelEntry, format_size};
use process::{ProcessManager, ServerStatus, SharedProcessManager};
use resources::SystemResources;

fn pick_model_folder() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .set_title("Select folder containing GGUF models")
        .pick_folder()
}

fn detect_repo_root() -> PathBuf {
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
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn detect_model_root(repo_root: &Path) -> PathBuf {
    let local = repo_root.join("models");
    if local.is_dir() { return local; }
    let shared = PathBuf::from(r"C:\AI_STUFF\LLM_MODEL");
    if shared.is_dir() { return shared; }
    local
}

fn detect_llama_root(repo_root: &Path) -> PathBuf {
    let local = repo_root.join("llama-cpp");
    if local.is_dir() { return local; }
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
    let mut shared_root = use_signal(|| {
        let root = detect_repo_root();
        detect_model_root(&root)
    });

    let mut model_list = use_signal(|| {
        let cfg = LauncherConfig::load();
        let root = detect_repo_root();
        let model_root = detect_model_root(&root);
        ModelCatalog::scan(&model_root, &cfg.recent_models).entries
    });

    // Cache system resources — detect once, reuse everywhere
    let cached_resources = use_signal(|| SystemResources::detect());

    let mut pm: Signal<Option<SharedProcessManager>> = use_signal(|| None);
    let mut status_text = use_signal(|| "Stopped".to_string());
    let mut status_color = use_signal(|| "#888".to_string());
    let mut log_lines: Signal<Vec<String>> = use_signal(Vec::new);
    let mut any_running = use_signal(|| false);
    let mut active_tab = use_signal(|| "launch".to_string());
    let mut message: Signal<Option<(String, bool)>> = use_signal(|| None);

    // Initialize process manager once
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
                let running = mgr.is_any_running();
                let (text, color) = mgr.aggregate_status();
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

                any_running.set(running);
                status_text.set(text);
                status_color.set(color);
                log_lines.set(lines);
            }
        }
    });

    // Toggle start/stop ALL slots
    let on_toggle_server = move |_| {
        let running = *any_running.read();
        spawn(async move {
            let pm_clone = pm.read().clone();
            if let Some(pm_ref) = pm_clone.as_ref() {
                let mut mgr = pm_ref.write().await;
                if running {
                    mgr.stop_all().await;
                } else {
                    let cfg = config.read().clone();
                    let _ = cfg.save();
                    mgr.ensure_slots(cfg.slots.len());
                    let results = mgr.start_all(&cfg).await;
                    let errors: Vec<_> = results.into_iter()
                        .filter_map(|r| r.err())
                        .collect();
                    if !errors.is_empty() {
                        message.set(Some((errors.join("\n"), true)));
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

    let on_browse_folder = move |_| {
        if let Some(folder) = pick_model_folder() {
            shared_root.set(folder.clone());
            let cfg = config.read().clone();
            let entries = ModelCatalog::scan(&folder, &cfg.recent_models).entries;
            let count = entries.len();
            model_list.set(entries);
            config.write().model_root = folder.display().to_string();
            let _ = config.read().save();
            message.set(Some((format!("Loaded {} models from {}", count, folder.display()), false)));
        }
    };

    let on_open_chat = move |_| {
        let cfg = config.read();
        if let Some(slot) = cfg.slots.first() {
            let _ = open::that(format!("http://{}:{}/", cfg.host, slot.port));
        }
    };

    let on_auto_tune = move |_| {
        let res = cached_resources.read().clone();
        let mut cfg = config.write();
        let mut last_threads = (0u32, 0u32);
        for i in 0..cfg.slots.len() {
            let model_path = cfg.slots[i].model_path.clone();
            let meta = gguf::ModelMetadata::from_file(&model_path);
            last_threads = res.auto_tune(&mut cfg.slots[i], &model_path, meta.as_ref());
        }
        cfg.threads = last_threads.0.to_string();
        cfg.threads_http = last_threads.1.to_string();
        let _ = cfg.save();
        drop(cfg);
        message.set(Some((format!("Auto-tuned all slots & saved!\n{}", res.summary()), false)));
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

    rsx! {
        style { {include_str!("../assets/style.css")} }

        div { class: "app",
            // Header
            div {
                class: if *any_running.read() { "header header-running" } else { "header header-stopped" },
                if *any_running.read() {
                    button { class: "btn btn-stop", onclick: on_toggle_server, "■ Stop All" }
                } else {
                    button { class: "btn btn-start", onclick: on_toggle_server, "▶ Start All" }
                }
                button { class: "btn", onclick: on_open_chat, "💬 Chat" }
                button { class: "btn", onclick: on_build, "🔧 Build" }
                button { class: "btn", onclick: on_save, "💾 Save" }
                button { class: "btn", onclick: on_refresh_models, "🔄 Models" }
                button { class: "btn", onclick: on_browse_folder, "📂 Browse" }
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
                        LaunchTab { config, model_list, pm, message, cached_resources }
                    },
                    "advanced" => rsx! {
                        AdvancedTab { config, repo_root: repo_root.read().clone() }
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

// ---------------------------------------------------------------------------
// Launch Tab — scrollable list of model cards + add button
// ---------------------------------------------------------------------------

#[component]
fn LaunchTab(
    config: Signal<LauncherConfig>,
    model_list: Signal<Vec<ModelEntry>>,
    pm: Signal<Option<SharedProcessManager>>,
    message: Signal<Option<(String, bool)>>,
    cached_resources: Signal<SystemResources>,
) -> Element {
    let slot_count = config.read().slots.len();

    rsx! {
        div { class: "model-cards-container",
            for i in 0..slot_count {
                ModelCard {
                    config,
                    model_list,
                    pm,
                    message,
                    cached_resources,
                    index: i,
                    can_delete: slot_count > 1,
                }
            }

            button {
                class: "btn btn-add-model",
                onclick: move |_| {
                    config.write().add_slot();
                    let _ = config.read().save();
                },
                "+ Add Model"
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Single model card
// ---------------------------------------------------------------------------

#[component]
fn ModelCard(
    config: Signal<LauncherConfig>,
    model_list: Signal<Vec<ModelEntry>>,
    pm: Signal<Option<SharedProcessManager>>,
    message: Signal<Option<(String, bool)>>,
    cached_resources: Signal<SystemResources>,
    index: usize,
    can_delete: bool,
) -> Element {
    // Read slot data
    let cfg = config.read();
    let slot = match cfg.slots.get(index) {
        Some(s) => s.clone(),
        None => return rsx! { div { "Slot not found" } },
    };
    drop(cfg);

    // Read per-slot process status
    let slot_status = use_memo(move || {
        let pm_guard = pm.read();
        if let Some(pm_ref) = pm_guard.as_ref() {
            if let Ok(mgr) = pm_ref.try_read() {
                if let Some(ps) = mgr.slots.get(index) {
                    return ps.status.clone();
                }
            }
        }
        ServerStatus::Stopped
    });

    let status_dot_color = match &*slot_status.read() {
        ServerStatus::Stopped => "#888",
        ServerStatus::Starting => "#e6a817",
        ServerStatus::Running => "#4caf50",
        ServerStatus::Unhealthy => "#ff9800",
        ServerStatus::Crashed(_) => "#f44336",
    };

    let status_label = format!("{}", &*slot_status.read());

    // Auto-save helper
    let save = move || { let _ = config.read().save(); };

    // Read GGUF metadata for capability icons
    let caps_icons: Vec<(&str, &str)> = if !slot.model_path.is_empty() {
        gguf::ModelMetadata::from_file(&slot.model_path)
            .map(|m| m.capabilities.icons())
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    rsx! {
        div { class: "model-card",
            // Card header
            div { class: "model-card-header",
                span { class: "model-card-title", "Model Slot {index + 1}" }
                if !caps_icons.is_empty() {
                    span { class: "capability-icons",
                        for (icon, label) in caps_icons.iter() {
                            span { class: "cap-badge", title: "{label}", "{icon}" }
                        }
                    }
                }
                div { class: "model-card-header-right",
                    span { class: "status-dot", style: "color: {status_dot_color};", title: "{status_label}",
                        "●"
                    }
                    if can_delete {
                        button {
                            class: "btn-delete-slot",
                            title: "Remove this slot",
                            onclick: move |_| {
                                config.write().remove_slot(index);
                                let _ = config.read().save();
                            },
                            "✕"
                        }
                    }
                }
            }

            // Card body — dense grid
            div { class: "model-card-grid",
                // Row 1: Model + Alias + Port
                div { class: "field field-wide",
                    label { "Model" }
                    select {
                        value: "{slot.model_path}",
                        onchange: move |e: Event<FormData>| {
                            let val = e.value();
                            let mut c = config.write();
                            if let Some(s) = c.slots.get_mut(index) {
                                s.model_path = val.clone();
                                // Read GGUF metadata for auto-tune and capability-based config
                                let meta = gguf::ModelMetadata::from_file(&val);

                                // Auto-configure based on detected capabilities
                                if let Some(ref m) = meta {
                                    // Set alias from model name if available
                                    if let Some(ref model_name) = m.name {
                                        s.alias = model_name.clone();
                                    }
                                    if m.capabilities.embedding {
                                        s.embedding_mode = true;
                                        s.cache_type_k = "f16".into();
                                        s.cache_type_v = "f16".into();
                                        s.flash_attention = "off".into();
                                        s.turbo_layer_adaptive = "off".into();
                                        s.parallel = "4".into();
                                    } else {
                                        s.embedding_mode = false;
                                    }
                                }

                                // Auto-tune this slot (uses cached resources)
                                let res = cached_resources.read().clone();
                                let (threads, http_threads) = res.auto_tune(s, &val, meta.as_ref());
                                c.threads = threads.to_string();
                                c.threads_http = http_threads.to_string();
                            }
                            let _ = c.save();
                        },
                        option { value: "", "-- Select a model --" }
                        for entry in model_list.read().iter() {
                            option {
                                value: "{entry.path.display()}",
                                selected: entry.path.to_string_lossy() == slot.model_path,
                                "{entry.display_name} ({format_size(entry.size_bytes)})"
                            }
                        }
                    }
                }
                div { class: "field",
                    label { "Alias" }
                    input {
                        r#type: "text",
                        value: "{slot.alias}",
                        onchange: move |e: Event<FormData>| {
                            if let Some(s) = config.write().slots.get_mut(index) { s.alias = e.value(); }
                            save();
                        },
                    }
                }
                div { class: "field",
                    label { "Port" }
                    input {
                        r#type: "text",
                        value: "{slot.port}",
                        onchange: move |e: Event<FormData>| {
                            if let Ok(p) = e.value().parse::<u16>() {
                                if let Some(s) = config.write().slots.get_mut(index) { s.port = p; }
                                save();
                            }
                        },
                    }
                }

                // Row 2: Context + Cache K + Cache V
                div { class: "field",
                    label { "Context" }
                    input {
                        r#type: "text",
                        value: "{slot.context_size}",
                        onchange: move |e: Event<FormData>| {
                            if let Some(s) = config.write().slots.get_mut(index) { s.context_size = e.value(); }
                            save();
                        },
                    }
                }
                SlotSelectField {
                    label: "Cache K",
                    value: slot.cache_type_k.clone(),
                    options: vec!["f16","q8_0","turbo2","turbo3","turbo4"],
                    on_change: move |v: String| {
                        if let Some(s) = config.write().slots.get_mut(index) { s.cache_type_k = v; }
                        save();
                    },
                }
                SlotSelectField {
                    label: "Cache V",
                    value: slot.cache_type_v.clone(),
                    options: vec!["f16","q8_0","turbo2","turbo3","turbo4"],
                    on_change: move |v: String| {
                        if let Some(s) = config.write().slots.get_mut(index) { s.cache_type_v = v; }
                        save();
                    },
                }

                // Row 3: FA + GPU Layers + Parallel Slots
                SlotSelectField {
                    label: "FA",
                    value: slot.flash_attention.clone(),
                    options: vec!["auto","on","off"],
                    on_change: move |v: String| {
                        if let Some(s) = config.write().slots.get_mut(index) { s.flash_attention = v; }
                        save();
                    },
                }
                div { class: "field",
                    label { "GPU Layers" }
                    input {
                        r#type: "text",
                        value: "{slot.gpu_layers}",
                        onchange: move |e: Event<FormData>| {
                            if let Some(s) = config.write().slots.get_mut(index) { s.gpu_layers = e.value(); }
                            save();
                        },
                    }
                }
                div { class: "field",
                    label { "Slots" }
                    input {
                        r#type: "text",
                        value: "{slot.parallel}",
                        onchange: move |e: Event<FormData>| {
                            if let Some(s) = config.write().slots.get_mut(index) { s.parallel = e.value(); }
                            save();
                        },
                    }
                }

                // Row 4: Batch + Ubatch + Embedding + Layer Adaptive
                div { class: "field",
                    label { "Batch" }
                    input {
                        r#type: "text",
                        value: "{slot.batch_size}",
                        onchange: move |e: Event<FormData>| {
                            if let Some(s) = config.write().slots.get_mut(index) { s.batch_size = e.value(); }
                            save();
                        },
                    }
                }
                div { class: "field",
                    label { "Ubatch" }
                    input {
                        r#type: "text",
                        value: "{slot.ubatch_size}",
                        onchange: move |e: Event<FormData>| {
                            if let Some(s) = config.write().slots.get_mut(index) { s.ubatch_size = e.value(); }
                            save();
                        },
                    }
                }
                SlotSelectField {
                    label: "Layer Adapt.",
                    value: slot.turbo_layer_adaptive.clone(),
                    options: vec!["off","1","5"],
                    on_change: move |v: String| {
                        if let Some(s) = config.write().slots.get_mut(index) { s.turbo_layer_adaptive = v; }
                        save();
                    },
                }

                // Embedding checkbox
                div { class: "field",
                    label { " " }
                    label { class: "checkbox-label",
                        input {
                            r#type: "checkbox",
                            checked: slot.embedding_mode,
                            onchange: move |e: Event<FormData>| {
                                if let Some(s) = config.write().slots.get_mut(index) {
                                    s.embedding_mode = e.value() == "true";
                                }
                                save();
                            },
                        }
                        " Embedding"
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Advanced Tab
// ---------------------------------------------------------------------------

#[component]
fn AdvancedTab(config: Signal<LauncherConfig>, repo_root: PathBuf) -> Element {
    let cfg = config.read().clone();
    let exe = repo_root.join("build").join("bin").join("llama-server.exe");

    // Build previews for all slots
    let previews: Vec<String> = cfg.slots.iter().enumerate().map(|(i, slot)| {
        format!("# Slot {}\n{}", i + 1, slot.command_preview(&exe, &cfg))
    }).collect();
    let cmd_preview = previews.join("\n\n");

    rsx! {
        div { class: "panel-row",
            div { class: "panel",
                h3 { "Shared Options" }
                div { class: "field",
                    label { "Host" }
                    input { r#type: "text", value: "{cfg.host}",
                        onchange: move |e: Event<FormData>| {
                            config.write().host = e.value();
                            let _ = config.read().save();
                        }
                    }
                }
                div { class: "field",
                    label { "Threads" }
                    input { r#type: "text", value: "{cfg.threads}",
                        onchange: move |e: Event<FormData>| {
                            config.write().threads = e.value();
                            let _ = config.read().save();
                        }
                    }
                }
                div { class: "field",
                    label { "HTTP Threads" }
                    input { r#type: "text", value: "{cfg.threads_http}",
                        onchange: move |e: Event<FormData>| {
                            config.write().threads_http = e.value();
                            let _ = config.read().save();
                        }
                    }
                }
                div { class: "field",
                    label { "API Key" }
                    input { r#type: "password", value: "{cfg.api_key}",
                        onchange: move |e: Event<FormData>| {
                            config.write().api_key = e.value();
                            let _ = config.read().save();
                        }
                    }
                }
                div { class: "field",
                    label { "Log Verbosity" }
                    input { r#type: "text", value: "{cfg.log_verbosity}",
                        onchange: move |e: Event<FormData>| {
                            config.write().log_verbosity = e.value();
                            let _ = config.read().save();
                        }
                    }
                }
                div { class: "field",
                    label { "Extra Args" }
                    textarea { value: "{cfg.extra_args}", rows: "3",
                        onchange: move |e: Event<FormData>| {
                            config.write().extra_args = e.value();
                            let _ = config.read().save();
                        }
                    }
                }
                div { class: "field-row",
                    label {
                        input { r#type: "checkbox", checked: cfg.no_mmap,
                            onchange: move |e: Event<FormData>| {
                                config.write().no_mmap = e.value() == "true";
                                let _ = config.read().save();
                            }
                        }
                        " Disable mmap"
                    }
                    label {
                        input { r#type: "checkbox", checked: cfg.disable_web_ui,
                            onchange: move |e: Event<FormData>| {
                                config.write().disable_web_ui = e.value() == "true";
                                let _ = config.read().save();
                            }
                        }
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

// ---------------------------------------------------------------------------
// Logs Tab
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Reusable select field for slot settings
// ---------------------------------------------------------------------------

#[component]
fn SlotSelectField(
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
