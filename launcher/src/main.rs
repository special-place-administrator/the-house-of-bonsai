#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod api;
mod config;
mod gguf;
mod huggingface;
mod models;
mod prism;
mod process;
mod resources;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

use dioxus::prelude::*;
use dioxus::prelude::Key;

use config::LauncherConfig;
use models::{ModelCatalog, ModelEntry, format_size};
use process::{ProcessManager, ServerStatus, SharedProcessManager};
use resources::{GpuVendor, SystemResources};

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
    let cfg = LauncherConfig::load();

    // 1. Persisted model directory from config (user chose via Browse)
    if !cfg.model_root.is_empty() {
        let p = PathBuf::from(&cfg.model_root);
        if p.is_dir() { return p; }
    }

    // 2. Derive from existing slot model paths — if a slot has a model,
    //    its parent directory is where models live
    for slot in &cfg.slots {
        if !slot.model_path.is_empty() {
            let p = PathBuf::from(&slot.model_path);
            if let Some(parent) = p.parent() {
                if parent.is_dir() { return parent.to_path_buf(); }
            }
        }
    }

    // 3. models/ next to the launcher exe (portable layout)
    if let Ok(exe_path) = std::env::current_exe() {
        let exe_dir = exe_path.parent().unwrap_or(std::path::Path::new("."));
        let portable_models = exe_dir.join("models");
        if portable_models.is_dir() { return portable_models; }
    }

    // 4. Project-local models/
    let local = repo_root.join("models");
    if local.is_dir() { return local; }

    // 5. User home fallback — create if needed
    let home_models = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("BonsaiLauncher")
        .join("models");
    let _ = std::fs::create_dir_all(&home_models);
    home_models
}

fn detect_llama_root(repo_root: &Path) -> PathBuf {
    // 1. Submodule within repo
    let local = repo_root.join("llama-cpp");
    if local.is_dir() { return local; }
    // 2. system/ folder next to launcher exe (package layout)
    if let Ok(exe_path) = std::env::current_exe() {
        let exe_dir = exe_path.parent().unwrap_or(Path::new("."));
        let system_dir = exe_dir.join("system");
        if system_dir.join("llama-server.exe").exists() { return system_dir; }
    }
    local
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("bonsai_launcher=info")
        .init();

    let _ = std::fs::create_dir_all(LauncherConfig::config_dir());
    let _ = std::fs::create_dir_all(LauncherConfig::log_dir());

    // The ProcessManager creates a Windows Job Object with
    // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE — all child llama-server processes
    // are automatically killed when the launcher exits (any exit path).

    // Kill orphaned llama-servers on Ctrl+C / terminal close
    ctrlc::set_handler(move || {
        process::kill_all_llama_servers();
        std::process::exit(0);
    }).ok();

    dioxus::launch(App);

    // Safety net: kill any surviving llama-server processes after GUI closes
    process::kill_all_llama_servers();
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

    // Prism deployment modal state
    let mut show_prism_modal = use_signal(|| false);
    let mut prism_readiness: Signal<Option<prism::PrismReadiness>> = use_signal(|| None);
    let mut harness_probes: Signal<Vec<prism::HarnessProbe>> = use_signal(Vec::new);
    let mut prism_deploy_log: Signal<Vec<String>> = use_signal(Vec::new);
    let mut prism_deploying = use_signal(|| false);
    let prism_dashboard_port = use_signal(|| 3333u16);

    // Initialize process manager once
    use_effect(move || {
        let root = llama_root.read().clone();
        let new_pm = Arc::new(RwLock::new(ProcessManager::new(&root)));
        process::spawn_health_loop(new_pm.clone());
        process::spawn_log_collector(new_pm.clone());

        // Spawn management API server
        let cfg = LauncherConfig::load();
        let api_port = cfg.api_port;
        let api_secret = if cfg.api_secret.is_empty() {
            None
        } else {
            Some(cfg.api_secret.clone())
        };
        let model_root = detect_model_root(&root);
        let api_state = api::ApiState {
            pm: new_pm.clone(),
            config: Arc::new(RwLock::new(cfg)),
            model_root,
            api_secret,
        };
        tokio::spawn(async move {
            if let Err(e) = api::start_api(api_state, api_port).await {
                tracing::error!("Management API failed: {}", e);
            }
        });

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
        let script = root.join("windows").join("Build-Bonsai.ps1");
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

    let on_deploy_prism = move |_| {
        show_prism_modal.set(true);
        prism_deploy_log.set(Vec::new());
        // Run detection in background
        let root = repo_root.read().clone();
        let cfg = config.read().clone();
        spawn(async move {
            // Detect harnesses
            let probes = prism::detect_all_harnesses();
            harness_probes.set(probes);
            // Check readiness
            let readiness = prism::check_prism_readiness(&cfg, &root).await;
            prism_readiness.set(Some(readiness));
        });
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
                button {
                    class: if *any_running.read() { "btn btn-prism" } else { "btn btn-prism-disabled" },
                    disabled: !*any_running.read(),
                    onclick: on_deploy_prism,
                    "🧠 Deploy Prism"
                }
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
                        LaunchTab { config, model_list, pm, message, cached_resources, shared_root }
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

            // Prism deployment modal
            if *show_prism_modal.read() {
                DeployPrismModal {
                    show_prism_modal,
                    prism_readiness,
                    harness_probes,
                    prism_deploy_log,
                    prism_deploying,
                    prism_dashboard_port,
                    config: config.clone(),
                    repo_root: repo_root.read().clone(),
                    message,
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
    shared_root: Signal<PathBuf>,
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
                    shared_root,
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
    shared_root: Signal<PathBuf>,
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

    // HuggingFace search state
    let mut hf_search_query = use_signal(|| String::new());
    let mut hf_sort_by = use_signal(|| "lastModified".to_string());
    let mut hf_results: Signal<Vec<(String, String, u64)>> = use_signal(Vec::new);
    let mut hf_searching = use_signal(|| false);
    let mut hf_selected: Signal<Option<(String, String, u64)>> = use_signal(|| None);
    let mut download_progress: Signal<Option<(u64, u64)>> = use_signal(|| None);
    let mut downloading = use_signal(|| false);

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
                // Backend selector — only show backends available for detected GPU
                div { class: "field",
                    label { "Backend" }
                    select {
                        value: "{slot.backend}",
                        onchange: move |e: Event<FormData>| {
                            let val = e.value();
                            if let Some(s) = config.write().slots.get_mut(index) {
                                s.backend = val.clone();
                                // When switching to CPU, force gpu_layers to 0;
                                // when switching back to a GPU backend, restore to auto
                                if val == "cpu" {
                                    s.gpu_layers = "0".into();
                                } else if s.gpu_layers == "0" {
                                    s.gpu_layers = "auto".into();
                                }
                            }
                            save();
                        },
                        {
                            let backends = cached_resources.read().available_backends();
                            rsx! {
                                for backend in backends.iter() {
                                    option {
                                        value: "{backend}",
                                        selected: *backend == slot.backend,
                                        "{GpuVendor::backend_label(backend)}"
                                    }
                                }
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

            // HuggingFace search section
            div { class: "hf-search-section",
                div { class: "hf-search-row",
                    input {
                        r#type: "text",
                        placeholder: "Search HuggingFace for GGUF models...",
                        value: "{hf_search_query}",
                        oninput: move |e: Event<FormData>| {
                            hf_search_query.set(e.value());
                        },
                        onkeypress: move |e: Event<KeyboardData>| {
                            if e.key() == Key::Enter && !hf_search_query.read().is_empty() && !*hf_searching.read() {
                                let query = hf_search_query.read().clone();
                                let sort = hf_sort_by.read().clone();
                                hf_searching.set(true);
                                hf_results.set(Vec::new());
                                hf_selected.set(None);
                                spawn(async move {
                                    match huggingface::search_models(&query, &sort).await {
                                        Ok(models) => {
                                            let mut all_files = Vec::new();
                                            for model in models.iter().take(5) {
                                                if let Ok(files) = huggingface::list_gguf_files(&model.model_id).await {
                                                    for f in files {
                                                        all_files.push((f.repo_id, f.filename, f.size_bytes));
                                                    }
                                                }
                                            }
                                            hf_results.set(all_files);
                                        }
                                        Err(e) => {
                                            message.set(Some((format!("HF search error: {}", e), true)));
                                        }
                                    }
                                    hf_searching.set(false);
                                });
                            }
                        },
                    }
                    select {
                        class: "hf-sort-select",
                        value: "{hf_sort_by}",
                        onchange: move |e: Event<FormData>| { hf_sort_by.set(e.value()); },
                        option { value: "lastModified", "Newest" }
                        option { value: "downloads", "Popular" }
                        option { value: "alphabetical", "A-Z" }
                    }
                    button {
                        class: "btn-search",
                        disabled: *hf_searching.read() || hf_search_query.read().is_empty(),
                        onclick: move |_| {
                            let query = hf_search_query.read().clone();
                            let sort = hf_sort_by.read().clone();
                            if query.is_empty() || *hf_searching.read() { return; }
                            hf_searching.set(true);
                            hf_results.set(Vec::new());
                            hf_selected.set(None);
                            spawn(async move {
                                match huggingface::search_models(&query, &sort).await {
                                    Ok(models) => {
                                        let mut all_files = Vec::new();
                                        for model in models.iter().take(5) {
                                            if let Ok(files) = huggingface::list_gguf_files(&model.model_id).await {
                                                for f in files {
                                                    all_files.push((f.repo_id, f.filename, f.size_bytes));
                                                }
                                            }
                                        }
                                        hf_results.set(all_files);
                                    }
                                    Err(e) => {
                                        message.set(Some((format!("HF search error: {}", e), true)));
                                    }
                                }
                                hf_searching.set(false);
                            });
                        },
                        if *hf_searching.read() { "Searching..." } else { "Search HF" }
                    }
                }

                // Results dropdown
                if !hf_results.read().is_empty() {
                    select {
                        class: "hf-results-select",
                        onchange: move |e: Event<FormData>| {
                            let val = e.value();
                            if val.is_empty() {
                                hf_selected.set(None);
                            } else {
                                let results = hf_results.read();
                                if let Some(idx) = val.parse::<usize>().ok() {
                                    if let Some(item) = results.get(idx) {
                                        hf_selected.set(Some(item.clone()));
                                    }
                                }
                            }
                        },
                        option { value: "", "-- Select a GGUF file --" }
                        for (i, (repo, fname, size)) in hf_results.read().iter().enumerate() {
                            option {
                                value: "{i}",
                                "{repo}/{fname} ({format_size(*size)})"
                            }
                        }
                    }
                }

                // Download button and progress
                if let Some((repo, fname, _size)) = hf_selected.read().as_ref() {
                    if !*downloading.read() {
                        {
                            let repo = repo.clone();
                            let fname = fname.clone();
                            rsx! {
                                button {
                                    class: "btn-download",
                                    onclick: move |_| {
                                        let root = shared_root.read().clone();
                                        let url = format!("https://huggingface.co/{}/resolve/main/{}", repo, fname);
                                        let dest = root.join(&fname);
                                        downloading.set(true);
                                        download_progress.set(Some((0, 0)));
                                        let fname_clone = fname.clone();
                                        spawn(async move {
                                            let (tx, mut rx) = tokio::sync::watch::channel((0u64, 0u64));
                                            // Spawn progress poller
                                            let _progress_handle = spawn(async move {
                                                loop {
                                                    tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                                                    let val = *rx.borrow_and_update();
                                                    download_progress.set(Some(val));
                                                    if rx.has_changed().is_err() { break; }
                                                }
                                            });
                                            match huggingface::download_model(&url, &dest, tx).await {
                                                Ok(()) => {
                                                    // Refresh model list
                                                    let cfg = config.read().clone();
                                                    let entries = crate::models::ModelCatalog::scan(&root, &cfg.recent_models).entries;
                                                    model_list.set(entries);
                                                    // Set this slot to the downloaded model
                                                    let dest_str = dest.display().to_string();
                                                    {
                                                        let mut c = config.write();
                                                        if let Some(s) = c.slots.get_mut(index) {
                                                            s.model_path = dest_str.clone();
                                                        }
                                                        c.add_recent_model(&dest_str);
                                                        let _ = c.save();
                                                    }
                                                    // Auto-tune the slot
                                                    let meta = crate::gguf::ModelMetadata::from_file(&dest_str);
                                                    let res = cached_resources.read().clone();
                                                    {
                                                        let mut c = config.write();
                                                        if let Some(s) = c.slots.get_mut(index) {
                                                            if let Some(ref m) = meta {
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
                                                                }
                                                            }
                                                            let (threads, http_threads) = res.auto_tune(s, &dest_str, meta.as_ref());
                                                            c.threads = threads.to_string();
                                                            c.threads_http = http_threads.to_string();
                                                        }
                                                        let _ = c.save();
                                                    }
                                                    message.set(Some((format!("Downloaded: {}", fname_clone), false)));
                                                }
                                                Err(e) => {
                                                    message.set(Some((format!("Download failed: {}", e), true)));
                                                }
                                            }
                                            downloading.set(false);
                                            download_progress.set(None);
                                            hf_selected.set(None);
                                        });
                                    },
                                    "Download to models/"
                                }
                            }
                        }
                    }
                }

                // Progress bar
                if let Some((downloaded, total)) = download_progress.read().as_ref() {
                    {
                        let pct = if *total > 0 { (*downloaded as f64 / *total as f64 * 100.0) as u32 } else { 0 };
                        let downloaded_display = format_size(*downloaded);
                        let total_display = if *total > 0 { format_size(*total) } else { "?".to_string() };
                        rsx! {
                            div { class: "progress-bar-container",
                                div {
                                    class: "progress-bar-fill",
                                    style: "width: {pct}%;",
                                }
                            }
                            div { class: "progress-text",
                                "{pct}% ({downloaded_display} / {total_display})"
                            }
                        }
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


// ---------------------------------------------------------------------------
// Deploy Prism MCP Modal
// ---------------------------------------------------------------------------

#[component]
fn DeployPrismModal(
    mut show_prism_modal: Signal<bool>,
    mut prism_readiness: Signal<Option<prism::PrismReadiness>>,
    mut harness_probes: Signal<Vec<prism::HarnessProbe>>,
    mut prism_deploy_log: Signal<Vec<String>>,
    mut prism_deploying: Signal<bool>,
    prism_dashboard_port: Signal<u16>,
    config: Signal<LauncherConfig>,
    repo_root: PathBuf,
    mut message: Signal<Option<(String, bool)>>,
) -> Element {
    let readiness = prism_readiness.read().clone();
    let probes = harness_probes.read().clone();
    let deploying = *prism_deploying.read();
    let log = prism_deploy_log.read().clone();
    let dashboard_port = *prism_dashboard_port.read();

    let selected_count = probes.iter().filter(|p| p.selected).count();
    let all_ready = readiness.as_ref().map(|r| r.all_ok()).unwrap_or(false);
    let can_finish = all_ready && selected_count > 0 && !deploying;

    // Build Prism handler
    let root_for_build = repo_root.clone();
    let on_build_prism = move |_| {
        let root = root_for_build.clone();
        let root2 = root.clone();
        let cfg = config.read().clone();
        prism_deploying.set(true);
        prism_deploy_log.with_mut(|l| l.push("Building Prism MCP...".into()));
        spawn(async move {
            let result = tokio::task::spawn_blocking(move || prism::build_prism(&root))
                .await
                .unwrap_or_else(|e| Err(format!("Build task panicked: {}", e)));
            match result {
                Ok(msg) => {
                    prism_deploy_log.with_mut(|l| l.push(format!("✓ {}", msg)));
                }
                Err(err) => {
                    prism_deploy_log.with_mut(|l| l.push(format!("✗ {}", err)));
                }
            }
            // Re-check readiness
            let cfg = config.read().clone();
            let r = prism::check_prism_readiness(&cfg, &root2).await;
            prism_readiness.set(Some(r));
            prism_deploying.set(false);
        });
    };

    // FINISH handler
    let root_for_deploy = repo_root.clone();
    let dp = dashboard_port;
    let on_finish = move |_| {
        let probes_snapshot = harness_probes.read().clone();
        let cfg = config.read().clone();
        let root = root_for_deploy.clone();
        let dp = dp;
        prism_deploying.set(true);
        spawn(async move {
            // Get readiness for building the entry
            let readiness = prism::check_prism_readiness(&cfg, &root).await;
            let prism_entry = prism::build_mcp_entry(&readiness, &cfg, dp);
            let bonsai_entry = prism::build_bonsai_entry(
                &readiness,
                &root,
                cfg.api_port,
                &cfg.host,
            );

            let entries: Vec<(&str, &serde_json::Value)> = vec![
                ("prism-mcp", &prism_entry),
                ("bonsai-mcp", &bonsai_entry),
            ];

            for probe in &probes_snapshot {
                if !probe.selected {
                    continue;
                }
                let label = probe.harness.label();
                match prism::deploy_to_harness(probe, &entries) {
                    Ok(msgs) => {
                        for msg in msgs {
                            prism_deploy_log.with_mut(|l| l.push(format!("✓ {} — {}", label, msg)));
                        }
                    }
                    Err(err) => {
                        prism_deploy_log.with_mut(|l| l.push(format!("✗ {} — {}", label, err)));
                    }
                }
            }
            prism_deploy_log.with_mut(|l| {
                l.push(String::new());
                l.push("Deployment complete! Restart your AI tools to activate Prism + Bonsai MCP.".into());
            });
            prism_deploying.set(false);
        });
    };

    // Open dashboard handler
    let on_open_dashboard = move |_| {
        let host = config.read().host.clone();
        let _ = open::that(format!("http://{}:{}", host, dashboard_port));
    };

    rsx! {
        // Modal overlay
        div {
            class: "modal-overlay",
            onclick: move |_| show_prism_modal.set(false),

            div {
                class: "modal",
                onclick: move |e| e.stop_propagation(),

                // Header
                div { class: "modal-header",
                    span { class: "modal-title", "🧠 Deploy Prism MCP" }
                    button {
                        class: "modal-close",
                        onclick: move |_| show_prism_modal.set(false),
                        "✕"
                    }
                }

                // Pre-flight checks
                div { class: "modal-section",
                    div { class: "modal-section-title", "Pre-flight Checks" }

                    if let Some(r) = readiness.as_ref() {
                        PreflightRow { label: "Text endpoint", ok: r.text_ok }
                        PreflightRow { label: "Embedding endpoint", ok: r.embedding_ok }
                        PreflightRow { label: "Embedding dims (768)", ok: r.embedding_dims_ok }
                        PreflightRow { label: "Node.js installed", ok: r.node_available }
                        PreflightRow { label: "npm installed", ok: r.npm_available }

                        div { class: "preflight-row",
                            span {
                                class: if r.prism_built { "preflight-ok" } else { "preflight-fail" },
                                if r.prism_built { "●" } else { "✗" }
                            }
                            span { "Prism built" }
                            if !r.prism_built {
                                button {
                                    class: "btn-build",
                                    disabled: deploying,
                                    onclick: on_build_prism,
                                    if deploying { "Building..." } else { "Build" }
                                }
                            }
                        }
                    } else {
                        div { class: "preflight-row", "Checking..." }
                    }
                }

                // Harness selection
                div { class: "modal-section",
                    div { class: "modal-section-title", "Deploy to Harnesses" }

                    for (i, probe) in probes.iter().enumerate() {
                        div { class: "harness-row",
                            input {
                                r#type: "checkbox",
                                checked: probe.selected && probe.is_available(),
                                disabled: !probe.is_available(),
                                onchange: {
                                    let i = i;
                                    move |e: Event<FormData>| {
                                        let checked = e.value() == "true";
                                        harness_probes.with_mut(|probes| {
                                            if let Some(p) = probes.get_mut(i) {
                                                p.selected = checked;
                                            }
                                        });
                                    }
                                },
                            }
                            span { class: "harness-label", "{probe.harness.label()}" }
                            match &probe.status {
                                prism::HarnessStatus::Checking => rsx! {
                                    span { class: "harness-checking", "..." }
                                },
                                prism::HarnessStatus::Found(_) => rsx! {
                                    span { class: "harness-found", "● Found" }
                                },
                                prism::HarnessStatus::ManualPath(_) => rsx! {
                                    span { class: "harness-found", "● Manual" }
                                },
                                prism::HarnessStatus::NotFound(reason) => rsx! {
                                    span { class: "harness-missing", "✗ {reason}" }
                                    button {
                                        class: "btn-browse-small",
                                        onclick: {
                                            let i = i;
                                            move |_| {
                                                if let Some(path) = rfd::FileDialog::new()
                                                    .set_title("Select MCP config file")
                                                    .add_filter("JSON", &["json"])
                                                    .pick_file()
                                                {
                                                    harness_probes.with_mut(|probes| {
                                                        if let Some(p) = probes.get_mut(i) {
                                                            p.status = prism::HarnessStatus::ManualPath(path);
                                                            p.selected = true;
                                                        }
                                                    });
                                                }
                                            }
                                        },
                                        "Browse..."
                                    }
                                },
                            }
                        }
                    }
                }

                // FINISH button
                div { class: "modal-section",
                    button {
                        class: if can_finish { "btn-finish" } else { "btn-finish btn-finish-disabled" },
                        disabled: !can_finish,
                        onclick: on_finish,
                        if deploying {
                            "Deploying..."
                        } else {
                            "FINISH — Deploy to {selected_count} harness(es)"
                        }
                    }
                }

                // Deploy log
                if !log.is_empty() {
                    div { class: "deploy-log",
                        for line in log.iter() {
                            div { class: "deploy-log-line", "{line}" }
                        }
                    }
                }

                // Open dashboard button (shown after deployment)
                if log.iter().any(|l| l.contains("Deployment complete")) {
                    div { class: "modal-section",
                        button {
                            class: "btn-dashboard",
                            onclick: on_open_dashboard,
                            "🏛 Open Mind Palace Dashboard"
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn PreflightRow(label: &'static str, ok: bool) -> Element {
    rsx! {
        div { class: "preflight-row",
            span {
                class: if ok { "preflight-ok" } else { "preflight-fail" },
                if ok { "●" } else { "✗" }
            }
            span { "{label}" }
        }
    }
}
