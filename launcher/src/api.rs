use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::Json,
    routing::{get, post},
    Router,
};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;

use crate::config::LauncherConfig;
use crate::models::ModelCatalog;
use crate::process::SharedProcessManager;
use crate::resources::SystemResources;

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ApiState {
    pub pm: SharedProcessManager,
    pub config: Arc<RwLock<LauncherConfig>>,
    pub model_root: PathBuf,
    pub api_secret: Option<String>,
}

// ---------------------------------------------------------------------------
// Auth middleware helper
// ---------------------------------------------------------------------------

fn check_auth(state: &ApiState, headers: &HeaderMap) -> Result<(), (StatusCode, Json<Value>)> {
    if let Some(ref secret) = state.api_secret {
        if secret.is_empty() {
            return Ok(());
        }
        let provided = headers
            .get("x-bonsai-secret")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if provided != secret {
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(json!({ "ok": false, "error": "Invalid or missing X-Bonsai-Secret header" })),
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

async fn health() -> Json<Value> {
    Json(json!({ "ok": true }))
}

async fn status_all(State(state): State<ApiState>) -> Json<Value> {
    let pm = state.pm.read().await;
    let mut slots = Vec::new();
    for (i, ps) in pm.slots.iter().enumerate() {
        let status_str = format!("{}", ps.status);
        let mut slot = json!({
            "index": i,
            "status": status_str,
        });
        if let Some(ref rt) = ps.runtime {
            slot["runtime"] = json!({
                "pid": rt.pid,
                "started_at": rt.started_at.to_rfc3339(),
                "model_path": rt.model_path,
                "host": rt.host,
                "port": rt.port,
            });
        }
        slots.push(slot);
    }

    // Include alias info from config
    let cfg = state.config.read().await;
    for (i, slot) in slots.iter_mut().enumerate() {
        if let Some(sc) = cfg.slots.get(i) {
            slot["alias"] = json!(sc.alias);
            slot["model_path"] = json!(sc.model_path);
            slot["port"] = json!(sc.port);
            slot["embedding_mode"] = json!(sc.embedding_mode);
        }
    }

    Json(json!({ "ok": true, "data": { "slots": slots } }))
}

async fn status_one(
    State(state): State<ApiState>,
    Path(index): Path<usize>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let pm = state.pm.read().await;
    let ps = pm.slots.get(index).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(json!({ "ok": false, "error": format!("Slot {} does not exist", index) })),
        )
    })?;

    let status_str = format!("{}", ps.status);
    let mut slot = json!({
        "index": index,
        "status": status_str,
    });
    if let Some(ref rt) = ps.runtime {
        slot["runtime"] = json!({
            "pid": rt.pid,
            "started_at": rt.started_at.to_rfc3339(),
            "model_path": rt.model_path,
            "host": rt.host,
            "port": rt.port,
        });
    }

    let cfg = state.config.read().await;
    if let Some(sc) = cfg.slots.get(index) {
        slot["alias"] = json!(sc.alias);
        slot["model_path"] = json!(sc.model_path);
        slot["port"] = json!(sc.port);
        slot["embedding_mode"] = json!(sc.embedding_mode);
    }

    Ok(Json(json!({ "ok": true, "data": slot })))
}

async fn start_slot(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path(index): Path<usize>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    check_auth(&state, &headers)?;

    let cfg = state.config.read().await;
    let slot_config = cfg.slots.get(index).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(json!({ "ok": false, "error": format!("Slot {} not configured", index) })),
        )
    })?;

    let mut pm = state.pm.write().await;
    pm.start_slot(index, slot_config, &cfg).await.map_err(|e| {
        (
            StatusCode::CONFLICT,
            Json(json!({ "ok": false, "error": e })),
        )
    })?;

    Ok(Json(json!({ "ok": true, "data": { "message": format!("Slot {} starting", index) } })))
}

async fn stop_slot(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path(index): Path<usize>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    check_auth(&state, &headers)?;

    let mut pm = state.pm.write().await;
    pm.stop_slot(index).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "ok": false, "error": e })),
        )
    })?;

    Ok(Json(json!({ "ok": true, "data": { "message": format!("Slot {} stopped", index) } })))
}

async fn start_all_slots(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    check_auth(&state, &headers)?;

    let cfg = state.config.read().await;
    let mut pm = state.pm.write().await;
    let results = pm.start_all(&cfg).await;

    let mut messages = Vec::new();
    let mut errors = Vec::new();
    for (i, r) in results.into_iter().enumerate() {
        match r {
            Ok(()) => messages.push(format!("Slot {} started", i)),
            Err(e) => errors.push(format!("Slot {}: {}", i, e)),
        }
    }

    if errors.is_empty() {
        Ok(Json(json!({ "ok": true, "data": { "started": messages } })))
    } else {
        Ok(Json(json!({ "ok": false, "data": { "started": messages, "errors": errors } })))
    }
}

async fn stop_all_slots(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    check_auth(&state, &headers)?;

    let mut pm = state.pm.write().await;
    pm.stop_all().await;

    Ok(Json(json!({ "ok": true, "data": { "message": "All slots stopped" } })))
}

async fn list_models(State(state): State<ApiState>) -> Json<Value> {
    let cfg = state.config.read().await;
    let catalog = ModelCatalog::scan(&state.model_root, &cfg.recent_models);
    let entries: Vec<Value> = catalog
        .entries
        .iter()
        .map(|e| {
            json!({
                "name": e.display_name,
                "path": e.path.to_string_lossy(),
                "source": format!("{:?}", e.source),
                "size_bytes": e.size_bytes,
                "size_human": crate::models::format_size(e.size_bytes),
            })
        })
        .collect();

    Json(json!({ "ok": true, "data": { "models": entries } }))
}

async fn get_config(State(state): State<ApiState>) -> Json<Value> {
    let cfg = state.config.read().await;
    // Serialize config — omit api_secret from response
    let slots: Vec<Value> = cfg
        .slots
        .iter()
        .enumerate()
        .map(|(i, s)| {
            json!({
                "index": i,
                "alias": s.alias,
                "model_path": s.model_path,
                "port": s.port,
                "embedding_mode": s.embedding_mode,
                "context_size": s.context_size,
                "gpu_layers": s.gpu_layers,
                "backend": s.backend,
            })
        })
        .collect();

    Json(json!({
        "ok": true,
        "data": {
            "host": cfg.host,
            "slots": slots,
            "api_port": cfg.api_port,
            "threads": cfg.threads,
            "extra_args": cfg.extra_args,
        }
    }))
}

async fn gpu_status() -> Json<Value> {
    let res = SystemResources::detect();
    Json(json!({
        "ok": true,
        "data": {
            "gpu_name": res.gpu_name,
            "gpu_vendor": format!("{:?}", res.gpu_vendor),
            "gpu_vram_total_mb": res.gpu_vram_total_mb,
            "gpu_vram_free_mb": res.gpu_vram_free_mb,
            "ram_total_mb": res.ram_total_mb,
            "ram_free_mb": res.ram_free_mb,
            "cpu_cores": res.cpu_cores,
            "cpu_threads": res.cpu_threads,
        }
    }))
}

// ---------------------------------------------------------------------------
// Server startup
// ---------------------------------------------------------------------------

pub async fn start_api(state: ApiState, port: u16) -> Result<(), String> {
    let app = Router::new()
        .route("/api/health", get(health))
        .route("/api/status", get(status_all))
        .route("/api/status/{index}", get(status_one))
        .route("/api/start/{index}", post(start_slot))
        .route("/api/stop/{index}", post(stop_slot))
        .route("/api/start-all", post(start_all_slots))
        .route("/api/stop-all", post(stop_all_slots))
        .route("/api/models", get(list_models))
        .route("/api/config", get(get_config))
        .route("/api/gpu", get(gpu_status))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("127.0.0.1:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| format!("Failed to bind management API to {}: {}", addr, e))?;

    tracing::info!("Bonsai management API listening on http://{}", addr);

    axum::serve(listener, app)
        .await
        .map_err(|e| format!("Management API error: {}", e))?;

    Ok(())
}
