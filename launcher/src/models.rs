use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub display_name: String,
    pub path: PathBuf,
    pub source: ModelSource,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelSource {
    GgufFile,
    Ollama,
    Recent,
}

pub struct ModelCatalog {
    pub entries: Vec<ModelEntry>,
}

impl ModelCatalog {
    pub fn scan(shared_root: &Path, recent_models: &[String]) -> Self {
        let mut entries = Vec::new();
        let mut seen = std::collections::HashSet::new();

        // Add recent models first
        for path_str in recent_models {
            let path = PathBuf::from(path_str);
            if path.exists() && seen.insert(path_str.to_lowercase()) {
                let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                let display = path.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| path_str.clone());
                entries.push(ModelEntry {
                    display_name: display,
                    path,
                    source: ModelSource::Recent,
                    size_bytes: size,
                });
            }
        }

        // Scan GGUF directory
        let gguf_dir = shared_root.join("gguf");
        if gguf_dir.is_dir() {
            Self::scan_gguf_dir(&gguf_dir, &mut entries, &mut seen);
        }

        // Scan Ollama blobs
        Self::scan_ollama(&mut entries, &mut seen);

        entries.sort_by(|a, b| a.display_name.to_lowercase().cmp(&b.display_name.to_lowercase()));
        Self { entries }
    }

    fn scan_gguf_dir(dir: &Path, entries: &mut Vec<ModelEntry>, seen: &mut std::collections::HashSet<String>) {
        let walker = match std::fs::read_dir(dir) {
            Ok(w) => w,
            Err(_) => return,
        };

        for entry in walker.flatten() {
            let path = entry.path();
            if path.is_dir() {
                Self::scan_gguf_dir(&path, entries, seen);
                continue;
            }

            let name = path.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();

            if !name.ends_with(".gguf") { continue; }
            if name.starts_with("ggml-vocab-") { continue; }
            if name.starts_with("ollama_") { continue; }

            let key = path.to_string_lossy().to_lowercase();
            if !seen.insert(key) { continue; }

            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            entries.push(ModelEntry {
                display_name: name,
                path,
                source: ModelSource::GgufFile,
                size_bytes: size,
            });
        }
    }

    fn scan_ollama(entries: &mut Vec<ModelEntry>, seen: &mut std::collections::HashSet<String>) {
        let ollama_root = Self::find_ollama_root();
        let Some(root) = ollama_root else { return };

        let manifests_root = root.join("manifests");
        let blobs_root = root.join("blobs");
        if !manifests_root.is_dir() || !blobs_root.is_dir() { return; }

        Self::walk_ollama_manifests(&manifests_root, &manifests_root, &blobs_root, entries, seen);
    }

    fn find_ollama_root() -> Option<PathBuf> {
        // Check OLLAMA_MODELS env var
        if let Ok(val) = std::env::var("OLLAMA_MODELS") {
            if !val.is_empty() {
                let p = PathBuf::from(&val);
                if p.is_dir() { return Some(p); }
            }
        }

        // Check shared model root
        let shared = PathBuf::from(r"C:\AI_STUFF\LLM_MODEL\ollama");
        if shared.is_dir() { return Some(shared); }

        // Default location
        if let Some(home) = dirs::home_dir() {
            let default = home.join(".ollama").join("models");
            if default.is_dir() { return Some(default); }
        }

        None
    }

    fn walk_ollama_manifests(
        dir: &Path,
        manifests_root: &Path,
        blobs_root: &Path,
        entries: &mut Vec<ModelEntry>,
        seen: &mut std::collections::HashSet<String>,
    ) {
        let walker = match std::fs::read_dir(dir) {
            Ok(w) => w,
            Err(_) => return,
        };

        for entry in walker.flatten() {
            let path = entry.path();
            if path.is_dir() {
                Self::walk_ollama_manifests(&path, manifests_root, blobs_root, entries, seen);
                continue;
            }

            // Parse manifest file
            let content = match std::fs::read_to_string(&path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let manifest: serde_json::Value = match serde_json::from_str(&content) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let layers = match manifest.get("layers").and_then(|l| l.as_array()) {
                Some(l) => l,
                None => continue,
            };

            let model_layer = layers.iter().find(|l| {
                l.get("mediaType").and_then(|m| m.as_str()) == Some("application/vnd.ollama.image.model")
            });

            let Some(layer) = model_layer else { continue };
            let Some(digest) = layer.get("digest").and_then(|d| d.as_str()) else { continue };

            let blob_name = digest.replace(':', "-");
            let blob_path = blobs_root.join(&blob_name);
            if !blob_path.exists() { continue; }

            // Build display name from relative path
            let relative = path.strip_prefix(manifests_root)
                .map(|r| r.to_string_lossy().replace('\\', "/"))
                .unwrap_or_default();

            let parts: Vec<&str> = relative.split('/').collect();
            if parts.len() < 2 { continue; }

            let tag = parts.last().unwrap_or(&"latest");
            let repo = parts[..parts.len() - 1].join("/");
            let display = format!("[Ollama] {}:{}", repo, tag);

            let key = blob_path.to_string_lossy().to_lowercase();
            if !seen.insert(key) { continue; }

            let size = std::fs::metadata(&blob_path).map(|m| m.len()).unwrap_or(0);
            entries.push(ModelEntry {
                display_name: display,
                path: blob_path,
                source: ModelSource::Ollama,
                size_bytes: size,
            });
        }
    }

    pub fn resolve(&self, name: &str) -> Option<&ModelEntry> {
        self.entries.iter().find(|e| {
            e.display_name.eq_ignore_ascii_case(name)
                || e.path.to_string_lossy().eq_ignore_ascii_case(name)
        })
    }
}

pub fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.0} MB", bytes as f64 / 1_048_576.0)
    } else {
        format!("{} KB", bytes / 1024)
    }
}
