use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct HfModelResult {
    #[serde(rename = "modelId")]
    pub model_id: String,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
}

#[derive(Debug, Clone)]
pub struct GgufFileInfo {
    pub repo_id: String,
    pub filename: String,
    pub size_bytes: u64,
    pub download_url: String,
}

/// Search HuggingFace for GGUF models
pub async fn search_models(query: &str) -> Result<Vec<HfModelResult>, String> {
    let url = format!(
        "https://huggingface.co/api/models?search={}&filter=gguf&sort=lastModified&direction=-1&limit=5",
        urlencoding::encode(query)
    );
    let resp = reqwest::get(&url).await.map_err(|e| e.to_string())?;
    let models: Vec<HfModelResult> = resp.json().await.map_err(|e| e.to_string())?;
    Ok(models)
}

/// List GGUF files in a HuggingFace repo
pub async fn list_gguf_files(repo_id: &str) -> Result<Vec<GgufFileInfo>, String> {
    let url = format!("https://huggingface.co/api/models/{}/tree/main", repo_id);
    let resp = reqwest::get(&url).await.map_err(|e| e.to_string())?;
    let files: Vec<serde_json::Value> = resp.json().await.map_err(|e| e.to_string())?;

    let mut gguf_files = Vec::new();
    for file in files {
        let path = file["path"].as_str().unwrap_or("");
        if path.ends_with(".gguf") {
            let size = file["size"].as_u64().unwrap_or(0);
            gguf_files.push(GgufFileInfo {
                repo_id: repo_id.to_string(),
                filename: path.to_string(),
                size_bytes: size,
                download_url: format!(
                    "https://huggingface.co/{}/resolve/main/{}",
                    repo_id, path
                ),
            });
        }
    }
    Ok(gguf_files)
}

/// Download a file with progress callback
pub async fn download_model(
    url: &str,
    dest_path: &Path,
    progress_tx: tokio::sync::watch::Sender<(u64, u64)>,
) -> Result<(), String> {
    // Ensure parent directory exists
    if let Some(parent) = dest_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| e.to_string())?;
    }

    let client = reqwest::Client::new();
    let resp = client
        .get(url)
        .header("User-Agent", "the-house-of-bonsai/1.0")
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let total = resp.content_length().unwrap_or(0);
    let _ = progress_tx.send((0, total));

    let mut file = tokio::fs::File::create(dest_path)
        .await
        .map_err(|e| e.to_string())?;
    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = 0;

    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| e.to_string())?;
        file.write_all(&chunk).await.map_err(|e| e.to_string())?;
        downloaded += chunk.len() as u64;
        let _ = progress_tx.send((downloaded, total));
    }

    file.flush().await.map_err(|e| e.to_string())?;
    Ok(())
}
