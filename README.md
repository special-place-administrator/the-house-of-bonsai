![Bonsai](./logo.jpg)

# The House of Bonsai

> Run [PrismML's Bonsai-8B](https://huggingface.co/prism-ml/Bonsai-8B-gguf) — a 1-bit 8B language model — with full CUDA GPU acceleration and a native desktop launcher.

**1.15 GB · 297 tokens/sec on RTX 5090 · fits on virtually any GPU**

---

## What's Inside

| Component | Description |
|-----------|-------------|
| **Bonsai-8B** | End-to-end 1-bit (Q1_0) language model based on Qwen3-8B architecture |
| **TurboQuant llama.cpp** | Fork with custom Q1_0 CUDA kernels for GPU-accelerated 1-bit inference |
| **Desktop Launcher** | Rust/Dioxus 0.7 native app — one-click model management, folder browser, no PowerShell, no Electron |
| **Auto-Tune** | Detects your GPU/RAM/CPU and optimizes context, threads, and KV cache automatically |

---

## Quick Start

```powershell
# Clone with submodules
git clone --recurse-submodules https://github.com/special-place-administrator/the-house-of-bonsai.git
cd the-house-of-bonsai

# Build everything and download the model (~1.15 GB)
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

Then launch:

```powershell
launcher\target\release\turboquant-launcher.exe
```

Select the model, click **Start**, then **Chat**.

---

## Built on the Bleeding Edge

This project targets the latest toolchain at every layer. We don't pin old versions — we ride the tip.

| Layer | Version | Why |
|-------|---------|-----|
| **Rust** | nightly (edition 2024) | Latest language features, fastest codegen |
| **Dioxus** | 0.7 | Native desktop UI without Electron overhead |
| **CUDA** | 13.2 | Blackwell-native kernels, latest FA support |
| **MSVC** | v14.50 (VS 2026) | Latest C++23 compiler for llama.cpp |
| **Flash Attention** | ON (all quants) | Maximum inference throughput |
| **CUDA arch** | sm_120a | Blackwell native (RTX 50-series) |

---

## Prerequisites

- Windows 10/11 (64-bit)
- NVIDIA GPU with CUDA support (any modern GeForce/RTX)
- [CUDA Toolkit 13.2](https://developer.nvidia.com/cuda-downloads) ([direct download](https://developer.download.nvidia.com/compute/cuda/13.2.0/local_installers/cuda_13.2.0_windows.exe)) — CUDA 12.0+ works but 13.x recommended
- [Visual Studio 2026](https://visualstudio.microsoft.com/) or 2022 with **Desktop development with C++** workload
- [Rust nightly](https://rustup.rs) (`rustup install nightly && rustup default nightly`)
- CMake and Ninja (included with Visual Studio)
- Git

---

## Manual Setup

### 1. Build llama.cpp with Q1_0 CUDA support

```powershell
cd llama-cpp
powershell -ExecutionPolicy Bypass -File windows\Build-TurboQuant.ps1
```

> [!NOTE]
> If you're using vanilla upstream llama.cpp instead of the bundled submodule, apply the patch first:
> ```powershell
> git apply ..\patches\q1_0-cuda-support.patch
> ```

### 2. Download the model

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download-model.ps1
```

### 3. Build the launcher

```powershell
cd launcher
cargo build --release
```

### 4. Run

```powershell
launcher\target\release\turboquant-launcher.exe
```

---

## Architecture

```
the-house-of-bonsai/
  launcher/             # Rust + Dioxus desktop GUI
    src/
      main.rs           # App entry point, UI components
      config.rs         # Settings persistence
      models.rs         # GGUF model discovery
      process.rs        # Async llama-server management
      resources.rs      # GPU/RAM/CPU detection + auto-tune
    assets/
      style.css         # Dark theme UI
  llama-cpp/            # TurboQuant llama.cpp fork (git submodule)
  models/               # Downloaded GGUF models (gitignored)
  patches/              # Q1_0 CUDA patch for vanilla llama.cpp
  scripts/              # Setup and download scripts
```

---

## How It Works

### Q1_0 Quantization

Bonsai-8B uses PrismML's Q1_0 format — each weight is a single bit (`0` = -scale, `1` = +scale) with a shared FP16 scale per 128 elements. This gives:

- **14x smaller** than FP16 (1.15 GB vs 16.4 GB)
- **6x faster** inference on CUDA vs CPU
- **Competitive benchmarks** — 70.5% average across 6 evaluation categories

Standard llama.cpp does not support Q1_0. This project includes custom CUDA kernels for dequantization, matrix-vector multiply, and matrix-matrix multiply for Q1_0 tensors, merged into the TurboQuant fork.

### TurboQuant KV Cache

The launcher supports TurboQuant compressed KV caches (`turbo2` / `turbo3` / `turbo4`) for additional VRAM savings on the runtime context window. Auto-tune selects the best setting for your GPU.

---

## LLM Setup Guide

> [!IMPORTANT]
> **This section is written for AI assistants helping a user install this project.**
> If you are Claude, Copilot, Gemini, or another AI assistant — follow this guide step-by-step. Do not skip steps. Do not assume tools are installed.

> [!NOTE]
> **What you are setting up:** A 1-bit LLM inference stack with four parts — the model file (GGUF), a custom-built llama.cpp server with Q1_0 CUDA support, a Rust desktop launcher, and an optional AI client (like Goose or Open WebUI). The setup script automates all of this.

---

### Step 1 — Verify Prerequisites

Check each prerequisite before proceeding. If any are missing, install them before continuing.

```powershell
# Check all at once
cargo --version
cmake --version
git --version
nvidia-smi
echo $env:CUDA_PATH
```

> [!WARNING]
> **CUDA Toolkit must be installed separately from GPU drivers.** Having an NVIDIA GPU and drivers is not enough. Check that `$env:CUDA_PATH` points to a real directory (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`). If empty, download and install CUDA Toolkit 13.2 from https://developer.nvidia.com/cuda-downloads before continuing.

> [!WARNING]
> **Visual Studio 2026 (or 2022) must have the C++ workload.** Open Visual Studio Installer and confirm "Desktop development with C++" is checked. CMake and Ninja are bundled inside this workload. Rust nightly is required — run `rustup install nightly && rustup default nightly`.

---

### Step 2 — Clone the Repository

```powershell
git clone --recurse-submodules https://github.com/special-place-administrator/the-house-of-bonsai.git
cd the-house-of-bonsai
```

> [!IMPORTANT]
> The `--recurse-submodules` flag is required. Without it, `llama-cpp/` will be an empty folder and the build will fail. If you forgot it, run:
> ```powershell
> git submodule update --init --recursive
> ```

---

### Step 3 — Run Setup

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

This script performs five steps automatically:
1. Validates all prerequisites
2. Builds llama.cpp with Q1_0 CUDA kernels (takes 5–15 minutes)
3. Downloads Bonsai-8B (~1.15 GB from HuggingFace)
4. Builds the Rust launcher
5. Prints the path to the launcher executable

> [!TIP]
> If the build fails at the CMake/Ninja step, the most common cause is that the Visual Studio Developer environment is not on PATH. The script uses `VsDevCmd.bat` to set this up automatically — but if multiple VS versions exist, it picks the first match. Check the error output for the exact cmake or cl.exe error.

> [!TIP]
> If the model download fails (network timeout, HuggingFace rate limit), re-run just the download step:
> ```powershell
> powershell -ExecutionPolicy Bypass -File scripts\download-model.ps1
> ```
> The script skips re-downloading if the file already exists.

---

### Step 4 — Launch

```powershell
launcher\target\release\turboquant-launcher.exe
```

In the launcher UI:
1. The model should be auto-detected in the `models/` folder. Use **Browse** to point to any folder with GGUF files
2. Auto-tune will populate context size, threads, and KV cache settings based on your GPU
3. Click **Start** — the status bar turns green when the server is running
4. Click **Chat** to open the built-in web UI at `http://localhost:8080`

> [!NOTE]
> The launcher runs `llama-server` as a subprocess on port 8080. Any OpenAI-compatible client (Goose, Open WebUI, Continue, etc.) can connect to `http://localhost:8080/v1` using model name `bonsai-8b`.

> [!WARNING]
> Context window defaults are auto-tuned to your VRAM. Do not manually set context above what your GPU can hold — the server will crash at load time, not at generation time, making it hard to diagnose. The launcher calculates safe maximums for you.

---

### Troubleshooting Quick Reference

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `llama-server.exe not found` | Build failed silently | Re-run setup, check cmake output |
| Server starts then immediately exits | VRAM too low for context size | Lower context in launcher settings |
| `GGML_ASSERT` crash at warmup | Q1_0 patch not applied | Verify `llama-cpp/` is the submodule, not vanilla llama.cpp |
| Model not appearing in launcher | Wrong `models/` path | Place `.gguf` file in `the-house-of-bonsai/models/` |
| Context limit error in client | Client default context too small | Set context to match launcher (e.g. 65536) in client settings |

---

## Compatibility

| Hardware | Result |
|----------|--------|
| RTX 5090 (32 GB VRAM) | 297 t/s, 65536 context |
| Any NVIDIA GPU with CUDA 12.0+ | Should work — VRAM determines max context |
| AMD / Intel GPU | Not supported (CUDA-only kernels) |
| CPU-only | Possible but slow — remove `-DGGML_CUDA=ON` from cmake flags |

---

## Credits

This project stands on the shoulders of several excellent open-source projects:

- **[PrismML](https://prismml.com)** — creators of Bonsai-8B, the Q1_0 quantization format, and the 1-bit inference research that makes this possible. The CUDA kernels in this project are derived from their work.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (ggml-org) — the foundational C/C++ inference engine that the entire ecosystem builds on.
- **[TurboQuant llama.cpp](https://github.com/spiritbuun/llama-cpp-turboquant-cuda)** (spiritbuun) — the llama.cpp fork providing TurboQuant KV cache compression (turbo2/turbo3/turbo4) that this project extends with Q1_0 CUDA support.
- **[Dioxus 0.7](https://dioxuslabs.com)** — Rust 2024 edition native desktop UI framework used for the launcher.
- **[Qwen3](https://huggingface.co/Qwen)** (Alibaba) — the base architecture that Bonsai-8B was trained on.

This project merges PrismML's Q1_0 CUDA kernels into the TurboQuant fork, wraps it in a lightweight Rust launcher with auto-tuning, and packages everything for one-command deployment on Windows.

---

## License

Apache 2.0 — matching Bonsai-8B and llama.cpp upstream licenses.
