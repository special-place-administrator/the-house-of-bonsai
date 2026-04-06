# The House of Bonsai

Run [PrismML's Bonsai-8B](https://huggingface.co/prism-ml/Bonsai-8B-gguf) — a 1-bit 8B language model — with full CUDA GPU acceleration and a native desktop launcher.

**1.15 GB model, 297 tokens/sec on RTX 5090, fits on virtually any GPU.**

## What's Inside

- **Bonsai-8B** — end-to-end 1-bit (Q1_0) language model based on Qwen3-8B architecture
- **TurboQuant llama.cpp** — fork with custom Q1_0 CUDA kernels for GPU-accelerated 1-bit inference
- **Desktop Launcher** — Rust/Dioxus native app for one-click model management (no PowerShell, no Electron)
- **Auto-Tune** — detects your GPU/RAM/CPU and optimizes settings automatically

## Prerequisites

- **Windows 10/11** (64-bit)
- **NVIDIA GPU** with CUDA support (any modern GeForce/RTX)
- **CUDA Toolkit 12.0+** — [download](https://developer.nvidia.com/cuda-downloads)
- **Visual Studio 2022** with "Desktop development with C++" workload
- **Rust** — [install via rustup](https://rustup.rs)
- **CMake** and **Ninja** (included with Visual Studio)
- **Git**

## Quick Start

```powershell
# Clone with submodules
git clone --recurse-submodules https://github.com/special-place-administrator/the-house-of-bonsai.git
cd the-house-of-bonsai

# Run the setup script (builds everything + downloads the model)
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

The setup script will:
1. Check prerequisites
2. Build llama.cpp with Q1_0 CUDA support
3. Download Bonsai-8B from HuggingFace (~1.15 GB)
4. Build the Rust launcher

Then launch:
```powershell
launcher\target\release\turboquant-launcher.exe
```

## Manual Setup

If you prefer to do it step by step:

### 1. Build llama.cpp

```powershell
cd llama-cpp
# Apply Q1_0 patch (if using upstream llama.cpp)
# git apply ..\patches\q1_0-cuda-support.patch

# Build with CUDA
powershell -ExecutionPolicy Bypass -File windows\Build-TurboQuant.ps1
```

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

Select the model, click **Start**, then **Chat**.

## Architecture

```
the-house-of-bonsai/
  launcher/          # Rust + Dioxus desktop GUI
    src/
      main.rs        # App entry, UI components
      config.rs      # Settings persistence
      models.rs      # GGUF model discovery
      process.rs     # Async llama-server management
      resources.rs   # GPU/RAM/CPU detection + auto-tune
    assets/
      style.css      # Dark theme UI
  llama-cpp/         # TurboQuant llama.cpp fork (git submodule)
  models/            # Downloaded GGUF models (gitignored)
  patches/           # Q1_0 CUDA patch for vanilla llama.cpp
  scripts/           # Setup and download scripts
```

## How It Works

### Q1_0 Quantization

Bonsai-8B uses PrismML's Q1_0 format — each weight is a single bit (`0` = -scale, `1` = +scale) with a shared FP16 scale per 128 elements. This gives:

- **14x smaller** than FP16 (1.15 GB vs 16.4 GB)
- **6x faster** inference on CUDA
- **Competitive benchmarks** — 70.5% avg across 6 categories

Standard llama.cpp doesn't support Q1_0. This project includes custom CUDA kernels for dequantization, matrix-vector multiply, and matrix-matrix multiply with Q1_0 tensors.

### TurboQuant KV Cache

The launcher supports TurboQuant compressed KV caches (turbo2/turbo3/turbo4) for even more VRAM savings on the runtime context window.

## Compatibility

Tested on:
- RTX 5090 (32 GB) — 297 t/s
- Should work on any NVIDIA GPU with CUDA 12.0+ support

## Credits

This project stands on the shoulders of several excellent open-source projects:

- **[PrismML](https://prismml.com)** — creators of Bonsai-8B model, Q1_0 quantization format, and the 1-bit inference research that makes this possible
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (ggml-org) — the foundational C/C++ inference engine
- **[TurboQuant llama.cpp](https://github.com/spiritbuun/llama-cpp-turboquant-cuda)** (spiritbuun) — the fork providing TurboQuant KV cache compression (turbo2/turbo3/turbo4) that we extended with Q1_0 CUDA support
- **[Dioxus](https://dioxuslabs.com)** — Rust native desktop UI framework
- **[Qwen3](https://huggingface.co/Qwen)** (Alibaba) — the base architecture that Bonsai-8B was trained on

This project merges PrismML's Q1_0 CUDA kernels into the TurboQuant fork, wraps it in a lightweight Rust launcher, and packages it all for one-click deployment.

## License

Apache 2.0 (matching Bonsai-8B and llama.cpp)
