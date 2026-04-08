![Bonsai](./logo.jpg)


# The House of Bonsai

> This project was inspired by the lovely name [PrismML](https://prismml.com) gave to their 1-bit model — **Bonsai**. A whole platform sprung from that small, beautiful fact. We thank them for their courage in pursuing what others dismissed: training production-quality models on an unproven Microsoft research paper about 1-bit quantization. When the rest of the industry played it safe, PrismML took the risky path — and proved that the impossible is just unexplored.

A multi-model inference platform with a native desktop launcher. Run any GGUF model — including PrismML's 1-bit Bonsai-8B — on NVIDIA, AMD, Intel, or CPU.

**Multi-GPU backend | Multi-model serving | Auto-tune | GGUF metadata-aware | Zero config**

---

## What's Inside

| Component | Description |
|-----------|-------------|
| **Multi-Model Launcher** | Rust/Dioxus 0.7 native app — run multiple models simultaneously, each on its own port with independent settings |
| **GGUF Intelligence** | Reads model metadata to auto-detect capabilities (text, embedding, vision, code, tool use, thinking), set optimal parameters, and display capability badges |
| **Multi-Backend GPU** | CUDA (NVIDIA), Vulkan (AMD/Intel/NVIDIA), CPU — per-model backend selection, auto-detected at build time |
| **Q1_0 CUDA Kernels** | Custom 1-bit quantization support for PrismML Bonsai-8B — not available in standard llama.cpp or Ollama |
| **Auto-Tune** | Detects GPU/RAM/CPU, reads model architecture from GGUF, calculates optimal context window, KV cache, batch size per model |
| **TurboQuant KV Cache** | Compressed KV caches (turbo2/turbo3/turbo4) for additional VRAM savings |

---

## Quick Start

```powershell
# Clone with submodules
git clone --recurse-submodules https://github.com/special-place-administrator/the-house-of-bonsai.git
cd the-house-of-bonsai

# Build everything (auto-detects CUDA + Vulkan) and download the model
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

Then launch:

```powershell
launcher\target\release\turboquant-launcher.exe
```

Select a model, click **Start**, then **Chat**. Add more models with **+ Add Model**.

### Getting Models

Every model card has a built-in **HuggingFace search** — type a name, pick a GGUF file, and download it directly inside the app with a progress bar. Sort by Newest, Popular, or A-Z. No manual downloads needed.

To get started, search for these two essential models:

| Search for | What you get | Use case |
|------------|-------------|----------|
| `bonsai 8b` | PrismML Bonsai-8B (1.15 GB, Q1_0) | Fast local text/chat with tool use and thinking |
| `nomic embed` | nomic-embed-text-v2-moe (1.8 GB) | Embeddings for MCP tools like Prism, RAG pipelines, semantic search |

These two models together give you a complete local AI stack — text generation and embeddings — for tools like [Prism MCP](https://github.com/dcostenco/prism-mcp), Goose, Open WebUI, or any OpenAI-compatible client.

You can also place `.gguf` files manually in the `models/` folder, or use **Browse** to point to any directory.

---

## Built on the Bleeding Edge

| Layer | Version | Why |
|-------|---------|-----|
| **Rust** | nightly 1.96+ (edition 2024) | Latest language features, fastest codegen |
| **Dioxus** | 0.7 | Native desktop UI without Electron overhead |
| **CUDA** | 13.2 | Blackwell-native kernels, latest FA support |
| **Vulkan** | 1.4 | Universal GPU compute for AMD + Intel |
| **MSVC** | v14.50 (VS 2026) | Latest C++ compiler for llama.cpp |
| **Flash Attention** | ON (all quants) | Maximum inference throughput |

---

## Multi-Backend GPU Support

The build system auto-detects available GPU SDKs and compiles all backends:

| Backend | DLL | Hardware | SDK Required |
|---------|-----|----------|-------------|
| **CUDA** | ggml-cuda.dll | NVIDIA GeForce/RTX | [CUDA Toolkit 13.2](https://developer.nvidia.com/cuda-downloads) ([direct download](https://developer.download.nvidia.com/compute/cuda/13.2.0/local_installers/cuda_13.2.0_windows.exe)) |
| **Vulkan** | ggml-vulkan.dll | AMD Radeon, Intel Arc/Battlemage, NVIDIA | [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) |
| **CPU** | ggml-cpu.dll | Any x86-64 processor | None |

Each model card has a **Backend** dropdown — you can run one model on CUDA and another on CPU to save GPU VRAM.

---

## Multi-Model Serving

Run multiple models simultaneously, each as a separate llama-server process:

- **Model 1** — Bonsai-8B on port 8080 (text/chat, CUDA)
- **Model 2** — nomic-embed-text on port 8081 (embeddings, CPU)
- **Model 3** — any other GGUF model on port 8082

Each model card has independent settings: port, context size, cache type, batch size, flash attention, GPU layers, backend, and parallel slots.

Any OpenAI-compatible client can connect: [Prism MCP](https://github.com/dcostenco/prism-mcp), Goose, Open WebUI, Continue, etc.

---

## Model Capability Detection

The launcher reads GGUF metadata and auto-detects model capabilities, shown as badges in each card:

| Badge | Capability | Detection |
|-------|-----------|-----------|
| Text | Chat/completion | Chat template present |
| Embed | Embedding model | BERT/nomic architecture |
| Vision | Image understanding | LLaVA/InternVL architecture |
| Code | Code generation | "coder" in model name |
| Tools | Function calling | Tools support in chat template |
| Think | Reasoning mode | Thinking tokens in chat template |
| Speech | Audio processing | Whisper architecture |

Embedding models are auto-configured on selection: f16 cache, flash attention off, parallel slots increased.

---

## Prerequisites

- Windows 10/11 (64-bit)
- **At least one of:**
  - NVIDIA GPU + [CUDA Toolkit 13.2](https://developer.nvidia.com/cuda-downloads) ([direct download](https://developer.download.nvidia.com/compute/cuda/13.2.0/local_installers/cuda_13.2.0_windows.exe))
  - AMD/Intel GPU + [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)
  - CPU only (no SDK needed — slower but works everywhere)
- [Visual Studio 2026](https://visualstudio.microsoft.com/) or 2022 with **Desktop development with C++** workload
- [Rust nightly](https://rustup.rs) (`rustup install nightly && rustup default nightly`)
- CMake and Ninja (included with Visual Studio)
- Git

---

## Architecture

```
the-house-of-bonsai/
  launcher/             # Rust + Dioxus 0.7 desktop GUI
    src/
      main.rs           # App entry point, multi-model card UI
      config.rs         # Per-slot settings, persistence, migration
      gguf.rs           # GGUF header reader, capability detection
      models.rs         # GGUF file discovery + Ollama scanner
      process.rs        # Multi-process llama-server management
      resources.rs      # GPU vendor detection, multi-backend, auto-tune
    assets/
      style.css         # Dark theme with model card layout
  llama-cpp/            # TurboQuant llama.cpp fork (git submodule)
  models/               # Downloaded GGUF models (gitignored)
  patches/              # Q1_0 CUDA patch for vanilla llama.cpp
  scripts/              # Setup and download scripts
```

---

## How It Works

### Q1_0 Quantization

![ver130](./ver131.png)

Bonsai-8B uses PrismML's Q1_0 format — each weight is a single bit (`0` = -scale, `1` = +scale) with a shared FP16 scale per 128 elements. This gives:

- **14x smaller** than FP16 (1.15 GB vs 16.4 GB)
- **6x faster** inference on CUDA vs CPU
- **Competitive benchmarks** — 70.5% average across 6 evaluation categories

Standard llama.cpp and Ollama do not support Q1_0. This project includes custom CUDA kernels for dequantization, matrix-vector multiply, and matrix-matrix multiply for Q1_0 tensors.

### GGUF-Aware Auto-Tune

When you select a model, the launcher reads its GGUF header to extract:
- Architecture (layers, heads, embedding dimensions) for accurate KV cache sizing
- Native context length as the upper bound
- Recommended sampling parameters (temperature, top_k, top_p)
- Model capabilities for auto-configuration

This replaces guesswork with model-specific optimization.

### TurboQuant KV Cache

The launcher supports TurboQuant compressed KV caches (`turbo2` / `turbo3` / `turbo4`) for additional VRAM savings on the runtime context window. Auto-tune selects the best setting for your GPU.

---

## LLM Setup Guide

> [!IMPORTANT]
> **This section is written for AI assistants helping a user install this project.**
> If you are Claude, Copilot, Gemini, or another AI assistant — follow this guide step-by-step. Do not skip steps. Do not assume tools are installed.

> [!NOTE]
> **What you are setting up:** A multi-model LLM inference platform with three parts — GGUF model files, a custom-built llama.cpp server with Q1_0 CUDA + Vulkan support, and a Rust desktop launcher that manages everything. The build script auto-detects GPU hardware and builds appropriate backends.

---

### Step 1 — Verify Prerequisites

```powershell
# Check all at once
cargo --version
cmake --version
git --version
nvidia-smi          # NVIDIA GPU (optional)
echo $env:CUDA_PATH # CUDA Toolkit (optional)
echo $env:VULKAN_SDK # Vulkan SDK (optional)
```

> [!WARNING]
> **GPU SDKs are optional but recommended.** Without CUDA or Vulkan SDK, the build produces a CPU-only binary. For NVIDIA: install CUDA Toolkit 13.2. For AMD/Intel: install Vulkan SDK. Both can coexist.

> [!WARNING]
> **Visual Studio 2026 (or 2022) must have the C++ workload.** Open Visual Studio Installer and confirm "Desktop development with C++" is checked. CMake and Ninja are bundled inside this workload.

> [!WARNING]
> **Rust nightly is required.** Run `rustup install nightly && rustup default nightly`. The project uses Rust 2024 edition features.

> [!WARNING]
> **MSVC M_PI compatibility.** If the build fails with `M_PI undeclared`, the fix is already applied in the submodule. If building from vanilla llama.cpp, add `#ifndef M_PI / #define M_PI 3.14159265358979323846 / #endif` after the includes in `ggml-turbo-quant.c`.

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

The build script auto-detects available GPU SDKs and builds accordingly:
1. Validates all prerequisites
2. Detects CUDA Toolkit and Vulkan SDK
3. Builds llama.cpp with all available backends (10-20 minutes)
4. Downloads Bonsai-8B (~1.15 GB from HuggingFace)
5. Builds the Rust launcher with `cargo build --release`
6. Reports which backend DLLs were produced

> [!TIP]
> Alternative: use `build-now.bat` for a quick rebuild without model download.

> [!TIP]
> If the model download fails, re-run just the download step:
> ```powershell
> powershell -ExecutionPolicy Bypass -File scripts\download-model.ps1
> ```

---

### Step 4 — Launch

```powershell
launcher\target\release\turboquant-launcher.exe
```

In the launcher UI:
1. Models in the `models/` folder are auto-detected. Use **Browse** to point to any folder with GGUF files
2. Select a model — auto-tune reads GGUF metadata and configures optimal settings
3. Capability badges appear (Text, Embed, Code, Tools, Think, etc.)
4. Choose a GPU backend from the dropdown (Auto/CUDA/Vulkan/CPU)
5. Click **Start** — the status dot turns green when running
6. Click **Chat** to open the built-in web UI
7. Use **+ Add Model** to run additional models on separate ports

> [!NOTE]
> Each model runs as a separate llama-server process. Any OpenAI-compatible client can connect to `http://localhost:<port>/v1`. Settings persist across restarts.

> [!WARNING]
> Context window defaults are auto-tuned to your VRAM and the specific model's architecture. Do not manually set context above what your GPU can hold.

---

### Troubleshooting Quick Reference

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `llama-server.exe not found` | Build failed | Re-run `build-now.bat`, check cmake output |
| Server starts then exits | VRAM too low for context | Lower context or switch backend to CPU |
| `GGML_ASSERT` crash | Q1_0 patch not applied | Verify `llama-cpp/` is the submodule, not vanilla |
| `M_PI undeclared` | MSVC compatibility | Already fixed in submodule; see Step 1 notes |
| No GPU backends built | Missing SDK | Install CUDA Toolkit and/or Vulkan SDK |
| Ninja permission denied | Stale build directory | Delete `llama-cpp/build/` and rebuild |
| Model not appearing | Wrong directory | Place `.gguf` files in `models/` or use Browse |
| Embedding model wrong settings | Stale config | Re-select the model to trigger auto-detection |

---

## Compatibility

| Hardware | Backend | Result |
|----------|---------|--------|
| RTX 5090 (32 GB) | CUDA | 297 t/s, 65536 context |
| RTX 5060 Ti (16 GB) | CUDA | 95 t/s, 53248 context |
| Any NVIDIA GPU | CUDA | VRAM determines max context |
| AMD Radeon RX 7000+ | Vulkan | Works — performance varies |
| Intel Arc / Battlemage | Vulkan | Works — performance varies |
| CPU only | CPU | Slow but functional on any machine |

---

## Credits

- **[PrismML](https://prismml.com)** — creators of Bonsai-8B, the Q1_0 quantization format, and the 1-bit inference research. CUDA kernels derived from their work.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (ggml-org) — the foundational C/C++ inference engine.
- **[TurboQuant llama.cpp](https://github.com/spiritbuun/llama-cpp-turboquant-cuda)** (spiritbuun) — TurboQuant KV cache compression fork extended with Q1_0 CUDA support.
- **[Dioxus 0.7](https://dioxuslabs.com)** — Rust native desktop UI framework.
- **[Qwen3](https://huggingface.co/Qwen)** (Alibaba) — base architecture for Bonsai-8B.

---

## License

**The House of Bonsai launcher** (everything under `launcher/`, `scripts/`, `patches/`) is licensed under [PolyForm Noncommercial License 1.0.0](./LICENSE). The official license text is also available from the [PolyForm Project](https://polyformproject.org/licenses/noncommercial/1.0.0/).

You may inspect, study, and use the source code for noncommercial purposes, but commercial use is prohibited unless separately licensed.

**Third-party components** retain their original licenses:
- llama.cpp / TurboQuant fork — MIT License
- Bonsai-8B model — Apache 2.0 (PrismML)
- nomic-embed-text-v2-moe — Apache 2.0 (Nomic AI)
- Dioxus — MIT / Apache 2.0
