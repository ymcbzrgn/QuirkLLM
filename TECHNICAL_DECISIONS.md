# 🧠 QuirkLLM Technical Decisions

> **Living Document**: This file records all major technical decisions made during QuirkLLM development.  
> **Last Updated**: 2025-11-26  
> **Status**: Phase 0 - Technical Foundation Complete ✅
> 
> **Purpose**: Single source of truth for "why" we made specific technical choices.  
> All decisions align 100% with README.md specifications.

---

## 📋 Table of Contents

1. [Executive Summary](#executive-summary)
2. [Core Architecture](#core-architecture)
3. [Model Strategy](#model-strategy)
4. [LLM Backend](#llm-backend)
5. [Embedding & RAG](#embedding--rag)
6. [RAM Detection Strategy](#ram-detection-strategy)
7. [Quantization Strategy](#quantization-strategy)
8. [Platform Support](#platform-support)
9. [Model Distribution](#model-distribution)
10. [Project Structure](#project-structure)
11. [Dependencies](#dependencies)
12. [Testing Strategy](#testing-strategy)
13. [PyPI Package Strategy](#pypi-package-strategy)
14. [MVP Scope](#mvp-scope)
15. [Decision Log](#decision-log)

---

## 📊 Executive Summary

**All critical technical decisions have been finalized.** QuirkLLM is ready for Phase 1 implementation.

### ✅ Finalized Decisions

| Category | Decision | Rationale |
|----------|----------|-----------|
| **Base Model** | DeepSeek-Coder-1.3B-base | MIT license, 1.3B params, 16K context, strong code performance |
| **Fine-tuned Model** | QuirkLLM-1.3B (Phase 1) | Fine-tune with 30 D-components, name honors origin |
| **LLM Backend** | **Hybrid**: llama-cpp + mlx | Optimal for all platforms (macOS Metal + others) |
| **Python Version** | **>=3.11** | 25% performance boost, modern syntax, 2027 support |
| **Quantization** | **Profile-based**: Q4_K_M / Q8_0 | 8GB RAM→Q4, 32GB RAM→Q8 (automatic) |
| **Embedding** | CodeBERT (small/base/large) | Code-specific, profile-adaptive sizes |
| **Vector DB** | LanceDB | Disk-backed, no server, Python-native |
| **Model Source** | HuggingFace Hub | Free hosting, CDN, resume downloads |
| **Package Name** | `quirkllm` | PyPI, Apache 2.0 license |
| **Initial Version** | `0.1.0` (Beta) | MVP with all 4 modes + fine-tuned model |
| **Project Root** | `~/.quirkllm/` | Config, models, cache, sessions, plans |

### 🎯 MVP Scope (Phase 1)

**Must Have (Non-Negotiable):**
- ✅ All 4 modes: Chat / YAMI / Plan / Ghost
- ✅ Fine-tuned QuirkLLM-1.3B model (not just base)
- ✅ RAM-adaptive profiles (Survival/Comfort/Power/Beast)
- ✅ Full RAG system (LanceDB + CodeBERT)
- ✅ Project analyzer + package manager detection
- ✅ **Pass all benchmarks**: HumanEval ≥70%, Multi-turn ≥85%

**Deferred to Phase 5:**
- ⏭️ MCP Server (Neural Link)

---

## 🏗️ Core Architecture

### Pipeline Design: 8-Stage Sequential

```
System Detector → Profile Manager → Interactive CLI → Project Analyzer
    ↓                                                           ↓
Adaptive Inference ← Conversation Engine ← Adaptive RAG ← Output Handler
```

**Decision**: Sequential pipeline with clear separation of concerns.

**Rationale**:
- ✅ Each stage has single responsibility
- ✅ Easy to test in isolation
- ✅ Simple to extend/modify stages
- ✅ Matches CLI-first philosophy (each stage = CLI command potential)

**Alternatives Considered**:
- ❌ Event-driven architecture (too complex for CLI tool)
- ❌ Monolithic REPL (hard to test/maintain)

**Status**: ✅ Finalized (Matches README exactly)

---

## 🎯 Model Strategy

### Base Model: **DeepSeek-Coder-1.3B-base**

**Decision**: Use DeepSeek-Coder-1.3B-base as foundation, fine-tune to create QuirkLLM-1.3B.

#### Why DeepSeek-Coder-1.3B?

| Criterion | DeepSeek-Coder-1.3B | Score |
|-----------|---------------------|-------|
| **Size** | 1.3B params (exact target) | ✅ Perfect fit |
| **License** | MIT | ✅ Commercial use OK |
| **Context** | 16K native (extendable to 128K) | ✅ Supports all profiles |
| **Performance** | HumanEval ~34% (base) | ✅ Good foundation |
| **GGUF** | Available (TheBloke) | ✅ Ready to use |
| **Training** | The Stack, StarCoder data | ✅ Code-focused |

**Alternatives Considered:**
- ❌ **Qwen2.5-Coder-1.5B**: Slightly larger (1.5B), less mature ecosystem
- ❌ **Phi-2 (2.7B)**: Too large for Survival mode (8GB RAM target)
- ❌ **StableCode-3B**: Non-commercial license (incompatible with README promise)

#### Fine-tuning: **QuirkLLM-1.3B**

**Timeline**: Phase 1 (MVP includes fine-tuned model)

**Naming Convention**:
```
Base:       deepseek-ai/deepseek-coder-1.3b-base
Fine-tuned: ymcbzrgn/QuirkLLM-1.3B
            ↑ Credits DeepSeek in model card + README
```

**Training Tool**: Axolotl (as per README)

**Dataset Components** (30 D-components from README):
- D1-D8: Instruction following, multi-turn, FIM, error recovery
- D9-D15: Refactoring, framework-aware, best practices, debug
- D16-D23: Security, accessibility, i18n, state, API, DB, typing
- D24-D30: Error handling, async, components, hooks, testing, CI/CD, package, monorepo

**Training Config** (Axolotl):
```yaml
base_model: deepseek-ai/deepseek-coder-1.3b-base
adapter: qlora
lora_r: 16
lora_alpha: 32
datasets:
  - path: ./data/multi_turn_conversations.jsonl  # We'll create this
sequence_len: 4096
micro_batch_size: 2
num_epochs: 3
learning_rate: 2e-4
output_dir: ./outputs/quirkllm-1.3b
```

**Success Criteria**: HumanEval ≥70% (vs base ~34%)

**Status**: ✅ Base model selected, fine-tuning in Phase 1

---

## 🤖 LLM Backend

### Decision: **HYBRID APPROACH** ⭐

**Strategy**: Use optimal backend for each platform automatically.

| Platform | Backend | Rationale |
|----------|---------|-----------|
| **macOS Apple Silicon** | `mlx-lm` | Metal-optimized, unified memory, **fastest** (12 tok/s) |
| **macOS Intel** | `llama-cpp-python` | GGUF support, CPU-optimized (4-5 tok/s) |
| **Linux (CPU)** | `llama-cpp-python` | Best CPU performance (4 tok/s) |
| **Linux (CUDA)** | `llama-cpp-python` | GPU offload support (8 tok/s) |
| **Windows** | `llama-cpp-python` | Cross-platform, stable (4 tok/s) |
| **Embeddings (all)** | `transformers` | CodeBERT support, mature |

#### Implementation: Backend Abstraction Layer

```python
# quirkllm/backends/base.py
from abc import ABC, abstractmethod

class InferenceBackend(ABC):
    @abstractmethod
    def load_model(self, model_path: str, **kwargs): ...
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int) -> str: ...
    
    @abstractmethod
    def get_embeddings(self, text: str) -> list[float]: ...

# quirkllm/backends/llamacpp_backend.py
from llama_cpp import Llama

class LlamaCppBackend(InferenceBackend):
    def load_model(self, model_path: str, **kwargs):
        self.model = Llama(
            model_path=model_path,
            n_ctx=kwargs.get("context_length", 32768),
            n_gpu_layers=kwargs.get("n_gpu_layers", 0)
        )

# quirkllm/backends/mlx_backend.py
from mlx_lm import load, generate

class MLXBackend(InferenceBackend):
    def load_model(self, model_path: str, **kwargs):
        self.model, self.tokenizer = load(model_path)

# quirkllm/backends/factory.py
def create_backend() -> InferenceBackend:
    """Detect platform and return optimal backend."""
    if sys.platform == "darwin" and platform.processor() == "arm":
        try:
            from .mlx_backend import MLXBackend
            return MLXBackend()
        except ImportError:
            pass  # Fallback to llama-cpp
    
    from .llamacpp_backend import LlamaCppBackend
    return LlamaCppBackend()
```

#### Why Hybrid?

**✅ Advantages:**
- ✅ **Best performance on every platform** (Mac users get Metal, others get llama-cpp)
- ✅ **Widest user base** (Windows/Linux/Mac all supported)
- ✅ **Automatic selection** (transparent to user)
- ✅ **Fallback mechanism** (if MLX fails, use llama-cpp)
- ✅ **Future-proof** (easy to add ONNX, TensorRT later)

**❌ Trade-offs:**
- ❌ Maintain two backend implementations (~400 lines total)
- ❌ Test complexity (CI must test both backends)
- ❌ Model format differences (GGUF for llama-cpp, Safetensors for MLX)

**Decision**: Trade-offs acceptable. Performance gains + user coverage worth the complexity.

#### Performance Comparison (Estimated)

| Platform | llama-cpp only | Hybrid (with MLX) | Improvement |
|----------|----------------|-------------------|-------------|
| Mac M1/M2/M3 | 5 tok/s | **12 tok/s** | **+140%** |
| Mac Intel | 4 tok/s | 4 tok/s | 0% |
| Linux CPU | 4 tok/s | 4 tok/s | 0% |
| Linux CUDA | 8 tok/s | 8 tok/s | 0% |
| Windows | 4 tok/s | 4 tok/s | 0% |

**Status**: ✅ Finalized

---

## ⚙️ Quantization Strategy

### Decision: **Profile-Based Automatic Quantization**

**Strategy**: Download and use the quantization variant that matches the detected RAM profile.

| RAM Profile | Available RAM | Quantization | Model Size | Quality | Speed |
|-------------|---------------|--------------|------------|---------|-------|
| **Survival** | <8GB | Q4_K_M (4-bit) | 900MB | Good | 3 tok/s |
| **Comfort** | 8-24GB | Q4_K_M (4-bit) | 900MB | Good | 5 tok/s |
| **Power** | 24-48GB | Q8_0 (8-bit) | 1.4GB | Excellent | 8 tok/s |
| **Beast** | 48GB+ | Q8_0 (8-bit) | 1.4GB | Excellent | 12 tok/s |

#### First Run Behavior

```bash
$ quirkllm

🔍 Detecting system resources...
💾 RAM: 16GB available → COMFORT MODE
📥 Downloading optimal model for your system...

Model: QuirkLLM-1.3B (Q4_K_M - 4-bit)
Size: 900MB
From: huggingface.co/ymcbzrgn/QuirkLLM-1.3B-GGUF

[████████████████████] 900MB/900MB

✓ Model ready! Starting QuirkLLM...
```

#### Manual Override

Users can download different quantizations:

```bash
# Download 8-bit version manually
$ quirkllm --download-model Q8_0

# Use specific quantization
$ quirkllm --quantization Q8_0
```

#### GGUF Quantization Variants

| Variant | Bits | Size | Quality | Use Case |
|---------|------|------|---------|----------|
| **Q4_K_S** | 4-bit | 750MB | Acceptable | Ultra-low RAM |
| **Q4_K_M** | 4-bit | 900MB | Good | **Default (Survival/Comfort)** |
| **Q5_K_M** | 5-bit | 1.1GB | Better | Optional upgrade |
| **Q8_0** | 8-bit | 1.4GB | Excellent | **Power/Beast** |
| **F16** | 16-bit | 2.6GB | Perfect | Research only |

**Decision Rationale**:
- Q4_K_M: Best balance (size vs quality) for most users
- Q8_0: Minimal quality loss, fits 32GB+ systems comfortably

**Status**: ✅ Finalized

---

## 🎯 Model Distribution

### Decision: **HuggingFace Hub + Manifest-Based Versioning**

#### Model Hosting

**Base Model**: HuggingFace (community, free CDN)
```
https://huggingface.co/TheBloke/deepseek-coder-1.3B-GGUF
```

**Fine-tuned Model**: Your HuggingFace account
```
https://huggingface.co/ymcbzrgn/QuirkLLM-1.3B-GGUF
├── quirkllm-1.3b-v1.0.Q4_K_M.gguf
├── quirkllm-1.3b-v1.0.Q8_0.gguf
└── README.md (credits DeepSeek)
```

#### Version Manifest (Future Phase)

**Location**: HuggingFace (deferred to later phase)
```
https://huggingface.co/ymcbzrgn/QuirkLLM-Manifests/resolve/main/models.json
```

**Format**:
```json
{
  "latest": "v1.0",
  "versions": {
    "base": {
      "model_id": "TheBloke/deepseek-coder-1.3B-GGUF",
      "files": {
        "Q4_K_M": {
          "filename": "deepseek-coder-1.3b.Q4_K_M.gguf",
          "size": 900000000,
          "sha256": "abc123..."
        }
      }
    },
    "v1.0": {
      "model_id": "ymcbzrgn/QuirkLLM-1.3B-GGUF",
      "files": {
        "Q4_K_M": {
          "filename": "quirkllm-1.3b-v1.0.Q4_K_M.gguf",
          "size": 950000000,
          "sha256": "def456..."
        },
        "Q8_0": {
          "filename": "quirkllm-1.3b-v1.0.Q8_0.gguf",
          "size": 1400000000,
          "sha256": "ghi789..."
        }
      },
      "changelog": [
        "Better multi-turn conversations",
        "Framework-aware responses",
        "+15% HumanEval score"
      ]
    }
  }
}
```

#### Update Command

```bash
# Check for updates
$ quirkllm
> /update

Checking for updates...
✨ New version available: QuirkLLM-1.3B-v1.0
   Current: base
   
   Improvements:
   • Better multi-turn conversations
   • Framework-aware responses  
   • +15% HumanEval score
   
Download? (y/n) y

[████████████████████] 950MB/950MB
✓ Updated! Restart to use new model.
```

#### Offline Support

```bash
# Manual model path
$ quirkllm --model-path /path/to/local/model.gguf

# Error when offline
$ quirkllm
❌ No model found and no internet connection.

   Download manually:
   1. Visit: https://huggingface.co/ymcbzrgn/QuirkLLM-1.3B-GGUF
   2. Download: quirkllm-1.3b-v1.0.Q4_K_M.gguf
   3. Place in: ~/.quirkllm/models/
   4. Run: quirkllm --model-path ~/.quirkllm/models/quirkllm-1.3b-v1.0.Q4_K_M.gguf
```

**Status**: ✅ Finalized (HF manifest in future phase)

---

## 📁 Project Structure

### Decision: `~/.quirkllm/` as User Data Root

```
~/.quirkllm/
├── config.yaml              # User configuration
├── models/                  # Downloaded models
│   ├── .manifest.json       # Local model metadata
│   ├── deepseek-coder-1.3b.Q4_K_M.gguf
│   └── quirkllm-1.3b-v1.0.Q8_0.gguf
├── cache/                   # Runtime caches
│   ├── lancedb/             # Vector database (RAG)
│   ├── semantic/            # Semantic cache (query results)
│   └── kv/                  # KV cache disk offload (if needed)
├── sessions/                # Saved conversation sessions
│   ├── session-2025-11-26-001.json
│   ├── session-2025-11-26-002.json
│   └── last.json            # Auto-resume last session
├── plans/                   # Plan mode outputs
│   ├── auth-refactor.md
│   └── api-redesign.md
├── logs/                    # Application logs
│   ├── quirkllm.log
│   ├── quirkllm.log.1       # Rotated logs
│   └── quirkllm.log.2
└── embeddings/              # CodeBERT model cache
    └── codebert-base/
        ├── model.safetensors
        └── config.json
```

#### Configuration File: `config.yaml`

```yaml
# ~/.quirkllm/config.yaml

# Profile override (default: auto-detect)
profile: auto  # auto | survival | comfort | power | beast

# Model settings
model:
  current: "quirkllm-1.3b-v1.0.Q4_K_M"
  auto_update_check: false  # Check on startup (default: false, use /update)
  fallback_model: "deepseek-coder-1.3b.Q4_K_M"

# Context management
context:
  max_tokens: auto  # auto (profile-based) | 16384 | 32768 | 65536 | 131072
  compaction_mode: auto  # auto | aggressive | smart | relaxed | minimal

# RAG settings
rag:
  enabled: true
  cache_size: auto  # auto (profile-based) | 200MB | 500MB | 2GB | 8GB
  embedding_model: "microsoft/codebert-base"
  reranking: true  # Enable reranking (if profile allows)

# UI preferences
ui:
  compact_mode: false
  syntax_highlighting: true
  colors: true
  mode: "chat"  # chat | yami | plan | ghost

# Cache management
cache:
  max_size: 10GB  # Total cache limit (auto-cleanup when exceeded)
  cleanup: auto  # auto (LRU) | manual

# Logging
logging:
  level: "INFO"  # DEBUG | INFO | WARNING | ERROR
  file: "~/.quirkllm/logs/quirkllm.log"
  max_file_size: 50MB
  backup_count: 3

# Advanced inference
inference:
  batch_size: auto  # auto (profile-based) | 1 | 4 | 8 | 16
  n_gpu_layers: auto  # auto-detect GPU | 0 (CPU only) | 32 (full offload)
  backend: auto  # auto | llama-cpp | mlx
```

#### Session Format: JSON

```json
// ~/.quirkllm/sessions/session-2025-11-26-001.json
{
  "id": "session-2025-11-26-001",
  "created_at": "2025-11-26T14:30:00Z",
  "last_updated": "2025-11-26T15:45:00Z",
  "project_path": "/Users/yamac/my-react-app",
  "profile": "comfort",
  "mode": "chat",
  "messages": [
    {
      "role": "user",
      "content": "create a button component",
      "timestamp": "2025-11-26T14:31:00Z"
    },
    {
      "role": "assistant",
      "content": "I'll create a Button component...",
      "timestamp": "2025-11-26T14:31:05Z",
      "files_modified": ["src/components/Button.tsx"],
      "commands_run": [],
      "tokens_used": 450
    }
  ],
  "context": {
    "active_files": ["src/components/Button.tsx"],
    "token_count": 8421,
    "compaction_applied": false
  },
  "metadata": {
    "total_messages": 12,
    "total_tokens": 24500,
    "files_created": 3,
    "files_modified": 5
  }
}
```

#### Plan Format: Markdown

```markdown
<!-- ~/.quirkllm/plans/auth-refactor.md -->

# Auth System Refactoring Plan

**Created:** 2025-11-26 14:30  
**Project:** ~/my-app  
**Mode:** Plan (Read-only)

## Current State Analysis

- Monolithic AuthService.ts (450 lines)
- JWT tokens stored in localStorage (security risk)
- No refresh token mechanism
- Tightly coupled with UI components

## Proposed Architecture

### 1. Create TokenService

**New File:** `src/services/TokenService.ts`

**Responsibilities:**
- Secure token storage (httpOnly cookies)
- Automatic refresh logic
- Token expiry checks

**Interface:**
\`\`\`typescript
interface TokenService {
  saveTokens(access: string, refresh: string): void
  getAccessToken(): string | null
  refreshAccessToken(): Promise<string>
}
\`\`\`

### 2. Split AuthService
...
```

**Status**: ✅ Finalized

#### Top Candidates

##### 1. **Qwen2.5-Coder-1.5B** (Qwen Team)
- **Size**: 1.5B params
- **Context**: 128K native (!!)
- **License**: Apache 2.0 ✅
- **GGUF**: Available on HF (TheBloke, etc.)
- **HumanEval**: ~35% (reported)
- **Pros**: 
  - Largest native context (perfect for Beast mode)
  - Strong code performance
  - Very active development
- **Cons**: 
  - Slightly above 1.3B target
  - Newer model (less battle-tested)

##### 2. **DeepSeek-Coder-1.3B** (DeepSeek AI)
- **Size**: 1.3B params (exact target!)
- **Context**: 16K native, extendable to 64K
- **License**: MIT ✅
- **GGUF**: Available
- **HumanEval**: ~34%
- **Pros**: 
  - Perfect size
  - Excellent code capabilities
  - MIT license (most permissive)
- **Cons**: 
  - Lower native context than Qwen
  - China-based (potential geo-restrictions?)

##### 3. **Phi-2** (Microsoft)
- **Size**: 2.7B params
- **Context**: 2K native (expandable)
- **License**: MIT ✅
- **GGUF**: Available
- **HumanEval**: ~47% (!)
- **Pros**: 
  - Best performance
  - Microsoft backing
- **Cons**: 
  - ❌ Too large for Survival mode (8GB RAM)
  - ❌ Very limited native context
  - Not code-specific (general model)

##### 4. **StableCode-3B** (Stability AI)
- **Size**: 3B params
- **Context**: 16K
- **License**: Non-commercial research only ❌
- **Pros**: Good performance
- **Cons**: 
  - ❌ License incompatible with README promise ("free, local, commercial-OK")
  - Too large

#### **Preliminary Decision**: Qwen2.5-Coder-1.5B

**Rationale**:
- ✅ 128K native context = future-proof
- ✅ Apache 2.0 = commercial use OK
- ✅ Strong code performance
- ✅ Size acceptable (1.5GB @ 4-bit = fits Survival profile)
- ✅ Active community + updates

**Action Items**:
- [ ] Download and test Qwen2.5-Coder-1.5B-Instruct GGUF (Q4_K_M variant)
- [ ] Benchmark on HumanEval subset (20 problems)
- [ ] Test RAM usage across all 4 profiles
- [ ] Compare with DeepSeek-Coder-1.3B
- [ ] Make final decision by end of Phase 0

**Status**: 🟡 Qwen2.5-Coder-1.5B leading, needs validation

---

## 🔍 Embedding & RAG

### Embedding Model: **CodeBERT** (Microsoft)

| Profile | Model | Size | Use Case |
|---------|-------|------|----------|
| **Survival** | `microsoft/codebert-base` (small variant) | 250MB | Basic semantic search |
| **Comfort** | `microsoft/codebert-base` | 500MB | Full RAG |
| **Power** | `microsoft/codebert-base` | 500MB | Same (no larger variant needed) |
| **Beast** | `microsoft/codebert-base` | 500MB | Same + cached in RAM |

**Why CodeBERT?**
- ✅ Code-specific embeddings (trained on code + docstrings)
- ✅ 768-dim vectors (good balance)
- ✅ Well-maintained by Microsoft
- ✅ Proven in code search use cases

**Alternatives Considered**:
- `sentence-transformers/all-MiniLM-L6-v2`: Too general (not code-focused)
- `Salesforce/codet5-base`: Larger, not optimized for embeddings
- OpenAI `text-embedding-ada-002`: ❌ Requires API (violates "local" promise)

**Status**: ✅ Finalized

### Vector Database: **LanceDB**

**Why LanceDB?**
- ✅ Pure Python (easy install)
- ✅ Disk-backed with fast memory mapping
- ✅ No server required (fits CLI-first approach)
- ✅ Good performance for <1M vectors

**Alternatives Considered**:
- ChromaDB: Requires SQLite, more heavyweight
- Qdrant: Requires server (overkill for local tool)
- FAISS: Low-level, no metadata support

**Status**: ✅ Finalized

**RAG Cache Sizes** (from README):
```
Survival: 200MB (10K chunks)
Comfort:  500MB (25K chunks)
Power:    2GB   (100K chunks)
Beast:    8GB   (entire index in RAM)
```

---

## 💾 RAM Detection Strategy

### Critical Decision: Use `psutil.virtual_memory().available`

```python
import psutil

def detect_profile():
    """Detect RAM profile based on AVAILABLE (free) memory."""
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)  # Bytes to GB
    
    # Profile thresholds based on AVAILABLE RAM
    if available_gb < 8:
        return ProfileType.SURVIVAL
    elif available_gb < 24:
        return ProfileType.COMFORT
    elif available_gb < 48:
        return ProfileType.POWER
    else:
        return ProfileType.BEAST
```

### Why `available` not `(total - used)`?

**Linux**:
- `available` = free + buffers + cache (reclaimable)
- More accurate than simple `total - used`

**macOS**:
- `available` = free + inactive memory
- Accounts for compressed memory

**Windows**:
- `available` = free + standby (reclaimable cache)

### Platform-Specific Adjustments

#### macOS (Apple Silicon)
```python
if platform == "darwin" and has_apple_silicon:
    # Reserve 2GB for Metal buffer pool
    available_gb -= 2.0
```

**Rationale**: macOS Metal keeps a dynamic buffer pool that `psutil` may count as "available" but isn't truly free.

#### Linux with Swap
```python
if swap_percent > 10:
    # System is swapping, be more conservative
    available_gb *= 0.8  # 20% penalty
```

**Rationale**: If swap is active, system is under pressure. Reduce our footprint.

### Safety Margins

Each profile reserves buffer to prevent OOM:

| Profile | Target Usage | Safety Buffer | Total Reserved |
|---------|--------------|---------------|----------------|
| Survival | ~5GB | 3GB | 8GB |
| Comfort | ~8.5GB | 7.5GB | 16GB |
| Power | ~17GB | 15GB | 32GB |
| Beast | ~40GB | 24GB | 64GB |

**Status**: ✅ Finalized, needs integration testing

---

## ⚙️ Quantization Strategy

### Profile-Based Quantization

| Profile | Quantization | Model Size | Quality | Speed |
|---------|--------------|------------|---------|-------|
| **Survival** | 4-bit (Q4_K_S) | 0.9GB | Good | ~3 tok/s |
| **Comfort** | 4-bit (Q4_K_M) | 1.0GB | Better | ~5 tok/s |
| **Power** | 8-bit (Q8_0) | 1.8GB | Best | ~8 tok/s |
| **Beast** | 8-bit (Q8_0) | 1.8GB | Best | ~12 tok/s* |

*Assumes GPU offloading or high RAM bandwidth

### GGUF Quant Variants

- **Q4_K_S**: Small, lowest quality (Survival)
- **Q4_K_M**: Medium, balanced (Comfort - default)
- **Q5_K_M**: Medium-high (optional upgrade)
- **Q8_0**: 8-bit, highest quality (Power/Beast)

**Decision**: We'll ship with Q4_K_M and Q8_0 variants pre-downloaded.

**Status**: ✅ Finalized

---

## 💻 Platform Support

### Target Platforms

| Platform | Tier | Backend | Support Level |
|----------|------|---------|---------------|
| **macOS (Apple Silicon)** | Tier 1 | MLX (primary) | Full, optimized |
| **macOS (Intel)** | Tier 1 | llama-cpp | Full |
| **Linux (x86_64)** | Tier 1 | llama-cpp | Full |
| **Linux (CUDA)** | Tier 1 | llama-cpp | Full, GPU-accelerated |
| **Windows (x86_64)** | Tier 1 | llama-cpp | Full |
| **ARM Linux (RPi)** | Tier 2 | llama-cpp | Community support |

**CI/CD Testing**:
- GitHub Actions: Ubuntu (primary), macOS (weekly)
- Test matrix: Python 3.11, 3.12
- Manual Windows testing (community-driven)

### Python Version

**Decision: Python >=3.11** ⚡

**Rationale**:
- ✅ **25% performance boost** vs 3.10 (critical for CPU inference!)
- ✅ Better error messages (improved debugging)
- ✅ Modern syntax: `|` unions, match statements
- ✅ **EOL: October 2027** (2+ years safe)
- ✅ Widely available (homebrew, apt, official installers)

**Code Quality Examples**:

```python
# Python 3.10 (old)
from typing import Union, Optional
def load_model(path: Optional[str]) -> Union[Model, None]:
    ...

# Python 3.11+ (clean)
def load_model(path: str | None) -> Model | None:
    ...
```

**Trade-off**: Some users on Python 3.9/3.10 excluded, but performance gain justifies it.

**Alternatives Considered**:
- ❌ Python 3.9: EOL Oct 2025 (too soon), no performance boost
- ❌ Python 3.10: No 25% speedup, EOL Oct 2026
- ❌ Python 3.12: Too new, less ecosystem adoption

**Status**: ✅ Finalized (Python >=3.11)

---

## 📦 Dependencies

### Core Dependencies

```toml
[tool.poetry.dependencies]
python = "^3.11"  # 25% performance boost

# LLM Backends
llama-cpp-python = "^0.2.0"  # Primary backend (CPU/CUDA)
mlx-lm = {version = "^0.1.0", optional = true, markers = "sys_platform == 'darwin' and platform_machine == 'arm64'"}

# ML & Embeddings
transformers = "^4.35.0"      # CodeBERT embeddings
torch = "^2.1.0"              # Transformers backend (CPU-only variant)
safetensors = "^0.4.0"        # Fast model loading

# Vector DB
lancedb = "^0.3.0"            # Disk-backed vector store

# System Detection
psutil = "^5.9.0"             # RAM/CPU/GPU detection

# CLI & UI
rich = "^13.7.0"              # Beautiful terminal UI
click = "^8.1.0"              # CLI framework
prompt-toolkit = "^3.0.0"    # Interactive REPL

# File & Data
watchdog = "^3.0.0"           # Ghost mode file watching
pyyaml = "^6.0.0"             # Config files
aiohttp = "^3.9.0"            # Async HuggingFace downloads

# Utilities
huggingface-hub = "^0.19.0"   # Model downloads
tqdm = "^4.66.0"              # Progress bars
```

### Development Dependencies

```toml
[tool.poetry.group.dev.dependencies]
# Testing
pytest = "^7.4.0"
pytest-cov = "^4.1.0"
pytest-asyncio = "^0.21.0"

# Code Quality
black = "^23.11.0"
ruff = "^0.1.6"
mypy = "^1.7.0"

# Git Hooks
pre-commit = "^3.5.0"

# Benchmarking
human-eval = "^1.0.0"  # HumanEval benchmark (we'll add this)
```

### Extras (GPU/Platform Support)

```toml
[tool.poetry.extras]
cuda = ["llama-cpp-python[cublas]"]  # NVIDIA GPU support
metal = ["mlx-lm"]                    # Apple Silicon optimization
all = ["llama-cpp-python[cublas]", "mlx-lm"]
```

### Installation Examples

```bash
# Standard (CPU only)
pip install quirkllm

# With CUDA support (NVIDIA GPU)
pip install quirkllm[cuda]

# With Metal support (Apple Silicon)
pip install quirkllm[metal]  # Auto-detected if on M1/M2/M3

# All backends
pip install quirkllm[all]
```

**Status**: ✅ Finalized

---

## 🧪 Testing Strategy

### MVP Testing Requirements

**Critical**: MVP MUST pass all benchmarks before release (as per user requirement).

### Test Pyramid

```
         /\
        /  \  E2E Tests (5%)
       /────\
      /      \ Integration Tests (15%)
     /────────\
    /          \ Unit Tests (80%)
   /────────────\
```

### Test Categories

#### 1. Unit Tests (80%)
**Location**: `tests/unit/`

**Coverage**:
```
tests/unit/
├── core/
│   ├── test_system_detector.py     # RAM detection (mock psutil)
│   ├── test_profile_manager.py     # Profile selection logic
│   ├── test_conversation_engine.py # Context management
│   └── test_compaction.py          # Message compaction
├── backends/
│   ├── test_llamacpp_backend.py    # llama-cpp interface
│   └── test_mlx_backend.py         # MLX interface (Mac only)
├── rag/
│   ├── test_lancedb.py             # Vector operations
│   ├── test_embeddings.py          # CodeBERT encoding
│   └── test_retrieval.py           # Semantic search
└── utils/
    ├── test_package_detector.py    # npm/yarn/pnpm detection
    └── test_file_operations.py     # Safe file I/O
```

**Target**: ≥90% line coverage

#### 2. Integration Tests (15%)
**Location**: `tests/integration/`

**Coverage**:
```
tests/integration/
├── test_cli_flow.py          # Full REPL session (startup → commands → quit)
├── test_conversation.py      # Multi-turn with context resolution
├── test_mode_switching.py    # Chat → YAMI → Plan → Ghost transitions
├── test_rag_pipeline.py      # Ingest → embed → retrieve → augment
└── test_backends.py          # Backend switching (llama-cpp ↔ MLX)
```

**Target**: 80% coverage of critical paths

#### 3. E2E Tests (5%)
**Location**: `tests/e2e/`

**Coverage**:
```
tests/e2e/
├── test_real_react_project.py    # Analyze actual React app
├── test_real_python_project.py   # Analyze actual Python project
├── test_memory_profiles.py       # Mock RAM: 8GB, 16GB, 32GB, 64GB
└── test_offline_mode.py          # No internet, local model only
```

### 4. Benchmark Tests (MVP Blockers!)
**Location**: `tests/benchmarks/`

**Critical**: MVP cannot ship until these pass.

```
tests/benchmarks/
├── test_humaneval.py
│   # Run HumanEval benchmark (164 problems)
│   # Target: ≥70% pass rate (README promise)
│   # Profiles: Test on Survival (8GB) and Comfort (16GB)
│
├── test_multiturn_accuracy.py
│   # 100 multi-turn conversation scenarios
│   # Target: ≥85% accuracy (README promise)
│   # Tests: Context resolution, reference tracking
│
├── test_ram_adaptation.py
│   # Mock RAM scenarios: 6GB, 12GB, 28GB, 60GB
│   # Verify correct profile selection
│   # Verify memory usage stays within limits
│
└── test_performance.py
    # Measure tokens/sec, latency, cache hit rate
    # Targets from README:
    #   Survival: ~3 tok/s
    #   Comfort: ~5 tok/s
    #   Power: ~8 tok/s
    #   Beast: ~12 tok/s
```

**Pass Criteria** (README Promises):

| Benchmark | Target | Blocking |
|-----------|--------|----------|
| **HumanEval** | ≥70% | ✅ YES |
| **Multi-turn Accuracy** | ≥85% | ✅ YES |
| **RAM Within Limits** | 100% | ✅ YES |
| **Context Resolution** | ≥90% | ✅ YES |
| **Response Time** | <10s (simple queries) | ⚠️ Advisory |

### CI/CD Pipeline

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    strategy:
      matrix:
        python: ["3.11", "3.12"]
        os: [ubuntu-latest, macos-latest]
    
    steps:
      - Unit tests (all)
      - Integration tests (all)
      - E2E tests (sample)
      - Benchmark tests (HumanEval subset - 20 problems)
      - Coverage report
      
  benchmark-full:
    # Weekly full benchmark (all 164 HumanEval problems)
    if: github.event_name == 'schedule'
    steps:
      - Full HumanEval (164 problems)
      - Multi-turn accuracy (100 scenarios)
      - Performance profiling
```

**Fail Conditions**:
- Any test failure
- Coverage drop >2%
- Lint errors (ruff, black, mypy)
- **Benchmark below targets** (HumanEval <70%, Multi-turn <85%)

### Test Data Creation

**HumanEval**:
```bash
# Install official dataset
pip install human-eval

# Run benchmark
pytest tests/benchmarks/test_humaneval.py --slow
```

**Multi-turn Dataset**: We'll create this
```
tests/benchmarks/data/
└── multiturn_scenarios.jsonl
    # 100 hand-crafted multi-turn conversations
    # Testing: context resolution, reference tracking, 
    #          framework awareness, package detection
```

**Status**: ✅ Strategy finalized, implementation in Phase 1

---

## 📦 PyPI Package Strategy

### Package Metadata

```toml
[tool.poetry]
name = "quirkllm"
version = "0.1.0"  # Beta (MVP includes all features but marking as beta for safety)
description = "Local, free, GPU-optional AI coding assistant. Claude CLI alternative."
authors = ["Yamac Bezirgan <your@email.com>"]
license = "Apache-2.0"  # Matches DeepSeek-Coder base + more permissive
readme = "README.md"
homepage = "https://github.com/ymcbzrgn/QuirkLLM"
repository = "https://github.com/ymcbzrgn/QuirkLLM"
documentation = "https://github.com/ymcbzrgn/QuirkLLM#readme"
keywords = [
    "llm",
    "coding-assistant",
    "local-ai",
    "offline",
    "deepseek",
    "cli",
    "code-generation"
]

classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Topic :: Software Development :: Code Generators",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Operating System :: OS Independent",
    "Environment :: Console",
]

[tool.poetry.scripts]
quirkllm = "quirkllm.__main__:main"
```

### Versioning Strategy

**Decision**: Start with `0.1.0` (Beta)

**Rationale**:
- MVP is feature-complete BUT marking as beta gives flexibility
- Breaking changes allowed in 0.x versions
- 1.0.0 becomes a major milestone (signals production-ready)

**Version Roadmap**:
```
0.1.0 → MVP Release
        • All 4 modes (Chat/YAMI/Plan/Ghost)
        • Fine-tuned QuirkLLM-1.3B
        • Full RAG system
        • All benchmarks passing

0.2.0 → Stability improvements
        • Bug fixes from community feedback
        • Performance optimizations

0.3.0 → Enhanced features
        • Better error messages
        • Improved context handling

1.0.0 → Production Release
        • Battle-tested
        • Documentation complete
        • Community-proven
```

### Package Distribution

**Wheel Contents**: Code only (~50MB)
```
quirkllm-0.1.0-py3-none-any.whl
├── quirkllm/
│   ├── __init__.py
│   ├── __main__.py
│   ├── core/
│   ├── backends/
│   ├── utils/
│   └── rag/
├── LICENSE
└── README.md
```

**Models**: Downloaded separately (900MB-1.4GB)
- First run: Auto-download from HuggingFace
- Stored in: `~/.quirkllm/models/`

### Installation Commands

```bash
# Basic installation
pip install quirkllm

# With CUDA (NVIDIA GPU)
pip install quirkllm[cuda]

# With Metal (Apple Silicon - auto-detected)
pip install quirkllm[metal]

# Development
git clone https://github.com/ymcbzrgn/QuirkLLM
cd QuirkLLM
poetry install
```

### PyPI Upload Process

```bash
# Build
poetry build

# Publish to TestPyPI first
poetry publish -r testpypi

# Test installation
pip install --index-url https://test.pypi.org/simple/ quirkllm

# If OK, publish to PyPI
poetry publish
```

**Status**: ✅ Finalized

---

## 🎯 MVP Scope (Phase 1)

### Non-Negotiable Requirements

**User Requirement**: MVP must be CORE-COMPLETE. All 4 modes + fine-tuned model + pass all benchmarks.

#### Must Have ✅

| Feature | Status | Blocker |
|---------|--------|---------|
| **Fine-tuned Model** | QuirkLLM-1.3B (not base) | ✅ YES |
| **4 Modes** | Chat / YAMI / Plan / Ghost | ✅ YES |
| **RAM Profiles** | All 4 (Survival/Comfort/Power/Beast) | ✅ YES |
| **Full RAG** | LanceDB + CodeBERT + all 18 components | ✅ YES |
| **Project Analyzer** | Framework + package manager detection | ✅ YES |
| **HumanEval ≥70%** | Benchmark passing | ✅ YES |
| **Multi-turn ≥85%** | Benchmark passing | ✅ YES |
| **Hybrid Backend** | llama-cpp + MLX | ✅ YES |
| **Auto-download** | HF model download | ✅ YES |
| **Session Save/Load** | Persistence | ✅ YES |

#### Deferred to Later Phases ⏭️

| Feature | Deferred To | Reason |
|---------|-------------|--------|
| **MCP Server** | Phase 5 | Not core to CLI functionality |
| **VS Code Extension** | Post-CLI | Wrapper, no core logic |
| **GUI Application** | Post-CLI | Wrapper, no core logic |
| **HF Manifest** | Future | Can hardcode model URLs initially |

### Timeline Estimate

**Phase 1 (MVP)**: 30 weeks (as per ROADMAP.md)
```
Alt-Faz A: System Detection       (2 weeks)
Alt-Faz B: Core CLI               (3 weeks)
Alt-Faz C: Project Analyzer       (2 weeks)
Alt-Faz D: Conversation Engine    (4 weeks)
Alt-Faz E: Model Integration      (5 weeks)
Alt-Faz F: RAG System             (5 weeks)
Alt-Faz G: Fine-tuning            (6 weeks)  ← CRITICAL
Alt-Faz H: Output & Polish        (3 weeks)
```

**Benchmark Testing**: 2 weeks (parallel with Alt-Faz H)

**Total**: ~32 weeks (~8 months)

**Status**: ✅ Scope finalized

---

---

## 📝 Decision Log

### Phase 0 Decisions (2025-11-26)

| # | Category | Decision | Rationale | Status |
|---|----------|----------|-----------|--------|
| 1 | **Base Model** | DeepSeek-Coder-1.3B-base | 1.3B params, MIT license, 16K context, strong code perf | ✅ Final |
| 2 | **Fine-tuned Model** | QuirkLLM-1.3B (MVP Phase 1) | Fine-tune with Axolotl, 30 D-components, credits DeepSeek | ✅ Final |
| 3 | **LLM Backend** | **Hybrid**: llama-cpp + mlx | Optimal perf on all platforms (Mac Metal + others) | ✅ Final |
| 4 | **Python Version** | **>=3.11** | **25% performance boost**, modern syntax, EOL 2027 | ✅ Final |
| 5 | **Quantization** | **Profile-based**: Q4_K_M / Q8_0 | 8GB→Q4, 32GB→Q8 (automatic) | ✅ Final |
| 6 | **Embedding** | CodeBERT (small/base/large) | Code-specific, profile-adaptive | ✅ Final |
| 7 | **Vector DB** | LanceDB | Disk-backed, no server, Python-native | ✅ Final |
| 8 | **Model Source** | HuggingFace Hub | Free hosting, CDN, resume downloads | ✅ Final |
| 9 | **Model Download** | Auto-download + manual fallback | First run auto, offline `--model-path` | ✅ Final |
| 10 | **Update Strategy** | Manual `/update` command | No auto-check on startup | ✅ Final |
| 11 | **Project Root** | `~/.quirkllm/` | Config, models, cache, sessions, plans | ✅ Final |
| 12 | **Cache Cleanup** | Size-based (10GB limit) + manual | LRU auto-cleanup when exceeded | ✅ Final |
| 13 | **Log Rotation** | 50MB max, 3 backups | Prevents disk bloat | ✅ Final |
| 14 | **Package Name** | `quirkllm` (PyPI) | Clean, matches README | ✅ Final |
| 15 | **Initial Version** | `0.1.0` (Beta) | Feature-complete MVP, beta for flexibility | ✅ Final |
| 16 | **License** | **Apache 2.0** | Permissive, commercial-friendly | ✅ Final |
| 17 | **MVP Scope** | All 4 modes + fine-tuned + benchmarks | Core-complete, no wrappers yet | ✅ Final |
| 18 | **MCP Server** | Phase 5 (deferred) | Not blocking CLI core | ✅ Final |

### Key Trade-offs Accepted

| Trade-off | Decision | Why Worth It |
|-----------|----------|--------------|
| Python 3.11 minimum | Exclude 3.9/3.10 users | **25% performance** critical for CPU inference |
| Hybrid backend | Maintain 2 backends | Optimal performance on every platform |
| 0.1.0 start | Beta label despite full features | Flexibility for breaking changes |
| Fine-tune in MVP | Longer development | Quality/benchmarks non-negotiable |
| No auto-update | Manual `/update` only | User control, no surprise bandwidth |

---

## 🔄 Next Steps

### Immediate (This Week)

- [x] **TECHNICAL_DECISIONS.md** complete
- [ ] Create `pyproject.toml` with all dependencies
- [ ] Setup project structure (`quirkllm/core`, `quirkllm/utils`, etc.)
- [ ] Implement system detector (RAM, GPU, platform)
- [ ] Create backend abstraction layer

### Phase 1 Kickoff (Week 1-2)

- [ ] Implement llama-cpp backend
- [ ] Implement MLX backend (macOS only)
- [ ] Build CLI entry point (`quirkllm/__main__.py`)
- [ ] Setup Rich terminal UI
- [ ] Implement profile manager

### Phase 1 Fine-tuning (Week 19-24)

- [ ] Create multi-turn conversation dataset
- [ ] Setup Axolotl training environment
- [ ] Train QuirkLLM-1.3B (30 D-components)
- [ ] Upload to HuggingFace (ymcbzrgn/QuirkLLM-1.3B-GGUF)
- [ ] Quantize (Q4_K_M, Q8_0)

### Phase 1 Benchmarking (Week 25-26)

- [ ] Run HumanEval (target: ≥70%)
- [ ] Run multi-turn accuracy tests (target: ≥85%)
- [ ] RAM adaptation tests (all 4 profiles)
- [ ] Performance profiling (tok/s per profile)

### MVP Release (Week 30)

- [ ] All benchmarks passing
- [ ] Documentation complete
- [ ] Publish to PyPI (`quirkllm==0.1.0`)
- [ ] Announce 🎉

---

## 🎓 Lessons Learned

### What Went Well

- ✅ **Thorough planning**: All major decisions finalized before coding
- ✅ **README alignment**: Every decision references README spec
- ✅ **User collaboration**: Discussing trade-offs led to better choices
- ✅ **Performance focus**: 25% Python speedup, profile-based quant

### Potential Risks

- ⚠️ **Fine-tuning complexity**: 6 weeks estimated, could slip
- ⚠️ **Benchmark targets**: 70% HumanEval is ambitious for 1.3B model
- ⚠️ **Hybrid backend**: Testing 2 backends increases CI/CD complexity
- ⚠️ **Python 3.11**: Some users stuck on 3.9/3.10 will complain

### Mitigation Plans

- 📋 **Fine-tuning**: Start dataset creation early (parallel with Phase 1-C)
- 📋 **Benchmarks**: If base model <50% HumanEval, adjust target to 60%
- 📋 **Testing**: Setup Mac + Linux CI from day 1
- 📋 **Python version**: Clear documentation + error message for <3.11

---

## 📚 References

### External Documents

- [DeepSeek-Coder Model Card](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base)
- [llama-cpp-python Docs](https://llama-cpp-python.readthedocs.io/)
- [MLX LM Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [LanceDB Python SDK](https://lancedb.github.io/lancedb/)
- [HumanEval Benchmark](https://github.com/openai/human-eval)
- [Axolotl Training](https://github.com/OpenAccess-AI-Collective/axolotl)

### Internal Documents

- `README.md` - Product specification (source of truth)
- `ROADMAP.md` - 30-week development plan
- `.github/copilot-instructions.md` - AI collaboration guidelines
- `.github/instructions/codacy.instructions.md` - Code quality rules

---

## ✅ Sign-off

**Phase 0: Technical Foundation** is **COMPLETE**.

All critical technical decisions have been made and documented. The team can proceed to Phase 1 implementation with confidence.

**Key Achievements**:
- ✅ Model strategy: DeepSeek-Coder base → QuirkLLM-1.3B fine-tuned
- ✅ Backend: Hybrid (llama-cpp + MLX) for optimal cross-platform performance
- ✅ Platform: Python 3.11+ for 25% performance boost
- ✅ Distribution: PyPI `quirkllm==0.1.0` with HuggingFace model hosting
- ✅ MVP scope: Core-complete with all 4 modes + benchmarks

**Next Milestone**: Phase 1 Alt-Faz A (System Detection) - 2 weeks

---

*Document Status: ✅ Finalized*  
*Last Updated: 2025-11-26*  
*Version: 1.0*

---

**This document is the single source of truth for "why" we made technical choices. When in doubt, refer to this first, then README.md.**
