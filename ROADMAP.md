# 🗺️ QuirkLLM Flexible Roadmap

> **Purpose**: High-level milestone tracking for QuirkLLM development.  
> **Philosophy**: Flexible, iterative, "Discovery & Fix" approach.  
> **Detail Level**: Phase overview + checkboxes (implementation details in `docs/PHASE*.md`).

**Last Updated**: 2025-11-26  
**Current Status**: Phase 0 Complete ✅ | Phase 1 Starting 🚀

---

## 📖 How to Use This Roadmap

- **This file**: High-level phases and completion tracking
- **`docs/PHASE*.md`**: Detailed implementation plans with code examples
- **Check boxes**: Updated as work progresses (one checkbox = one deliverable)
- **Discovery cycles**: Built into each phase for course correction

## ✅ Phase 0: Technical Foundation
**Duration:** 1 week  
**Status:** ✅ COMPLETE (2025-11-26)  
**Goal:** Finalize all technical decisions before writing code.

### Deliverables
- [x] **Research & Decision Making**
  - [x] Model selection (DeepSeek-Coder-1.3B-base → QuirkLLM-1.3B)
  - [x] Backend strategy (Hybrid: llama-cpp + MLX)
  - [x] Python version (>=3.11 for 25% performance boost)
  - [x] Quantization strategy (Profile-based: Q4/Q8)
  - [x] Model distribution (HuggingFace Hub)
  - [x] Project structure (~/.quirkllm/)
  - [x] PyPI strategy (quirkllm==0.1.0, Apache 2.0)

- [x] **Documentation**
  - [x] TECHNICAL_DECISIONS.md (comprehensive tech spec)
  - [x] README.md alignment verification
  - [x] Decision log with 18 finalized choices

### Key Learnings
- ✅ Python 3.11's 25% boost justifies breaking 3.9/3.10 compatibility
- ✅ Hybrid backend complexity worth optimal cross-platform performance
- ✅ MVP requires fine-tuned model (quality over speed-to-market)

---

## 🏗️ Phase 1: Foundation & Core CLI (The Skeleton)
**Duration:** 2 weeks  
**Status:** 🚀 IN PROGRESS (Started: 2025-11-26)  
**Goal:** Build the CLI structure, system detection, and basic user interaction loop.  
**Details:** `docs/PHASE1_PLAN.md`

### Week 1: Foundation (Day 1-7)
- [x] **1.1 Project Scaffolding (Day 1-2)**
    - [x] Set up Python project structure with Poetry
    - [x] Configure linting tools (Black, Ruff, Mypy)
    - [x] Set up testing infrastructure (Pytest, coverage)
    - [x] Pre-commit hooks and quality control script

- [x] **1.2 System Detector (Day 3-5)**
    - [x] Implement `psutil` integration for RAM/CPU detection
    - [x] **Critical:** Implement "Available RAM" logic (not just Total RAM)
    - [x] Detect GPU (CUDA/Metal) availability
    - [x] Platform detection with comprehensive tests (14 tests, 100% coverage)

- [x] **1.3 Profile Manager (Day 3-5)**
    - [x] Define `Survival`, `Comfort`, `Power`, `Beast` profiles
    - [x] Implement logic to select profile based on system stats
    - [x] Allow manual overrides via CLI args (`--profile`)
    - [x] Comprehensive tests (22 tests, 100% coverage)

### Week 2: CLI & Integration (Day 8-14)
- [x] **1.4 Interactive CLI (Day 8-10)** ✅ COMPLETED
    - [x] Build the REPL (Read-Eval-Print Loop)
    - [x] Implement Rich Terminal UI (colors, spinners, tables)
    - [x] Handle basic commands (`/help`, `/status`, `/quit`)
    - [x] Command history with prompt_toolkit
    - [x] Command aliases support (?, h, info, stat, exit, q)
    - [x] Graceful Ctrl+C / Ctrl+D handling

- [ ] **1.5 Configuration System (Day 11-12)** 🔄 NEXT
    - [ ] Config file structure (~/.quirkllm/config.yaml)
    - [ ] User override support and profile persistence
    - [ ] Default config generation on first run
    - [ ] Config validation and migration

- [ ] **1.6 Backend Abstraction (Day 11-12)** 🔄 NEXT
    - [ ] Abstract backend interface
    - [ ] Placeholder backend (mock responses)
    - [ ] Backend factory pattern

- [ ] **🔄 Discovery & Fix Cycle 1 (Day 13-14)**
    - [ ] Test on different simulated RAM environments
    - [ ] Refine UI/UX based on initial feel
    - [ ] Integration tests for full CLI flow
    - [ ] Cross-platform testing (Linux VM)

### Progress: 71% Complete (10/14 days)
**Week 1 Complete! (Day 1-10)**
- ✅ Project scaffolding with Poetry
- ✅ System detector (162 lines, 15 tests)
- ✅ Profile manager with platform-aware logic (170 lines, 25 tests)
- ✅ CLI entry point with Click (__main__.py, 140 lines)
- ✅ Interactive REPL with Rich UI (repl.py, 261 lines)
- ✅ Platform-aware RAM detection strategy implemented
- ✅ 41 passing tests (0.12s), 100% coverage for core modules
- ✅ Zero Codacy issues, all quality checks green

**Week 2 Remaining (Day 11-14):**
- ⏭️ Configuration system
- ⏭️ Backend abstraction layer
- ⏭️ Integration tests
- ⏭️ Discovery & Fix Cycle 1
## 🧠 Phase 2: Model & Inference
**Duration:** 2 weeks  
**Status:** ⏳ PENDING  
**Goal:** Integrate LLM and implement RAM-aware inference.  
**Details:** `docs/PHASE2_PLAN.md` (will be created when Phase 1 completes)

### Key Deliverables
- [ ] **Model Engine**
  - [ ] HuggingFace downloader (resume support)
  - [ ] llama-cpp-python integration
  - [ ] MLX integration (macOS)
  - [ ] Model loading with profile-based quantization

- [ ] **Inference Pipeline**
  - [ ] Dynamic context length calculation
  - [ ] KV-cache management
  - [ ] GPU offload logic (if available)
  - [ ] Streaming response support

- [ ] **First Working Inference**
  - [ ] Simple prompt → response working
  - [ ] Multi-turn conversation support
  - [ ] Context window tracking

- [ ] **🔄 Discovery & Fix Cycle**
  - [ ] Benchmark CPU vs GPU performance
  - [ ] Memory usage optimization
  - [ ] Latency profiling

### Success Criteria
- ✅ Can download and load model automatically
- ✅ Inference works on all platforms
- ✅ RAM usage stays within profile limits
- ✅ Acceptable response speed (>3 tok/s)

## 📂 Phase 3: Project Awareness & RAG
**Duration:** 2 weeks  
**Status:** ⏳ PENDING  
**Goal:** Project analysis and RAG system implementation.  
**Details:** `docs/PHASE3_PLAN.md` (will be created)

### Key Deliverables
- [ ] **Project Analyzer**
  - [ ] Framework detection (React, Django, Next.js, etc.)
  - [ ] Package manager detection (npm/yarn/pnpm/bun/poetry)
  - [ ] File structure mapping
  - [ ] Dependency graph analysis

- [ ] **RAG System**
  - [ ] LanceDB integration
  - [ ] CodeBERT embeddings
  - [ ] Retrieval pipeline with reranking
  - [ ] Hybrid semantic + keyword search

- [ ] **File Operations**
  - [ ] Safe file read/write
  - [ ] Diff generation
  - [ ] Multi-file editing support

- [ ] **🔄 Discovery & Fix Cycle**
  - [ ] Test with large monorepos (>1000 files)
  - [ ] RAG relevance tuning
  - [ ] Performance optimization

### Success Criteria
- ✅ Correctly detects common frameworks
- ✅ RAG retrieves relevant code snippets
- ✅ File operations work safely
- ✅ Works with projects >10,000 files

## 🎮 Phase 4: The Four Modes
**Duration:** 2 weeks  
**Status:** ⏳ PENDING  
**Goal:** Implement Chat, YAMI, Plan, and Ghost modes.  
**Details:** `docs/PHASE4_PLAN.md` (will be created)

### Key Deliverables
- [ ] **Mode State Machine**
  - [ ] Mode switching logic (`/mode` command)
  - [ ] Visual indicators in REPL
  - [ ] Mode persistence across sessions

- [ ] **Chat Mode** (Default)
  - [ ] Ask & confirm flow
  - [ ] Safe edit confirmations

- [ ] **YAMI Mode** (Auto-accept)
  - [ ] Auto-accept non-destructive actions
  - [ ] Safety checks (`rm -rf` blocking)
  - [ ] Undo mechanism

- [ ] **Plan Mode** (Read-only)
  - [ ] Read-only constraint
  - [ ] Generate `.quirkllm/plans/*.md`
  - [ ] Architecture documentation

- [ ] **🔄 Discovery & Fix Cycle**
  - [ ] Safety testing for YAMI mode
  - [ ] Plan quality validation
  - [ ] Mode switching UX refinement

### Success Criteria
- ✅ All 4 modes functional
- ✅ YAMI mode safe (no destructive actions)
- ✅ Plan mode generates useful docs
- ✅ Mode switching smooth and intuitive

## 👻 Phase 5: Advanced Features
**Duration:** 3 weeks  
**Status:** ⏳ PENDING  
**Goal:** Ghost mode, Knowledge Eater, and MCP server.  
**Details:** `docs/PHASE5_PLAN.md` (will be created)

### Key Deliverables
- [ ] **Ghost Mode** (File watcher)
  - [ ] File watcher integration (watchdog)
  - [ ] Background analysis on save
  - [ ] Unobtrusive notifications
  - [ ] Performance impact analysis

- [ ] **Knowledge Eater**
  - [ ] Web crawler (`/learn --url`)
  - [ ] PDF parser (`/learn --pdf`)
  - [ ] Documentation ingestion pipeline
  - [ ] RAG index updates

- [ ] **Neural Link** (MCP Server)
  - [ ] MCP protocol implementation
  - [ ] Tool exposure (WebSocket/Stdio)
  - [ ] Claude Desktop compatibility
  - [ ] Resource management

- [ ] **🔄 Discovery & Fix Cycle**
  - [ ] Ghost mode CPU/RAM impact
  - [ ] MCP integration testing
  - [ ] Knowledge ingestion quality

### Success Criteria
- ✅ Ghost mode watches files without lag
- ✅ Can ingest documentation from web/PDF
- ✅ MCP server works with Claude Desktop
- ✅ Minimal performance overhead

## 🚀 Phase 6: Fine-Tuning & Benchmarks
**Duration:** 2 weeks  
**Status:** ⏳ PENDING  
**Goal:** Train QuirkLLM-1.3B and pass benchmark targets.  
**Details:** `docs/PHASE6_PLAN.md` (will be created)

### Key Deliverables
- [ ] **Model Fine-Tuning**
  - [ ] Prepare training dataset (30 D-components)
  - [ ] Axolotl training pipeline
  - [ ] Train QuirkLLM-1.3B from DeepSeek-Coder-1.3B-base
  - [ ] Upload to HuggingFace Hub

- [ ] **Benchmark Testing**
  - [ ] HumanEval evaluation (target: ≥70%)
  - [ ] Multi-turn conversation accuracy (target: ≥85%)
  - [ ] RAM-aware profile verification
  - [ ] Cross-platform testing (macOS/Linux/Windows)

- [ ] **🔄 Discovery & Fix Cycle**
  - [ ] Hyperparameter tuning
  - [ ] Dataset quality refinement
  - [ ] Benchmark failure analysis

### Success Criteria
- ✅ HumanEval score ≥70% (vs base ~34%)
- ✅ Multi-turn accuracy ≥85%
- ✅ All 4 RAM profiles work correctly
- ✅ Model uploaded to HuggingFace

---

## 🎉 Phase 7: Polish & Launch
**Duration:** 1 week  
**Status:** ⏳ PENDING  
**Goal:** Final polish and public release.

### Key Deliverables
- [ ] **Documentation**
  - [ ] Finalize README.md
  - [ ] User guides for each mode
  - [ ] API documentation
  - [ ] Video demo

- [ ] **Packaging**
  - [ ] PyPI package (`quirkllm==0.1.0`)
  - [ ] Docker image
  - [ ] Installation testing (Windows/macOS/Linux)

- [ ] **Launch**
  - [ ] Publish to PyPI
  - [ ] GitHub release with binaries
  - [ ] Social media announcement
  - [ ] Blog post / documentation site

### Success Criteria
- ✅ `pip install quirkllm` works
- ✅ Docker image available
- ✅ All documentation complete
- ✅ Public release successful

---

## 📊 Progress Tracking

| Phase | Status | Start | End | Duration | Progress |
|-------|--------|-------|-----|----------|----------|
| Phase 0: Technical Foundation | ✅ Complete | - | 2025-11-26 | 1 week | 100% |
| Phase 1: Foundation & CLI | 🚀 In Progress | 2025-11-26 | TBD | 2 weeks | **71%** (10/14 days) |
| Phase 2: Model & Inference | ⏳ Pending | TBD | TBD | 2 weeks | 0% |
| Phase 3: Project & RAG | ⏳ Pending | TBD | TBD | 2 weeks | 0% |
| Phase 4: Four Modes | ⏳ Pending | TBD | TBD | 2 weeks | 0% |
| Phase 5: Advanced Features | ⏳ Pending | TBD | TBD | 3 weeks | 0% |
| Phase 6: Fine-Tuning | ⏳ Pending | TBD | TBD | 2 weeks | 0% |
| Phase 7: Polish & Launch | ⏳ Pending | TBD | TBD | 1 week | 0% |

**Total Estimated Duration:** ~14 weeks (~3.5 months)

---

*This roadmap is a living document. Updated as we learn and evolve.*
