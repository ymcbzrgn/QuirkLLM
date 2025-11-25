# 🗺️ QuirkLLM Flexible Roadmap

This roadmap outlines the development journey of QuirkLLM. It is designed to be flexible, allowing for "Discovery & Fix" cycles where we address unforeseen challenges and optimize as we go.

## 🗓️ Phase 1: Foundation & Core CLI (The Skeleton)
**Goal:** Build the CLI structure, system detection, and basic user interaction loop.

- [ ] **1.1 Project Scaffolding**
    - [ ] Set up Python project structure (poetry/pip).
    - [ ] Configure linting, formatting, and testing tools.
- [ ] **1.2 System Detector (The Senses)**
    - [ ] Implement `psutil` integration for RAM/CPU detection.
    - [ ] **Critical:** Implement "Available RAM" logic (not just Total RAM).
    - [ ] Detect GPU (CUDA/Metal) availability.
- [ ] **1.3 Profile Manager (The Brain Stem)**
    - [ ] Define `Survival`, `Comfort`, `Power`, `Beast` profiles.
    - [ ] Implement logic to select profile based on system stats.
    - [ ] Allow manual overrides via CLI args (`--profile`).
- [ ] **1.4 Interactive CLI (The Face)**
    - [ ] Build the REPL (Read-Eval-Print Loop).
    - [ ] Implement Rich Terminal UI (colors, spinners, tables).
    - [ ] Handle basic commands (`/help`, `/status`, `/quit`).
- [ ] **🔄 Discovery & Fix Cycle 1**
    - [ ] Test on different simulated RAM environments.
    - [ ] Refine UI/UX based on initial feel.

## 🧠 Phase 2: The Intelligence (Model & Inference)
**Goal:** Integrate the LLM and make it run locally with RAM-aware optimizations.

- [ ] **2.1 Model Engine**
    - [ ] Integrate `llama-cpp-python` or similar for GGUF support.
    - [ ] Implement model downloading/loading logic.
    - [ ] **Feature:** "Model Foundry" support (custom .gguf loading).
- [ ] **2.2 RAM-Aware Inference**
    - [ ] Implement dynamic context length calculation.
    - [ ] Configure quantization (4-bit vs 8-bit) based on profile.
    - [ ] Implement KV-cache management strategies.
- [ ] **2.3 Hybrid Inference**
    - [ ] Implement GPU offloading logic (if GPU detected).
    - [ ] Fallback mechanisms for pure CPU environments.
- [ ] **🔄 Discovery & Fix Cycle 2**
    - [ ] Benchmark inference speed on CPU vs GPU.
    - [ ] Optimize memory usage to prevent swapping.

## 📂 Phase 3: Project Awareness (The Context)
**Goal:** Make QuirkLLM understand the user's codebase.

- [ ] **3.1 Project Analyzer**
    - [ ] Detect frameworks (React, Django, etc.).
    - [ ] Detect package managers (npm, yarn, pnpm, bun, poetry).
    - [ ] Map file structure.
- [ ] **3.2 File Operations**
    - [ ] Implement safe file reading/writing.
    - [ ] Implement diff generation for code changes.
- [ ] **3.3 RAG System (The Memory)**
    - [ ] Integrate LanceDB for vector storage.
    - [ ] Implement CodeBERT embedding generation.
    - [ ] Build the retrieval pipeline.
- [ ] **🔄 Discovery & Fix Cycle 3**
    - [ ] Test with large monorepos.
    - [ ] Refine RAG relevance and speed.

## 🎮 Phase 4: The Modes (The Personality)
**Goal:** Implement the different interaction modes and state management.

- [ ] **4.1 Mode State Machine**
    - [ ] Implement switching logic (`Shift+Tab` simulation/command).
    - [ ] Visual indicators for current mode in CLI.
- [ ] **4.2 Chat Mode (Default)**
    - [ ] Standard "Ask & Confirm" flow.
- [ ] **4.3 YAMI Mode (Yamac/YOLO)**
    - [ ] Implement "Auto-Accept" logic.
    - [ ] Bypass confirmation prompts for non-destructive actions.
- [ ] **4.4 Plan Mode (Architect)**
    - [ ] Implement "Read-Only" constraint.
    - [ ] Logic to generate `.quirkllm/plans/TODO.md`.
- [ ] **🔄 Discovery & Fix Cycle 4**
    - [ ] Ensure YAMI mode doesn't destroy the system (`rm -rf` checks).
    - [ ] Verify Plan mode produces useful outputs.

## 👻 Phase 5: Advanced Capabilities (The Superpowers)
**Goal:** Implement the "Killer Features" that set QuirkLLM apart.

- [ ] **5.1 Ghost Mode (Watcher)**
    - [ ] Implement file watcher (watchdog).
    - [ ] Background analysis logic on file save.
    - [ ] Unobtrusive notification system.
- [ ] **5.2 Knowledge Eater**
    - [ ] Implement web crawler for documentation (`--url`).
    - [ ] Implement PDF parser (`--pdf`).
    - [ ] Pipeline to feed this data into the local RAG.
- [ ] **5.3 Neural Link (MCP Server)**
    - [ ] Implement MCP protocol support.
    - [ ] Expose tools and resources via WebSocket/Stdio.
- [ ] **🔄 Discovery & Fix Cycle 5**
    - [ ] Test Ghost mode performance impact.
    - [ ] Verify MCP compatibility with Claude Desktop.

## 🚀 Phase 6: Polish & Release (The Launch)
**Goal:** Prepare for public release.

- [ ] **6.1 Testing**
    - [ ] Run the full Test Suite (HumanEval, Multi-turn).
    - [ ] Verify "RAM-Aware" behavior across profiles.
- [ ] **6.2 Documentation**
    - [ ] Finalize README.md.
    - [ ] Create user guides for each mode.
- [ ] **6.3 Packaging**
    - [ ] Build pip package.
    - [ ] Build Docker image.
- [ ] **6.4 Launch**
    - [ ] Publish to PyPI.
    - [ ] Announce to the world!

---
*This roadmap is a living document. We will update it as we learn and evolve.*
