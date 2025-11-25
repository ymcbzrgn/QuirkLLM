# Contributing to QuirkLLM

Thanks for helping build a local-first coding companion! This guide explains how to plan work, keep the docs truthful, and land safe changes.

## 🧭 Core Principles

- **README is the source of truth.** Every feature, RAM profile tweak, command, or benchmark must be reflected there before or alongside code changes.
- **CLI-first architecture.** All runtime logic lives in `quirkllm/core` and helpers in `quirkllm/utils`. IDE or GUI layers must shell out to the CLI.
- **RAM-aware behavior is mandatory.** When touching inference, caching, or context logic, keep the four profiles (Survival, Comfort, Power, Beast) aligned with the tables in the README.
- **Documentation mirrors code.** Update `ROADMAP.md` when milestones move and mention changes in the relevant README sections (RAM profiles, modes, CLI commands, etc.).

## 🚀 Getting Started

1. **Fork & clone** the repository.
2. **Create a virtual environment** (Python 3.10+ is recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```
3. **Install tooling** required for your change (e.g., Codacy CLI is triggered automatically via MCP, pytest for tests).
4. **Create a feature branch**: `git checkout -b feat/profile-compaction`.

## 🔄 Workflow Checklist

1. **Open/assign an issue** describing the change and desired RAM profile impact.
2. **Design first**: if the change spans multiple components, capture the plan in `.quirkllm/plans/<topic>.md` (Plan Mode) or the issue.
3. **Implement in the CLI pipeline** (System Detector → Profile Manager → CLI → Analyzer → Conversation Engine → Adaptive Inference → Adaptive RAG → Output Handler). Don’t bypass stages.
4. **Keep package operations smart.** Use the prioritized package-manager detection (bun → pnpm → yarn → npm; poetry → pipenv → pip) and log commands like the CLI demo in the README.
5. **Synchronize docs**:
   - Update the relevant README sections (RAM tables, mode descriptions, CLI command list, etc.).
   - Tick or annotate `ROADMAP.md` milestones when features ship.
6. **Add or update tests** under `tests/`. Prefer CLI-end-to-end scenarios (e.g., `/status`, `learn --url`).
7. **Run the suite**:
   ```bash
   pytest
   quirkllm --test
   ```
8. **Codacy analysis**: after every edit, run `codacy_cli_analyze` for each touched file via the MCP command palette. For dependency changes, also run it with `tool=trivy`.
9. **Open a pull request** referencing the issue and include the PR checklist (below).

## 🧪 Quality Bar

- All tests green (`pytest`, `quirkllm --test`).
- No lint errors or typing regressions (add `ruff`/`mypy` outputs if applicable).
- RAM-profile tables remain accurate (context, cache sizes, compaction rules).
- CLI command reference matches actual behavior.
- Codacy CLI reports no blocking issues.

## 📄 Documentation Expectations

- For any behavior change, update:
  - `README.md` (spec + screenshots + sample flows)
  - `ROADMAP.md` (if milestones shift)
  - Any relevant guide (e.g., new slash command → CLI table)
- Mention new docs (CONTRIBUTING, CODE_OF_CONDUCT, etc.) inside the README index if needed.

## 🔐 Security & Privacy

- Never commit secrets, API keys, or proprietary datasets.
- Favor local/offline flows—QuirkLLM is privacy-first.
- If you handle user data or telemetry, document what is stored and provide opt-outs.

## ✅ Pull Request Checklist

- [ ] Issue linked and scope agreed.
- [ ] README/ROADMAP updated.
- [ ] Tests added/updated (`pytest`, `quirkllm --test`).
- [ ] Codacy CLI run for each edited file + `tool=trivy` after dependency changes.
- [ ] No lint/type errors.
- [ ] Feature adheres to CLI-first + RAM profile contracts.

Questions? Open a discussion or ping @maintainers in the issue tracker. Happy hacking! 🛠️
