"""
Smart Package Manager Detection for QuirkLLM

Detects package managers and extracts dependencies from various ecosystems:
- JavaScript/TypeScript: bun, pnpm, yarn, npm
- Python: poetry, pipenv, pip
- Go: go modules
- Rust: cargo
- Java: maven, gradle
- PHP: composer
"""

import json
import tomli
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set


class PackageManager(Enum):
    """Supported package managers."""

    # JavaScript/TypeScript
    BUN = "bun"
    PNPM = "pnpm"
    YARN = "yarn"
    NPM = "npm"

    # Python
    POETRY = "poetry"
    PIPENV = "pipenv"
    PIP = "pip"

    # Other languages
    GO_MODULES = "go"
    CARGO = "cargo"
    MAVEN = "maven"
    GRADLE = "gradle"
    COMPOSER = "composer"

    # Unknown/None
    UNKNOWN = "unknown"


@dataclass
class PackageInfo:
    """Package manager and dependency information."""

    manager: PackageManager
    dependencies: Dict[str, str] = field(default_factory=dict)
    dev_dependencies: Dict[str, str] = field(default_factory=dict)
    scripts: Dict[str, str] = field(default_factory=dict)
    python_version: Optional[str] = None
    node_version: Optional[str] = None
    lockfile_path: Optional[Path] = None
    config_path: Optional[Path] = None


def detect_package_manager(project_root: str) -> List[PackageManager]:
    """
    Detect package manager(s) used in a project.

    Supports monorepos with multiple package managers.
    Detection order matters (most specific first).

    Args:
        project_root: Root directory of the project

    Returns:
        List of detected package managers (can be empty or multiple)
    """
    root = Path(project_root)
    if not root.exists() or not root.is_dir():
        return [PackageManager.UNKNOWN]

    managers: Set[PackageManager] = set()

    # JavaScript/TypeScript (order: bun → pnpm → yarn → npm)
    if (root / "bun.lockb").exists():
        managers.add(PackageManager.BUN)
    if (root / "pnpm-lock.yaml").exists():
        managers.add(PackageManager.PNPM)
    if (root / "yarn.lock").exists():
        managers.add(PackageManager.YARN)
    if (root / "package-lock.json").exists():
        managers.add(PackageManager.NPM)

    # Python (order: poetry → pipenv → pip)
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            with open(pyproject, "rb") as f:
                data = tomli.load(f)
                if "tool" in data and "poetry" in data["tool"]:
                    managers.add(PackageManager.POETRY)
        except Exception:
            pass

    if (root / "Pipfile").exists():
        managers.add(PackageManager.PIPENV)

    if (root / "requirements.txt").exists():
        managers.add(PackageManager.PIP)

    # Go
    if (root / "go.mod").exists():
        managers.add(PackageManager.GO_MODULES)

    # Rust
    if (root / "Cargo.toml").exists():
        managers.add(PackageManager.CARGO)

    # Java
    if (root / "pom.xml").exists():
        managers.add(PackageManager.MAVEN)
    if (root / "build.gradle").exists() or (root / "build.gradle.kts").exists():
        managers.add(PackageManager.GRADLE)

    # PHP
    if (root / "composer.json").exists():
        managers.add(PackageManager.COMPOSER)

    # Return sorted list for consistency
    if not managers:
        return [PackageManager.UNKNOWN]

    # Sort by priority (most specific first)
    priority = {
        PackageManager.BUN: 0,
        PackageManager.PNPM: 1,
        PackageManager.YARN: 2,
        PackageManager.NPM: 3,
        PackageManager.POETRY: 4,
        PackageManager.PIPENV: 5,
        PackageManager.PIP: 6,
        PackageManager.GO_MODULES: 7,
        PackageManager.CARGO: 8,
        PackageManager.MAVEN: 9,
        PackageManager.GRADLE: 10,
        PackageManager.COMPOSER: 11,
    }

    return sorted(managers, key=lambda m: priority.get(m, 99))


def get_dependencies(project_root: str, manager: PackageManager) -> PackageInfo:
    """
    Extract dependencies from package manager config files.

    Args:
        project_root: Root directory of the project
        manager: Package manager to extract from

    Returns:
        PackageInfo with dependencies, scripts, and metadata
    """
    root = Path(project_root)
    info = PackageInfo(manager=manager)

    try:
        if manager in [PackageManager.BUN, PackageManager.PNPM, 
                       PackageManager.YARN, PackageManager.NPM]:
            info = _extract_npm_dependencies(root, manager, info)
        elif manager == PackageManager.POETRY:
            info = _extract_poetry_dependencies(root, info)
        elif manager == PackageManager.PIPENV:
            info = _extract_pipenv_dependencies(root, info)
        elif manager == PackageManager.PIP:
            info = _extract_pip_dependencies(root, info)
        elif manager == PackageManager.GO_MODULES:
            info = _extract_go_dependencies(root, info)
        elif manager == PackageManager.CARGO:
            info = _extract_cargo_dependencies(root, info)
        elif manager == PackageManager.MAVEN:
            info = _extract_maven_dependencies(root, info)
        elif manager == PackageManager.GRADLE:
            info = _extract_gradle_dependencies(root, info)
        elif manager == PackageManager.COMPOSER:
            info = _extract_composer_dependencies(root, info)
    except Exception:
        # Return empty info on any error
        pass

    return info


def _extract_npm_dependencies(
    root: Path, manager: PackageManager, info: PackageInfo
) -> PackageInfo:
    """Extract dependencies from package.json."""
    package_json = root / "package.json"
    if not package_json.exists():
        return info

    info.config_path = package_json

    # Set lockfile path
    if manager == PackageManager.BUN:
        info.lockfile_path = root / "bun.lockb"
    elif manager == PackageManager.PNPM:
        info.lockfile_path = root / "pnpm-lock.yaml"
    elif manager == PackageManager.YARN:
        info.lockfile_path = root / "yarn.lock"
    elif manager == PackageManager.NPM:
        info.lockfile_path = root / "package-lock.json"

    with open(package_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract dependencies
    info.dependencies = data.get("dependencies", {})
    info.dev_dependencies = data.get("devDependencies", {})
    info.scripts = data.get("scripts", {})

    # Extract Node version
    engines = data.get("engines", {})
    info.node_version = engines.get("node")

    return info


def _extract_poetry_dependencies(root: Path, info: PackageInfo) -> PackageInfo:
    """Extract dependencies from pyproject.toml (Poetry)."""
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return info

    info.config_path = pyproject
    info.lockfile_path = root / "poetry.lock"

    with open(pyproject, "rb") as f:
        data = tomli.load(f)

    # Extract dependencies
    tool_poetry = data.get("tool", {}).get("poetry", {})
    deps = tool_poetry.get("dependencies", {})
    dev_deps = tool_poetry.get("dev-dependencies", {}) or tool_poetry.get("group", {}).get("dev", {}).get("dependencies", {})

    # Python version is special in poetry
    if "python" in deps:
        info.python_version = deps.pop("python")

    info.dependencies = deps
    info.dev_dependencies = dev_deps
    info.scripts = tool_poetry.get("scripts", {})

    return info


def _extract_pipenv_dependencies(root: Path, info: PackageInfo) -> PackageInfo:
    """Extract dependencies from Pipfile."""
    pipfile = root / "Pipfile"
    if not pipfile.exists():
        return info

    info.config_path = pipfile
    info.lockfile_path = root / "Pipfile.lock"

    with open(pipfile, "rb") as f:
        data = tomli.load(f)

    # Extract dependencies
    info.dependencies = data.get("packages", {})
    info.dev_dependencies = data.get("dev-packages", {})

    # Python version
    requires = data.get("requires", {})
    info.python_version = requires.get("python_version")

    return info


def _extract_pip_dependencies(root: Path, info: PackageInfo) -> PackageInfo:
    """Extract dependencies from requirements.txt."""
    requirements = root / "requirements.txt"
    if not requirements.exists():
        return info

    info.config_path = requirements

    with open(requirements, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Parse requirement (handle ==, >=, etc.)
            if "==" in line:
                pkg, version = line.split("==", 1)
                info.dependencies[pkg.strip()] = version.strip()
            elif ">=" in line:
                pkg, version = line.split(">=", 1)
                info.dependencies[pkg.strip()] = f">={version.strip()}"
            else:
                # No version specified
                info.dependencies[line] = "*"

    return info


def _extract_go_dependencies(root: Path, info: PackageInfo) -> PackageInfo:
    """Extract dependencies from go.mod."""
    go_mod = root / "go.mod"
    if not go_mod.exists():
        return info

    info.config_path = go_mod
    info.lockfile_path = root / "go.sum"

    with open(go_mod, "r", encoding="utf-8") as f:
        in_require_block = False
        for line in f:
            line = line.strip()

            if line.startswith("require ("):
                in_require_block = True
                continue
            elif line == ")":
                in_require_block = False
                continue

            if in_require_block or line.startswith("require "):
                # Remove "require " prefix if present
                line = line.replace("require ", "")
                parts = line.split()
                if len(parts) >= 2:
                    pkg, version = parts[0], parts[1]
                    info.dependencies[pkg] = version

    return info


def _extract_cargo_dependencies(root: Path, info: PackageInfo) -> PackageInfo:
    """Extract dependencies from Cargo.toml."""
    cargo_toml = root / "Cargo.toml"
    if not cargo_toml.exists():
        return info

    info.config_path = cargo_toml
    info.lockfile_path = root / "Cargo.lock"

    with open(cargo_toml, "rb") as f:
        data = tomli.load(f)

    info.dependencies = data.get("dependencies", {})
    info.dev_dependencies = data.get("dev-dependencies", {})

    return info


def _extract_maven_dependencies(root: Path, info: PackageInfo) -> PackageInfo:
    """Extract dependencies from pom.xml (basic parsing)."""
    pom_xml = root / "pom.xml"
    if not pom_xml.exists():
        return info

    info.config_path = pom_xml

    # Basic XML parsing (just presence detection for now)
    # Full Maven parsing would require xml.etree or lxml
    with open(pom_xml, "r", encoding="utf-8") as f:
        content = f.read()
        if "<dependencies>" in content:
            # Mark that dependencies exist
            info.dependencies["<maven-dependencies>"] = "detected"

    return info


def _extract_gradle_dependencies(root: Path, info: PackageInfo) -> PackageInfo:
    """Extract dependencies from build.gradle (basic parsing)."""
    build_gradle = root / "build.gradle"
    if not build_gradle.exists():
        build_gradle = root / "build.gradle.kts"
        if not build_gradle.exists():
            return info

    info.config_path = build_gradle

    # Basic parsing (just presence detection for now)
    with open(build_gradle, "r", encoding="utf-8") as f:
        content = f.read()
        if "dependencies {" in content:
            # Mark that dependencies exist
            info.dependencies["<gradle-dependencies>"] = "detected"

    return info


def _extract_composer_dependencies(root: Path, info: PackageInfo) -> PackageInfo:
    """Extract dependencies from composer.json."""
    composer_json = root / "composer.json"
    if not composer_json.exists():
        return info

    info.config_path = composer_json
    info.lockfile_path = root / "composer.lock"

    with open(composer_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    info.dependencies = data.get("require", {})
    info.dev_dependencies = data.get("require-dev", {})
    info.scripts = data.get("scripts", {})

    return info
