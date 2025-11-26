"""
Smart Framework Detection for QuirkLLM

Detects frameworks, bundlers, styling solutions, and state management:
- Frontend: React, Next.js, Vue, Nuxt, Angular, Svelte, SvelteKit
- Backend: Django, FastAPI, Flask, Express, NestJS, Koa
- Bundlers: Webpack, Vite, Rollup, Parcel, esbuild
- Styling: Tailwind CSS, CSS Modules, Styled Components, Sass
- State: Redux, Zustand, Jotai, Recoil, MobX, Pinia
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set


@dataclass
class FrameworkInfo:
    """Framework and tooling information."""

    # Primary framework
    framework: Optional[str] = None
    framework_version: Optional[str] = None

    # Build tools
    bundler: Optional[str] = None
    bundler_version: Optional[str] = None

    # Styling
    styling: List[str] = field(default_factory=list)

    # State management
    state_management: Optional[str] = None

    # Testing frameworks
    test_framework: Optional[str] = None

    # All detected technologies
    technologies: Set[str] = field(default_factory=set)


def detect_framework(project_root: str) -> FrameworkInfo:
    """
    Detect primary framework used in a project.

    Args:
        project_root: Root directory of the project

    Returns:
        FrameworkInfo with detected framework and tools
    """
    root = Path(project_root)
    if not root.exists() or not root.is_dir():
        return FrameworkInfo()

    info = FrameworkInfo()

    # Try to read package.json for JS/TS projects
    package_json_path = root / "package.json"
    if package_json_path.exists():
        info = _detect_js_framework(package_json_path, info)

    # Try to detect Python frameworks
    pyproject_path = root / "pyproject.toml"
    requirements_path = root / "requirements.txt"
    if pyproject_path.exists() or requirements_path.exists():
        info = _detect_python_framework(root, info)

    # Detect bundler
    info = _detect_bundler(root, info)

    # Detect styling solutions
    info = _detect_styling(root, info)

    # Detect state management
    info = _detect_state_management(root, info)

    # Detect test framework
    info = _detect_test_framework(root, info)

    return info


def _detect_js_framework(package_json_path: Path, info: FrameworkInfo) -> FrameworkInfo:
    """Detect JavaScript/TypeScript framework from package.json."""
    try:
        with open(package_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        deps = data.get("dependencies", {})
        dev_deps = data.get("devDependencies", {})
        all_deps = {**deps, **dev_deps}

        # Frontend frameworks (priority order)
        if "next" in deps:
            info.framework = "Next.js"
            info.framework_version = deps["next"]
            info.technologies.add("Next.js")
            info.technologies.add("React")
        elif "@remix-run/react" in deps:
            info.framework = "Remix"
            info.framework_version = deps.get("@remix-run/react")
            info.technologies.add("Remix")
            info.technologies.add("React")
        elif "gatsby" in deps:
            info.framework = "Gatsby"
            info.framework_version = deps["gatsby"]
            info.technologies.add("Gatsby")
            info.technologies.add("React")
        elif "react-scripts" in all_deps:
            info.framework = "Create React App"
            info.framework_version = all_deps["react-scripts"]
            info.technologies.add("CRA")
            info.technologies.add("React")
        elif "react" in deps:
            # Generic React app
            if "vite" in dev_deps:
                info.framework = "React + Vite"
            else:
                info.framework = "React"
            info.framework_version = deps["react"]
            info.technologies.add("React")

        # Vue.js
        elif "nuxt" in deps:
            info.framework = "Nuxt"
            info.framework_version = deps["nuxt"]
            info.technologies.add("Nuxt")
            info.technologies.add("Vue")
        elif "vue" in deps:
            if "vite" in dev_deps and "@vitejs/plugin-vue" in dev_deps:
                info.framework = "Vue + Vite"
            else:
                info.framework = "Vue"
            info.framework_version = deps["vue"]
            info.technologies.add("Vue")

        # Angular
        elif "@angular/core" in deps:
            info.framework = "Angular"
            info.framework_version = deps["@angular/core"]
            info.technologies.add("Angular")

        # Svelte
        elif "@sveltejs/kit" in deps or "@sveltejs/kit" in dev_deps:
            info.framework = "SvelteKit"
            info.framework_version = all_deps.get("@sveltejs/kit")
            info.technologies.add("SvelteKit")
            info.technologies.add("Svelte")
        elif "svelte" in deps or "svelte" in dev_deps:
            info.framework = "Svelte"
            info.framework_version = all_deps.get("svelte")
            info.technologies.add("Svelte")

        # Backend frameworks (Node.js)
        elif "@nestjs/core" in deps:
            info.framework = "NestJS"
            info.framework_version = deps["@nestjs/core"]
            info.technologies.add("NestJS")
        elif "express" in deps:
            info.framework = "Express"
            info.framework_version = deps["express"]
            info.technologies.add("Express")
        elif "koa" in deps:
            info.framework = "Koa"
            info.framework_version = deps["koa"]
            info.technologies.add("Koa")
        elif "fastify" in deps:
            info.framework = "Fastify"
            info.framework_version = deps["fastify"]
            info.technologies.add("Fastify")
        elif "hono" in deps:
            info.framework = "Hono"
            info.framework_version = deps["hono"]
            info.technologies.add("Hono")

    except Exception:
        pass

    return info


def _detect_python_framework(root: Path, info: FrameworkInfo) -> FrameworkInfo:
    """Detect Python framework from dependencies."""
    dependencies: Set[str] = set()

    # Check pyproject.toml
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomli
            with open(pyproject, "rb") as f:
                data = tomli.load(f)
                # Poetry style
                poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
                dependencies.update(poetry_deps.keys())
                # PEP 621 style
                project_deps = data.get("project", {}).get("dependencies", [])
                for dep in project_deps:
                    # Handle "package>=1.0.0" format
                    pkg_name = dep.split("[")[0].split(">=")[0].split("==")[0].split("<")[0].strip()
                    dependencies.add(pkg_name)
        except Exception:
            pass

    # Check requirements.txt
    requirements = root / "requirements.txt"
    if requirements.exists():
        try:
            with open(requirements, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        pkg_name = line.split("==")[0].split(">=")[0].split("<")[0].strip()
                        dependencies.add(pkg_name)
        except Exception:
            pass

    # Detect frameworks
    if "django" in dependencies:
        info.framework = "Django"
        info.technologies.add("Django")
        # Try to get version from requirements
        for dep in dependencies:
            if dep.startswith("django"):
                break
    elif "fastapi" in dependencies:
        info.framework = "FastAPI"
        info.technologies.add("FastAPI")
    elif "flask" in dependencies:
        info.framework = "Flask"
        info.technologies.add("Flask")
    elif "tornado" in dependencies:
        info.framework = "Tornado"
        info.technologies.add("Tornado")
    elif "sanic" in dependencies:
        info.framework = "Sanic"
        info.technologies.add("Sanic")

    return info


def _detect_bundler(root: Path, info: FrameworkInfo) -> FrameworkInfo:
    """Detect bundler/build tool."""
    # Check for config files
    if (root / "vite.config.js").exists() or (root / "vite.config.ts").exists():
        info.bundler = "Vite"
        info.technologies.add("Vite")
    elif (root / "webpack.config.js").exists() or (root / "webpack.config.ts").exists():
        info.bundler = "Webpack"
        info.technologies.add("Webpack")
    elif (root / "rollup.config.js").exists() or (root / "rollup.config.ts").exists():
        info.bundler = "Rollup"
        info.technologies.add("Rollup")
    elif (root / ".parcelrc").exists():
        info.bundler = "Parcel"
        info.technologies.add("Parcel")
    elif (root / "esbuild.config.js").exists():
        info.bundler = "esbuild"
        info.technologies.add("esbuild")

    # Check package.json for bundler dependencies
    package_json = root / "package.json"
    if package_json.exists():
        try:
            with open(package_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}

                if not info.bundler:
                    if "vite" in all_deps:
                        info.bundler = "Vite"
                        info.bundler_version = all_deps["vite"]
                        info.technologies.add("Vite")
                    elif "webpack" in all_deps:
                        info.bundler = "Webpack"
                        info.bundler_version = all_deps["webpack"]
                        info.technologies.add("Webpack")
                    elif "rollup" in all_deps:
                        info.bundler = "Rollup"
                        info.bundler_version = all_deps["rollup"]
                        info.technologies.add("Rollup")
                    elif "parcel" in all_deps:
                        info.bundler = "Parcel"
                        info.bundler_version = all_deps["parcel"]
                        info.technologies.add("Parcel")
                    elif "esbuild" in all_deps:
                        info.bundler = "esbuild"
                        info.bundler_version = all_deps["esbuild"]
                        info.technologies.add("esbuild")
        except Exception:
            pass

    return info


def _detect_styling(root: Path, info: FrameworkInfo) -> FrameworkInfo:
    """Detect styling solutions."""
    styling_solutions = []

    # Check for Tailwind CSS
    if (root / "tailwind.config.js").exists() or (root / "tailwind.config.ts").exists():
        styling_solutions.append("Tailwind CSS")
        info.technologies.add("Tailwind CSS")

    # Check for PostCSS
    if (root / "postcss.config.js").exists():
        styling_solutions.append("PostCSS")
        info.technologies.add("PostCSS")

    # Check for Sass/SCSS files
    if list(root.glob("**/*.scss")) or list(root.glob("**/*.sass")):
        styling_solutions.append("Sass")
        info.technologies.add("Sass")

    # Check package.json for styling libraries
    package_json = root / "package.json"
    if package_json.exists():
        try:
            with open(package_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}

                if "tailwindcss" in all_deps:
                    if "Tailwind CSS" not in styling_solutions:
                        styling_solutions.append("Tailwind CSS")
                        info.technologies.add("Tailwind CSS")

                if "styled-components" in all_deps:
                    styling_solutions.append("Styled Components")
                    info.technologies.add("Styled Components")

                if "@emotion/react" in all_deps or "@emotion/styled" in all_deps:
                    styling_solutions.append("Emotion")
                    info.technologies.add("Emotion")

                if "sass" in all_deps:
                    if "Sass" not in styling_solutions:
                        styling_solutions.append("Sass")
                        info.technologies.add("Sass")

                # Check for CSS Modules (heuristic: check for .module.css files)
                if list(root.glob("**/*.module.css")) or list(root.glob("**/*.module.scss")):
                    styling_solutions.append("CSS Modules")
                    info.technologies.add("CSS Modules")

        except Exception:
            pass

    info.styling = styling_solutions
    return info


def _detect_state_management(root: Path, info: FrameworkInfo) -> FrameworkInfo:
    """Detect state management solution."""
    package_json = root / "package.json"
    if not package_json.exists():
        return info

    try:
        with open(package_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}

            # Priority order
            if "@reduxjs/toolkit" in all_deps:
                info.state_management = "Redux Toolkit"
                info.technologies.add("Redux Toolkit")
            elif "redux" in all_deps:
                info.state_management = "Redux"
                info.technologies.add("Redux")
            elif "zustand" in all_deps:
                info.state_management = "Zustand"
                info.technologies.add("Zustand")
            elif "jotai" in all_deps:
                info.state_management = "Jotai"
                info.technologies.add("Jotai")
            elif "recoil" in all_deps:
                info.state_management = "Recoil"
                info.technologies.add("Recoil")
            elif "mobx" in all_deps:
                info.state_management = "MobX"
                info.technologies.add("MobX")
            elif "pinia" in all_deps:
                info.state_management = "Pinia"
                info.technologies.add("Pinia")
            elif "vuex" in all_deps:
                info.state_management = "Vuex"
                info.technologies.add("Vuex")

    except Exception:
        pass

    return info


def _detect_test_framework(root: Path, info: FrameworkInfo) -> FrameworkInfo:
    """Detect testing framework."""
    package_json = root / "package.json"
    
    # Check for pytest (Python)
    requirements = root / "requirements.txt"
    pyproject = root / "pyproject.toml"
    
    if requirements.exists():
        try:
            with open(requirements, "r", encoding="utf-8") as f:
                content = f.read().lower()
                if "pytest" in content:
                    info.test_framework = "pytest"
                    info.technologies.add("pytest")
                elif "unittest" in content or "nose" in content:
                    info.test_framework = "unittest"
                    info.technologies.add("unittest")
        except Exception:
            pass
    
    if pyproject.exists() and not info.test_framework:
        try:
            import tomli
            with open(pyproject, "rb") as f:
                data = tomli.load(f)
                deps = data.get("tool", {}).get("poetry", {}).get("dev-dependencies", {})
                if "pytest" in deps:
                    info.test_framework = "pytest"
                    info.technologies.add("pytest")
        except Exception:
            pass

    # Check for JavaScript test frameworks
    if package_json.exists():
        try:
            with open(package_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}

                if "vitest" in all_deps:
                    info.test_framework = "Vitest"
                    info.technologies.add("Vitest")
                elif "jest" in all_deps or "@jest/core" in all_deps:
                    info.test_framework = "Jest"
                    info.technologies.add("Jest")
                elif "mocha" in all_deps:
                    info.test_framework = "Mocha"
                    info.technologies.add("Mocha")
                elif "@playwright/test" in all_deps:
                    info.test_framework = "Playwright"
                    info.technologies.add("Playwright")
                elif "cypress" in all_deps:
                    info.test_framework = "Cypress"
                    info.technologies.add("Cypress")

        except Exception:
            pass

    return info
