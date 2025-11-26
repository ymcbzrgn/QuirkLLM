"""
Unit tests for Framework Detection (Phase 3.2)

Tests framework detection for:
- Frontend: React, Next.js, Vue, Angular, Svelte
- Backend: Django, FastAPI, Express, NestJS
- Bundlers: Vite, Webpack, Rollup, Parcel
- Styling: Tailwind CSS, Styled Components, Sass
- State: Redux, Zustand, MobX, Pinia
"""

import json
import pytest
from pathlib import Path
from quirkllm.analyzer.framework_detector import (
    FrameworkInfo,
    detect_framework,
)


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project directory."""
    return tmp_path


class TestFrontendFrameworkDetection:
    """Test suite for frontend framework detection"""

    def test_detect_nextjs(self, temp_project):
        """Next.js should be detected"""
        package_json = {
            "dependencies": {
                "next": "14.0.0",
                "react": "^18.2.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.framework == "Next.js"
        assert info.framework_version == "14.0.0"
        assert "Next.js" in info.technologies
        assert "React" in info.technologies

    def test_detect_remix(self, temp_project):
        """Remix should be detected"""
        package_json = {
            "dependencies": {
                "@remix-run/react": "^2.0.0",
                "react": "^18.2.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.framework == "Remix"
        assert "Remix" in info.technologies
        assert "React" in info.technologies

    def test_detect_gatsby(self, temp_project):
        """Gatsby should be detected"""
        package_json = {
            "dependencies": {
                "gatsby": "^5.0.0",
                "react": "^18.2.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.framework == "Gatsby"
        assert "Gatsby" in info.technologies
        assert "React" in info.technologies

    def test_detect_create_react_app(self, temp_project):
        """Create React App should be detected"""
        package_json = {
            "dependencies": {
                "react": "^18.2.0"
            },
            "devDependencies": {
                "react-scripts": "5.0.1"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.framework == "Create React App"
        assert "CRA" in info.technologies
        assert "React" in info.technologies

    def test_detect_react_with_vite(self, temp_project):
        """React + Vite should be detected"""
        package_json = {
            "dependencies": {
                "react": "^18.2.0"
            },
            "devDependencies": {
                "vite": "^5.0.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.framework == "React + Vite"
        assert "React" in info.technologies

    def test_detect_plain_react(self, temp_project):
        """Plain React app should be detected"""
        package_json = {
            "dependencies": {
                "react": "^18.2.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.framework == "React"
        assert info.framework_version == "^18.2.0"
        assert "React" in info.technologies

    def test_detect_nuxt(self, temp_project):
        """Nuxt should be detected"""
        package_json = {
            "dependencies": {
                "nuxt": "^3.0.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.framework == "Nuxt"
        assert "Nuxt" in info.technologies
        assert "Vue" in info.technologies

    def test_detect_vue_with_vite(self, temp_project):
        """Vue + Vite should be detected"""
        package_json = {
            "dependencies": {
                "vue": "^3.3.0"
            },
            "devDependencies": {
                "vite": "^5.0.0",
                "@vitejs/plugin-vue": "^4.0.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.framework == "Vue + Vite"
        assert "Vue" in info.technologies

    def test_detect_plain_vue(self, temp_project):
        """Plain Vue app should be detected"""
        package_json = {
            "dependencies": {
                "vue": "^3.3.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.framework == "Vue"
        assert "Vue" in info.technologies

    def test_detect_angular(self, temp_project):
        """Angular should be detected"""
        package_json = {
            "dependencies": {
                "@angular/core": "^17.0.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.framework == "Angular"
        assert "Angular" in info.technologies

    def test_detect_sveltekit(self, temp_project):
        """SvelteKit should be detected"""
        package_json = {
            "devDependencies": {
                "@sveltejs/kit": "^2.0.0",
                "svelte": "^4.0.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.framework == "SvelteKit"
        assert "SvelteKit" in info.technologies
        assert "Svelte" in info.technologies

    def test_detect_plain_svelte(self, temp_project):
        """Plain Svelte app should be detected"""
        package_json = {
            "devDependencies": {
                "svelte": "^4.0.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.framework == "Svelte"
        assert "Svelte" in info.technologies


class TestBackendFrameworkDetection:
    """Test suite for backend framework detection"""

    def test_detect_nestjs(self, temp_project):
        """NestJS should be detected"""
        package_json = {
            "dependencies": {
                "@nestjs/core": "^10.0.0",
                "@nestjs/common": "^10.0.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.framework == "NestJS"
        assert "NestJS" in info.technologies

    def test_detect_express(self, temp_project):
        """Express should be detected"""
        package_json = {
            "dependencies": {
                "express": "^4.18.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.framework == "Express"
        assert "Express" in info.technologies

    def test_detect_koa(self, temp_project):
        """Koa should be detected"""
        package_json = {
            "dependencies": {
                "koa": "^2.14.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.framework == "Koa"
        assert "Koa" in info.technologies

    def test_detect_fastify(self, temp_project):
        """Fastify should be detected"""
        package_json = {
            "dependencies": {
                "fastify": "^4.0.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.framework == "Fastify"
        assert "Fastify" in info.technologies

    def test_detect_hono(self, temp_project):
        """Hono should be detected"""
        package_json = {
            "dependencies": {
                "hono": "^3.0.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.framework == "Hono"
        assert "Hono" in info.technologies

    def test_detect_django(self, temp_project):
        """Django should be detected from requirements.txt"""
        (temp_project / "requirements.txt").write_text("django==4.2.0\npsycopg2-binary")

        info = detect_framework(str(temp_project))

        assert info.framework == "Django"
        assert "Django" in info.technologies

    def test_detect_fastapi(self, temp_project):
        """FastAPI should be detected"""
        (temp_project / "requirements.txt").write_text("fastapi==0.104.0\nuvicorn")

        info = detect_framework(str(temp_project))

        assert info.framework == "FastAPI"
        assert "FastAPI" in info.technologies

    def test_detect_flask(self, temp_project):
        """Flask should be detected"""
        (temp_project / "requirements.txt").write_text("flask==3.0.0\nwerkzeug")

        info = detect_framework(str(temp_project))

        assert info.framework == "Flask"
        assert "Flask" in info.technologies

    def test_detect_django_from_pyproject(self, temp_project):
        """Django should be detected from pyproject.toml"""
        pyproject_content = """
[tool.poetry.dependencies]
python = "^3.11"
django = "^4.2"
"""
        (temp_project / "pyproject.toml").write_text(pyproject_content)

        info = detect_framework(str(temp_project))

        assert info.framework == "Django"
        assert "Django" in info.technologies


class TestBundlerDetection:
    """Test suite for bundler detection"""

    def test_detect_vite_from_config(self, temp_project):
        """Vite should be detected from config file"""
        (temp_project / "vite.config.js").write_text("export default {}")

        info = detect_framework(str(temp_project))

        assert info.bundler == "Vite"
        assert "Vite" in info.technologies

    def test_detect_vite_from_package_json(self, temp_project):
        """Vite should be detected from package.json"""
        package_json = {
            "devDependencies": {
                "vite": "^5.0.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.bundler == "Vite"
        assert info.bundler_version == "^5.0.0"

    def test_detect_webpack(self, temp_project):
        """Webpack should be detected"""
        (temp_project / "webpack.config.js").write_text("module.exports = {}")

        info = detect_framework(str(temp_project))

        assert info.bundler == "Webpack"
        assert "Webpack" in info.technologies

    def test_detect_rollup(self, temp_project):
        """Rollup should be detected"""
        (temp_project / "rollup.config.js").write_text("export default {}")

        info = detect_framework(str(temp_project))

        assert info.bundler == "Rollup"
        assert "Rollup" in info.technologies

    def test_detect_parcel(self, temp_project):
        """Parcel should be detected"""
        (temp_project / ".parcelrc").write_text('{"extends": "@parcel/config-default"}')

        info = detect_framework(str(temp_project))

        assert info.bundler == "Parcel"
        assert "Parcel" in info.technologies

    def test_detect_esbuild(self, temp_project):
        """esbuild should be detected"""
        (temp_project / "esbuild.config.js").write_text("module.exports = {}")

        info = detect_framework(str(temp_project))

        assert info.bundler == "esbuild"
        assert "esbuild" in info.technologies


class TestStylingDetection:
    """Test suite for styling solution detection"""

    def test_detect_tailwind_from_config(self, temp_project):
        """Tailwind CSS should be detected from config"""
        (temp_project / "tailwind.config.js").write_text("module.exports = {}")

        info = detect_framework(str(temp_project))

        assert "Tailwind CSS" in info.styling
        assert "Tailwind CSS" in info.technologies

    def test_detect_tailwind_from_package_json(self, temp_project):
        """Tailwind CSS should be detected from package.json"""
        package_json = {
            "devDependencies": {
                "tailwindcss": "^3.0.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert "Tailwind CSS" in info.styling

    def test_detect_styled_components(self, temp_project):
        """Styled Components should be detected"""
        package_json = {
            "dependencies": {
                "styled-components": "^6.0.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert "Styled Components" in info.styling
        assert "Styled Components" in info.technologies

    def test_detect_emotion(self, temp_project):
        """Emotion should be detected"""
        package_json = {
            "dependencies": {
                "@emotion/react": "^11.0.0",
                "@emotion/styled": "^11.0.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert "Emotion" in info.styling

    def test_detect_sass_from_files(self, temp_project):
        """Sass should be detected from .scss files"""
        (temp_project / "src").mkdir()
        (temp_project / "src" / "styles.scss").write_text("body { color: red; }")

        info = detect_framework(str(temp_project))

        assert "Sass" in info.styling

    def test_detect_sass_from_package_json(self, temp_project):
        """Sass should be detected from package.json"""
        package_json = {
            "devDependencies": {
                "sass": "^1.69.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert "Sass" in info.styling

    def test_detect_postcss(self, temp_project):
        """PostCSS should be detected"""
        (temp_project / "postcss.config.js").write_text("module.exports = {}")

        info = detect_framework(str(temp_project))

        assert "PostCSS" in info.styling

    def test_detect_css_modules(self, temp_project):
        """CSS Modules should be detected"""
        (temp_project / "src").mkdir()
        (temp_project / "src" / "Button.module.css").write_text(".button {}")
        package_json = {"name": "test"}
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert "CSS Modules" in info.styling

    def test_detect_multiple_styling_solutions(self, temp_project):
        """Multiple styling solutions should be detected"""
        (temp_project / "tailwind.config.js").write_text("{}")
        (temp_project / "postcss.config.js").write_text("{}")
        package_json = {
            "devDependencies": {
                "sass": "^1.69.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert "Tailwind CSS" in info.styling
        assert "PostCSS" in info.styling
        assert "Sass" in info.styling
        assert len(info.styling) >= 3


class TestStateManagementDetection:
    """Test suite for state management detection"""

    def test_detect_redux_toolkit(self, temp_project):
        """Redux Toolkit should be detected"""
        package_json = {
            "dependencies": {
                "@reduxjs/toolkit": "^2.0.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.state_management == "Redux Toolkit"
        assert "Redux Toolkit" in info.technologies

    def test_detect_redux(self, temp_project):
        """Redux should be detected"""
        package_json = {
            "dependencies": {
                "redux": "^4.2.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.state_management == "Redux"

    def test_detect_zustand(self, temp_project):
        """Zustand should be detected"""
        package_json = {
            "dependencies": {
                "zustand": "^4.0.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.state_management == "Zustand"

    def test_detect_jotai(self, temp_project):
        """Jotai should be detected"""
        package_json = {
            "dependencies": {
                "jotai": "^2.0.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.state_management == "Jotai"

    def test_detect_recoil(self, temp_project):
        """Recoil should be detected"""
        package_json = {
            "dependencies": {
                "recoil": "^0.7.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.state_management == "Recoil"

    def test_detect_mobx(self, temp_project):
        """MobX should be detected"""
        package_json = {
            "dependencies": {
                "mobx": "^6.0.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.state_management == "MobX"

    def test_detect_pinia(self, temp_project):
        """Pinia should be detected"""
        package_json = {
            "dependencies": {
                "pinia": "^2.0.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.state_management == "Pinia"

    def test_detect_vuex(self, temp_project):
        """Vuex should be detected"""
        package_json = {
            "dependencies": {
                "vuex": "^4.0.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.state_management == "Vuex"


class TestTestFrameworkDetection:
    """Test suite for test framework detection"""

    def test_detect_vitest(self, temp_project):
        """Vitest should be detected"""
        package_json = {
            "devDependencies": {
                "vitest": "^1.0.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.test_framework == "Vitest"
        assert "Vitest" in info.technologies

    def test_detect_jest(self, temp_project):
        """Jest should be detected"""
        package_json = {
            "devDependencies": {
                "jest": "^29.0.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.test_framework == "Jest"

    def test_detect_playwright(self, temp_project):
        """Playwright should be detected"""
        package_json = {
            "devDependencies": {
                "@playwright/test": "^1.40.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = detect_framework(str(temp_project))

        assert info.test_framework == "Playwright"

    def test_detect_pytest(self, temp_project):
        """pytest should be detected"""
        (temp_project / "requirements.txt").write_text("pytest==7.4.0")

        info = detect_framework(str(temp_project))

        assert info.test_framework == "pytest"


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_no_framework_detected(self, temp_project):
        """Empty project should return empty FrameworkInfo"""
        info = detect_framework(str(temp_project))

        assert info.framework is None
        assert info.bundler is None
        assert len(info.styling) == 0

    def test_non_existent_directory(self, temp_project):
        """Non-existent directory should return empty FrameworkInfo"""
        info = detect_framework(str(temp_project / "nonexistent"))

        assert info.framework is None

    def test_malformed_package_json(self, temp_project):
        """Malformed package.json should be handled gracefully"""
        (temp_project / "package.json").write_text("{invalid json")

        info = detect_framework(str(temp_project))

        assert info.framework is None

    def test_malformed_pyproject_toml(self, temp_project):
        """Malformed pyproject.toml should be handled gracefully"""
        (temp_project / "pyproject.toml").write_text("[invalid toml")

        info = detect_framework(str(temp_project))

        assert info.framework is None

    def test_empty_package_json(self, temp_project):
        """Empty package.json should be handled"""
        (temp_project / "package.json").write_text("{}")

        info = detect_framework(str(temp_project))

        assert info.framework is None

    def test_full_stack_project(self, temp_project):
        """Full-stack project with multiple technologies"""
        package_json = {
            "dependencies": {
                "next": "14.0.0",
                "react": "^18.2.0",
                "@reduxjs/toolkit": "^2.0.0"
            },
            "devDependencies": {
                "tailwindcss": "^3.0.0",
                "vitest": "^1.0.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))
        (temp_project / "tailwind.config.js").write_text("{}")

        info = detect_framework(str(temp_project))

        assert info.framework == "Next.js"
        assert info.state_management == "Redux Toolkit"
        assert info.test_framework == "Vitest"
        assert "Tailwind CSS" in info.styling
        assert "Next.js" in info.technologies
        assert "React" in info.technologies
        assert "Redux Toolkit" in info.technologies
        assert "Vitest" in info.technologies
