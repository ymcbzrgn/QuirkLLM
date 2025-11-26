"""
Unit tests for Package Manager Detection (Phase 3.1)

Tests package manager detection and dependency extraction for:
- JavaScript/TypeScript (bun, pnpm, yarn, npm)
- Python (poetry, pipenv, pip)
- Other languages (Go, Rust, Java, PHP)
"""

import json
import pytest
from pathlib import Path
from quirkllm.analyzer.package_detector import (
    PackageManager,
    PackageInfo,
    detect_package_manager,
    get_dependencies,
)


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project directory."""
    return tmp_path


class TestPackageManagerDetection:
    """Test suite for package manager detection"""

    def test_detect_bun_project(self, temp_project):
        """Bun project should be detected (highest priority)"""
        (temp_project / "bun.lockb").touch()
        (temp_project / "package.json").write_text("{}")

        managers = detect_package_manager(str(temp_project))
        assert PackageManager.BUN in managers

    def test_detect_pnpm_project(self, temp_project):
        """pnpm project should be detected"""
        (temp_project / "pnpm-lock.yaml").touch()
        (temp_project / "package.json").write_text("{}")

        managers = detect_package_manager(str(temp_project))
        assert PackageManager.PNPM in managers

    def test_detect_yarn_project(self, temp_project):
        """Yarn project should be detected"""
        (temp_project / "yarn.lock").touch()
        (temp_project / "package.json").write_text("{}")

        managers = detect_package_manager(str(temp_project))
        assert PackageManager.YARN in managers

    def test_detect_npm_project(self, temp_project):
        """npm project should be detected"""
        (temp_project / "package-lock.json").touch()
        (temp_project / "package.json").write_text("{}")

        managers = detect_package_manager(str(temp_project))
        assert PackageManager.NPM in managers

    def test_detect_poetry_project(self, temp_project):
        """Poetry project should be detected"""
        pyproject_content = """
[tool.poetry]
name = "test-project"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
"""
        (temp_project / "pyproject.toml").write_text(pyproject_content)

        managers = detect_package_manager(str(temp_project))
        assert PackageManager.POETRY in managers

    def test_detect_pipenv_project(self, temp_project):
        """Pipenv project should be detected"""
        (temp_project / "Pipfile").write_text("[[source]]\nurl = 'https://pypi.org/simple'")

        managers = detect_package_manager(str(temp_project))
        assert PackageManager.PIPENV in managers

    def test_detect_pip_project(self, temp_project):
        """pip project (requirements.txt) should be detected"""
        (temp_project / "requirements.txt").write_text("requests==2.31.0")

        managers = detect_package_manager(str(temp_project))
        assert PackageManager.PIP in managers

    def test_detect_go_modules(self, temp_project):
        """Go modules project should be detected"""
        (temp_project / "go.mod").write_text("module example.com/myproject\n")

        managers = detect_package_manager(str(temp_project))
        assert PackageManager.GO_MODULES in managers

    def test_detect_cargo_project(self, temp_project):
        """Cargo (Rust) project should be detected"""
        (temp_project / "Cargo.toml").write_text("[package]\nname = 'test'")

        managers = detect_package_manager(str(temp_project))
        assert PackageManager.CARGO in managers

    def test_detect_maven_project(self, temp_project):
        """Maven (Java) project should be detected"""
        (temp_project / "pom.xml").write_text("<project></project>")

        managers = detect_package_manager(str(temp_project))
        assert PackageManager.MAVEN in managers

    def test_detect_gradle_project(self, temp_project):
        """Gradle (Java) project should be detected"""
        (temp_project / "build.gradle").write_text("plugins { }")

        managers = detect_package_manager(str(temp_project))
        assert PackageManager.GRADLE in managers

    def test_detect_composer_project(self, temp_project):
        """Composer (PHP) project should be detected"""
        (temp_project / "composer.json").write_text('{"name": "test/project"}')

        managers = detect_package_manager(str(temp_project))
        assert PackageManager.COMPOSER in managers

    def test_detect_monorepo_multiple_managers(self, temp_project):
        """Monorepo with multiple package managers should detect all"""
        # JS + Python project
        (temp_project / "package.json").write_text("{}")
        (temp_project / "yarn.lock").touch()
        (temp_project / "requirements.txt").write_text("django==4.2")

        managers = detect_package_manager(str(temp_project))
        assert PackageManager.YARN in managers
        assert PackageManager.PIP in managers
        assert len(managers) >= 2

    def test_priority_order_npm_ecosystem(self, temp_project):
        """bun should have priority over pnpm, yarn, npm"""
        # Create all lockfiles
        (temp_project / "bun.lockb").touch()
        (temp_project / "pnpm-lock.yaml").touch()
        (temp_project / "yarn.lock").touch()
        (temp_project / "package-lock.json").touch()
        (temp_project / "package.json").write_text("{}")

        managers = detect_package_manager(str(temp_project))
        # bun should be first
        assert managers[0] == PackageManager.BUN

    def test_priority_order_python_ecosystem(self, temp_project):
        """Poetry should have priority over pipenv and pip"""
        pyproject_content = "[tool.poetry]\nname = 'test'\n"
        (temp_project / "pyproject.toml").write_text(pyproject_content)
        (temp_project / "Pipfile").touch()
        (temp_project / "requirements.txt").write_text("requests")

        managers = detect_package_manager(str(temp_project))
        # Poetry should come before pipenv and pip
        poetry_idx = managers.index(PackageManager.POETRY)
        if PackageManager.PIPENV in managers:
            pipenv_idx = managers.index(PackageManager.PIPENV)
            assert poetry_idx < pipenv_idx
        if PackageManager.PIP in managers:
            pip_idx = managers.index(PackageManager.PIP)
            assert poetry_idx < pip_idx

    def test_no_package_manager_detected(self, temp_project):
        """Empty project should return UNKNOWN"""
        managers = detect_package_manager(str(temp_project))
        assert managers == [PackageManager.UNKNOWN]

    def test_non_existent_directory(self, temp_project):
        """Non-existent directory should return UNKNOWN"""
        managers = detect_package_manager(str(temp_project / "nonexistent"))
        assert managers == [PackageManager.UNKNOWN]

    def test_file_instead_of_directory(self, temp_project):
        """File path instead of directory should return UNKNOWN"""
        file_path = temp_project / "file.txt"
        file_path.write_text("test")

        managers = detect_package_manager(str(file_path))
        assert managers == [PackageManager.UNKNOWN]


class TestNPMDependencyExtraction:
    """Test suite for npm ecosystem dependency extraction"""

    def test_extract_npm_dependencies(self, temp_project):
        """Extract dependencies from package.json"""
        package_json = {
            "name": "test-project",
            "version": "1.0.0",
            "dependencies": {
                "react": "^18.2.0",
                "next": "14.0.0"
            },
            "devDependencies": {
                "typescript": "^5.0.0",
                "@types/react": "^18.0.0"
            },
            "scripts": {
                "dev": "next dev",
                "build": "next build"
            },
            "engines": {
                "node": ">=18.0.0"
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))
        (temp_project / "package-lock.json").touch()

        info = get_dependencies(str(temp_project), PackageManager.NPM)

        assert info.manager == PackageManager.NPM
        assert info.dependencies == package_json["dependencies"]
        assert info.dev_dependencies == package_json["devDependencies"]
        assert info.scripts == package_json["scripts"]
        assert info.node_version == ">=18.0.0"
        assert info.config_path == temp_project / "package.json"
        assert info.lockfile_path == temp_project / "package-lock.json"

    def test_extract_bun_lockfile_path(self, temp_project):
        """Bun should use bun.lockb as lockfile"""
        package_json = {"name": "test", "dependencies": {"hono": "^3.0.0"}}
        (temp_project / "package.json").write_text(json.dumps(package_json))
        (temp_project / "bun.lockb").touch()

        info = get_dependencies(str(temp_project), PackageManager.BUN)

        assert info.lockfile_path == temp_project / "bun.lockb"

    def test_extract_pnpm_lockfile_path(self, temp_project):
        """pnpm should use pnpm-lock.yaml as lockfile"""
        package_json = {"name": "test", "dependencies": {}}
        (temp_project / "package.json").write_text(json.dumps(package_json))
        (temp_project / "pnpm-lock.yaml").touch()

        info = get_dependencies(str(temp_project), PackageManager.PNPM)

        assert info.lockfile_path == temp_project / "pnpm-lock.yaml"

    def test_extract_yarn_lockfile_path(self, temp_project):
        """Yarn should use yarn.lock as lockfile"""
        package_json = {"name": "test", "dependencies": {}}
        (temp_project / "package.json").write_text(json.dumps(package_json))
        (temp_project / "yarn.lock").touch()

        info = get_dependencies(str(temp_project), PackageManager.YARN)

        assert info.lockfile_path == temp_project / "yarn.lock"

    def test_npm_missing_package_json(self, temp_project):
        """Missing package.json should return empty dependencies"""
        info = get_dependencies(str(temp_project), PackageManager.NPM)

        assert info.manager == PackageManager.NPM
        assert info.dependencies == {}
        assert info.dev_dependencies == {}

    def test_npm_empty_dependencies(self, temp_project):
        """Empty dependencies should be handled correctly"""
        package_json = {"name": "test"}
        (temp_project / "package.json").write_text(json.dumps(package_json))

        info = get_dependencies(str(temp_project), PackageManager.NPM)

        assert info.dependencies == {}
        assert info.dev_dependencies == {}
        assert info.scripts == {}


class TestPythonDependencyExtraction:
    """Test suite for Python dependency extraction"""

    def test_extract_poetry_dependencies(self, temp_project):
        """Extract dependencies from pyproject.toml (Poetry)"""
        pyproject_content = """
[tool.poetry]
name = "test-project"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
django = "^4.2"
requests = "^2.31.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
black = "^23.0.0"

[tool.poetry.scripts]
serve = "project.cli:main"
"""
        (temp_project / "pyproject.toml").write_text(pyproject_content)
        (temp_project / "poetry.lock").touch()

        info = get_dependencies(str(temp_project), PackageManager.POETRY)

        assert info.manager == PackageManager.POETRY
        assert info.python_version == "^3.11"
        assert "django" in info.dependencies
        assert "requests" in info.dependencies
        assert "python" not in info.dependencies  # Should be extracted separately
        assert "pytest" in info.dev_dependencies
        assert "black" in info.dev_dependencies
        assert "serve" in info.scripts
        assert info.config_path == temp_project / "pyproject.toml"
        assert info.lockfile_path == temp_project / "poetry.lock"

    def test_extract_pipenv_dependencies(self, temp_project):
        """Extract dependencies from Pipfile"""
        pipfile_content = """
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
flask = "*"
sqlalchemy = "==2.0.0"

[dev-packages]
pytest = "*"

[requires]
python_version = "3.11"
"""
        (temp_project / "Pipfile").write_text(pipfile_content)
        (temp_project / "Pipfile.lock").touch()

        info = get_dependencies(str(temp_project), PackageManager.PIPENV)

        assert info.manager == PackageManager.PIPENV
        assert "flask" in info.dependencies
        assert "sqlalchemy" in info.dependencies
        assert "pytest" in info.dev_dependencies
        assert info.python_version == "3.11"
        assert info.config_path == temp_project / "Pipfile"
        assert info.lockfile_path == temp_project / "Pipfile.lock"

    def test_extract_pip_requirements(self, temp_project):
        """Extract dependencies from requirements.txt"""
        requirements_content = """
# Web framework
django==4.2.0
djangorestframework>=3.14.0

# Database
psycopg2-binary==2.9.0

# Utilities
requests
python-dotenv==1.0.0
"""
        (temp_project / "requirements.txt").write_text(requirements_content)

        info = get_dependencies(str(temp_project), PackageManager.PIP)

        assert info.manager == PackageManager.PIP
        assert info.dependencies["django"] == "4.2.0"
        assert info.dependencies["djangorestframework"] == ">=3.14.0"
        assert info.dependencies["psycopg2-binary"] == "2.9.0"
        assert info.dependencies["requests"] == "*"
        assert info.dependencies["python-dotenv"] == "1.0.0"
        assert info.config_path == temp_project / "requirements.txt"

    def test_pip_empty_requirements(self, temp_project):
        """Empty requirements.txt should be handled"""
        (temp_project / "requirements.txt").write_text("")

        info = get_dependencies(str(temp_project), PackageManager.PIP)

        assert info.dependencies == {}

    def test_pip_comments_only(self, temp_project):
        """requirements.txt with only comments should be handled"""
        requirements_content = """
# This is a comment
# Another comment
"""
        (temp_project / "requirements.txt").write_text(requirements_content)

        info = get_dependencies(str(temp_project), PackageManager.PIP)

        assert info.dependencies == {}


class TestOtherLanguageDependencyExtraction:
    """Test suite for Go, Rust, Java, PHP dependency extraction"""

    def test_extract_go_dependencies(self, temp_project):
        """Extract dependencies from go.mod"""
        go_mod_content = """
module example.com/myproject

go 1.21

require (
    github.com/gin-gonic/gin v1.9.1
    gorm.io/gorm v1.25.0
)

require github.com/stretchr/testify v1.8.4
"""
        (temp_project / "go.mod").write_text(go_mod_content)
        (temp_project / "go.sum").touch()

        info = get_dependencies(str(temp_project), PackageManager.GO_MODULES)

        assert info.manager == PackageManager.GO_MODULES
        assert "github.com/gin-gonic/gin" in info.dependencies
        assert info.dependencies["github.com/gin-gonic/gin"] == "v1.9.1"
        assert "gorm.io/gorm" in info.dependencies
        assert "github.com/stretchr/testify" in info.dependencies
        assert info.config_path == temp_project / "go.mod"
        assert info.lockfile_path == temp_project / "go.sum"

    def test_extract_cargo_dependencies(self, temp_project):
        """Extract dependencies from Cargo.toml"""
        cargo_toml_content = """
[package]
name = "my-project"
version = "0.1.0"

[dependencies]
serde = "1.0"
tokio = { version = "1.0", features = ["full"] }

[dev-dependencies]
criterion = "0.5"
"""
        (temp_project / "Cargo.toml").write_text(cargo_toml_content)
        (temp_project / "Cargo.lock").touch()

        info = get_dependencies(str(temp_project), PackageManager.CARGO)

        assert info.manager == PackageManager.CARGO
        assert "serde" in info.dependencies
        assert "criterion" in info.dev_dependencies
        assert info.config_path == temp_project / "Cargo.toml"
        assert info.lockfile_path == temp_project / "Cargo.lock"

    def test_extract_composer_dependencies(self, temp_project):
        """Extract dependencies from composer.json"""
        composer_json = {
            "name": "vendor/project",
            "require": {
                "php": "^8.1",
                "laravel/framework": "^10.0"
            },
            "require-dev": {
                "phpunit/phpunit": "^10.0"
            },
            "scripts": {
                "test": "phpunit"
            }
        }
        (temp_project / "composer.json").write_text(json.dumps(composer_json))
        (temp_project / "composer.lock").touch()

        info = get_dependencies(str(temp_project), PackageManager.COMPOSER)

        assert info.manager == PackageManager.COMPOSER
        assert "php" in info.dependencies
        assert "laravel/framework" in info.dependencies
        assert "phpunit/phpunit" in info.dev_dependencies
        assert "test" in info.scripts
        assert info.config_path == temp_project / "composer.json"
        assert info.lockfile_path == temp_project / "composer.lock"

    def test_extract_maven_dependencies_detected(self, temp_project):
        """Maven dependencies should be detected (basic)"""
        pom_xml_content = """
<project>
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
    </dependencies>
</project>
"""
        (temp_project / "pom.xml").write_text(pom_xml_content)

        info = get_dependencies(str(temp_project), PackageManager.MAVEN)

        assert info.manager == PackageManager.MAVEN
        assert "<maven-dependencies>" in info.dependencies
        assert info.config_path == temp_project / "pom.xml"

    def test_extract_gradle_dependencies_detected(self, temp_project):
        """Gradle dependencies should be detected (basic)"""
        build_gradle_content = """
plugins {
    id 'java'
}

dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-web'
    testImplementation 'junit:junit:4.13.2'
}
"""
        (temp_project / "build.gradle").write_text(build_gradle_content)

        info = get_dependencies(str(temp_project), PackageManager.GRADLE)

        assert info.manager == PackageManager.GRADLE
        assert "<gradle-dependencies>" in info.dependencies
        assert info.config_path == temp_project / "build.gradle"


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_malformed_json_package_json(self, temp_project):
        """Malformed package.json should return empty dependencies"""
        (temp_project / "package.json").write_text("{invalid json")

        info = get_dependencies(str(temp_project), PackageManager.NPM)

        assert info.dependencies == {}

    def test_malformed_toml_pyproject(self, temp_project):
        """Malformed pyproject.toml should return empty dependencies"""
        (temp_project / "pyproject.toml").write_text("[invalid toml")

        info = get_dependencies(str(temp_project), PackageManager.POETRY)

        assert info.dependencies == {}

    def test_pyproject_without_poetry(self, temp_project):
        """pyproject.toml without [tool.poetry] should not detect Poetry"""
        pyproject_content = """
[build-system]
requires = ["setuptools", "wheel"]
"""
        (temp_project / "pyproject.toml").write_text(pyproject_content)

        managers = detect_package_manager(str(temp_project))

        assert PackageManager.POETRY not in managers

    def test_unknown_package_manager(self, temp_project):
        """Unknown package manager should return empty PackageInfo"""
        info = get_dependencies(str(temp_project), PackageManager.UNKNOWN)

        assert info.manager == PackageManager.UNKNOWN
        assert info.dependencies == {}
        assert info.dev_dependencies == {}

    def test_missing_config_files(self, temp_project):
        """Missing config files should return empty dependencies"""
        # Try to extract from non-existent files
        info_npm = get_dependencies(str(temp_project), PackageManager.NPM)
        info_poetry = get_dependencies(str(temp_project), PackageManager.POETRY)
        info_pip = get_dependencies(str(temp_project), PackageManager.PIP)

        assert info_npm.dependencies == {}
        assert info_poetry.dependencies == {}
        assert info_pip.dependencies == {}
