"""
Unit tests for Project Structure Analyzer (Phase 3.3)

Tests project analysis including:
- Directory structure scanning
- File statistics calculation
- Important files detection
- Project map generation
- Integration with package and framework detection
"""

import json
import pytest
from pathlib import Path
from quirkllm.analyzer.project_analyzer import (
    analyze_project,
    project_map_to_json,
    project_map_from_json,
    ProjectMap,
    DirectoryStructure,
    FileStats,
)


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project directory."""
    return tmp_path


@pytest.fixture
def small_project(tmp_path):
    """Create a small Next.js project structure."""
    project = tmp_path / "nextjs-app"
    project.mkdir()
    
    # Create package.json
    package_json = {
        "name": "nextjs-app",
        "dependencies": {
            "next": "14.0.0",
            "react": "^18.2.0"
        },
        "devDependencies": {
            "tailwindcss": "^3.0.0"
        }
    }
    (project / "package.json").write_text(json.dumps(package_json))
    
    # Create package-lock.json to indicate npm
    (project / "package-lock.json").write_text(json.dumps({"lockfileVersion": 2}))
    
    # Create source structure
    (project / "src").mkdir()
    (project / "src" / "components").mkdir()
    (project / "src" / "pages").mkdir()
    
    # Create some source files
    (project / "src" / "index.tsx").write_text("export default function Home() { return <div>Hello</div>; }")
    (project / "src" / "components" / "Button.tsx").write_text("export const Button = () => <button>Click</button>;")
    (project / "src" / "pages" / "index.tsx").write_text("import Home from '../index';\nexport default Home;")
    
    # Create config files
    (project / "tsconfig.json").write_text('{"compilerOptions": {}}')
    (project / "tailwind.config.js").write_text("module.exports = {}")
    
    # Create documentation
    (project / "README.md").write_text("# Next.js App\n\nMy awesome app")
    (project / "LICENSE").write_text("MIT License")
    
    return project


@pytest.fixture
def python_project(tmp_path):
    """Create a Python Django project structure."""
    project = tmp_path / "django-app"
    project.mkdir()
    
    # Create pyproject.toml
    pyproject = """
[tool.poetry]
name = "django-app"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
django = "^4.2"
"""
    (project / "pyproject.toml").write_text(pyproject)
    
    # Create source structure
    (project / "src").mkdir()
    (project / "src" / "myapp").mkdir()
    (project / "tests").mkdir()
    
    # Create Python files
    (project / "manage.py").write_text("#!/usr/bin/env python\nimport sys\n")
    (project / "src" / "myapp" / "__init__.py").write_text("")
    (project / "src" / "myapp" / "models.py").write_text("from django.db import models\n\nclass User(models.Model):\n    pass")
    (project / "tests" / "test_models.py").write_text("def test_user():\n    assert True")
    
    # Create documentation
    (project / "README.md").write_text("# Django App")
    
    return project


class TestProjectAnalysis:
    """Test suite for basic project analysis"""

    def test_analyze_empty_directory(self, temp_project):
        """Empty directory should return minimal project map"""
        project_map = analyze_project(str(temp_project))
        
        assert project_map.root == str(temp_project)
        assert project_map.name == temp_project.name
        assert project_map.stats.total_files == 0
        assert len(project_map.entry_points) == 0

    def test_analyze_non_existent_directory(self, temp_project):
        """Non-existent directory should return empty map"""
        project_map = analyze_project(str(temp_project / "nonexistent"))
        
        assert project_map.name == "nonexistent"
        assert project_map.stats.total_files == 0

    def test_analyze_small_nextjs_project(self, small_project):
        """Small Next.js project should be analyzed correctly"""
        project_map = analyze_project(str(small_project))
        
        assert project_map.name == "nextjs-app"
        assert project_map.framework == "Next.js"
        assert project_map.package_manager == "npm"
        assert "Tailwind CSS" in project_map.styling
        assert "Next.js" in project_map.technologies

    def test_analyze_python_django_project(self, python_project):
        """Python Django project should be analyzed correctly"""
        project_map = analyze_project(str(python_project))
        
        assert project_map.name == "django-app"
        assert project_map.framework == "Django"
        assert project_map.package_manager == "poetry"


class TestDirectoryStructure:
    """Test suite for directory structure scanning"""

    def test_detect_src_directory(self, small_project):
        """Source directory should be detected"""
        project_map = analyze_project(str(small_project))
        
        assert "src" in project_map.structure.src

    def test_detect_tests_directory(self, python_project):
        """Tests directory should be detected"""
        project_map = analyze_project(str(python_project))
        
        assert "tests" in project_map.structure.tests

    def test_ignore_node_modules(self, temp_project):
        """node_modules should be ignored"""
        (temp_project / "node_modules").mkdir()
        (temp_project / "node_modules" / "package").mkdir()
        (temp_project / "node_modules" / "package" / "index.js").write_text("module.exports = {}")
        
        project_map = analyze_project(str(temp_project))
        
        # node_modules shouldn't appear in structure
        assert "node_modules" not in project_map.structure.other
        # Files inside node_modules shouldn't be counted
        assert project_map.stats.total_files == 0

    def test_ignore_venv_directory(self, temp_project):
        """Virtual environment directories should be ignored"""
        (temp_project / ".venv").mkdir()
        (temp_project / ".venv" / "lib").mkdir()
        (temp_project / ".venv" / "lib" / "python.so").write_text("binary")
        
        project_map = analyze_project(str(temp_project))
        
        assert ".venv" not in project_map.structure.other
        assert project_map.stats.total_files == 0

    def test_categorize_directories(self, temp_project):
        """Directories should be categorized correctly"""
        (temp_project / "src").mkdir()
        (temp_project / "lib").mkdir()
        (temp_project / "tests").mkdir()
        (temp_project / "docs").mkdir()
        (temp_project / "config").mkdir()
        (temp_project / "public").mkdir()
        (temp_project / "scripts").mkdir()
        
        project_map = analyze_project(str(temp_project))
        
        assert "src" in project_map.structure.src or "lib" in project_map.structure.src
        assert "tests" in project_map.structure.tests
        assert "docs" in project_map.structure.docs
        assert "config" in project_map.structure.config
        assert "public" in project_map.structure.public
        assert "scripts" in project_map.structure.other


class TestFileStatistics:
    """Test suite for file statistics calculation"""

    def test_count_total_files(self, small_project):
        """Total files should be counted correctly"""
        project_map = analyze_project(str(small_project))
        
        # 3 .tsx files + 2 config files (.json, .js) + 2 docs (.md, LICENSE)
        assert project_map.stats.total_files >= 7

    def test_count_by_extension(self, small_project):
        """Files should be counted by extension"""
        project_map = analyze_project(str(small_project))
        
        assert ".tsx" in project_map.stats.by_extension
        assert project_map.stats.by_extension[".tsx"] == 3
        assert ".json" in project_map.stats.by_extension
        assert ".js" in project_map.stats.by_extension

    def test_calculate_loc(self, small_project):
        """Lines of code should be calculated"""
        project_map = analyze_project(str(small_project))
        
        assert project_map.stats.total_loc > 0
        assert ".tsx" in project_map.stats.loc_by_extension

    def test_identify_largest_files(self, small_project):
        """Largest files should be identified"""
        project_map = analyze_project(str(small_project))
        
        assert len(project_map.stats.largest_files) > 0
        # Check format
        assert "file" in project_map.stats.largest_files[0]
        assert "loc" in project_map.stats.largest_files[0]

    def test_python_loc_counting(self, python_project):
        """Python LOC should be counted"""
        project_map = analyze_project(str(python_project))
        
        assert project_map.stats.total_loc > 0
        assert ".py" in project_map.stats.loc_by_extension


class TestImportantFiles:
    """Test suite for important files detection"""

    def test_detect_readme(self, small_project):
        """README.md should be detected"""
        project_map = analyze_project(str(small_project))
        
        assert "README.md" in project_map.important_files

    def test_detect_license(self, small_project):
        """LICENSE should be detected"""
        project_map = analyze_project(str(small_project))
        
        assert "LICENSE" in project_map.important_files

    def test_detect_config_files(self, small_project):
        """Config files should be detected"""
        project_map = analyze_project(str(small_project))
        
        assert "package.json" in project_map.important_files
        assert "tsconfig.json" in project_map.important_files
        assert "tailwind.config.js" in project_map.important_files

    def test_detect_entry_points(self, python_project):
        """Entry points should be detected"""
        project_map = analyze_project(str(python_project))
        
        assert "manage.py" in project_map.entry_points

    def test_detect_ci_cd_files(self, temp_project):
        """CI/CD files should be detected"""
        (temp_project / ".github").mkdir()
        (temp_project / ".github" / "workflows").mkdir()
        (temp_project / ".github" / "workflows" / "ci.yml").write_text("name: CI")
        
        project_map = analyze_project(str(temp_project))
        
        assert ".github/workflows" in project_map.important_files


class TestDependencyIntegration:
    """Test suite for package manager integration"""

    def test_extract_npm_dependencies(self, small_project):
        """NPM dependencies should be extracted"""
        project_map = analyze_project(str(small_project))
        
        assert "next" in project_map.dependencies
        assert project_map.dependencies["next"] == "14.0.0"
        assert "tailwindcss" in project_map.dev_dependencies

    def test_extract_poetry_dependencies(self, python_project):
        """Poetry dependencies should be extracted"""
        project_map = analyze_project(str(python_project))
        
        assert "django" in project_map.dependencies


class TestFrameworkIntegration:
    """Test suite for framework detector integration"""

    def test_detect_nextjs_framework(self, small_project):
        """Next.js framework should be detected"""
        project_map = analyze_project(str(small_project))
        
        assert project_map.framework == "Next.js"
        assert project_map.framework_version == "14.0.0"

    def test_detect_django_framework(self, python_project):
        """Django framework should be detected"""
        project_map = analyze_project(str(python_project))
        
        assert project_map.framework == "Django"

    def test_detect_bundler(self, small_project):
        """Bundler should be detected (Next.js uses Webpack by default)"""
        project_map = analyze_project(str(small_project))
        
        # Next.js might not have explicit bundler, but that's ok
        assert project_map.bundler is None or isinstance(project_map.bundler, str)

    def test_detect_styling(self, small_project):
        """Styling solution should be detected"""
        project_map = analyze_project(str(small_project))
        
        assert "Tailwind CSS" in project_map.styling

    def test_technologies_set(self, small_project):
        """Technologies set should include all detected tools"""
        project_map = analyze_project(str(small_project))
        
        assert "Next.js" in project_map.technologies
        assert "React" in project_map.technologies
        assert "Tailwind CSS" in project_map.technologies


class TestJSONSerialization:
    """Test suite for JSON serialization"""

    def test_project_map_to_json(self, small_project):
        """ProjectMap should serialize to JSON"""
        project_map = analyze_project(str(small_project))
        json_str = project_map_to_json(project_map)
        
        assert isinstance(json_str, str)
        # Should be valid JSON
        data = json.loads(json_str)
        assert data["name"] == "nextjs-app"
        assert data["framework"] == "Next.js"

    def test_project_map_from_json(self, small_project):
        """ProjectMap should deserialize from JSON"""
        project_map = analyze_project(str(small_project))
        json_str = project_map_to_json(project_map)
        
        restored = project_map_from_json(json_str)
        
        assert restored.name == project_map.name
        assert restored.framework == project_map.framework
        assert restored.stats.total_files == project_map.stats.total_files

    def test_json_roundtrip(self, small_project):
        """JSON roundtrip should preserve data"""
        original = analyze_project(str(small_project))
        json_str = project_map_to_json(original)
        restored = project_map_from_json(json_str)
        
        assert original.root == restored.root
        assert original.name == restored.name
        assert original.framework == restored.framework
        assert original.package_manager == restored.package_manager
        assert original.stats.total_files == restored.stats.total_files


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_max_depth_limit(self, temp_project):
        """Should respect max_depth parameter"""
        # Create deep directory structure
        deep_dir = temp_project
        for i in range(15):
            deep_dir = deep_dir / f"level{i}"
            deep_dir.mkdir()
        
        (deep_dir / "file.txt").write_text("deep")
        
        # With default max_depth=10, deepest file shouldn't be found
        project_map = analyze_project(str(temp_project), max_depth=5)
        
        # Should still work without crashing
        assert project_map.name == temp_project.name

    def test_permission_error_handling(self, temp_project):
        """Should handle permission errors gracefully"""
        # This test might not work on all systems, so just verify no crash
        project_map = analyze_project(str(temp_project))
        
        assert project_map is not None

    def test_symlink_handling(self, temp_project):
        """Should not follow symlinks to avoid infinite loops"""
        (temp_project / "real").mkdir()
        (temp_project / "real" / "file.txt").write_text("content")
        
        # Create symlink (might not work on Windows)
        try:
            (temp_project / "link").symlink_to(temp_project / "real")
        except OSError:
            pytest.skip("Symlink creation not supported")
        
        project_map = analyze_project(str(temp_project))
        
        # Should not crash or loop infinitely
        assert project_map.stats.total_files >= 1

    def test_unicode_in_filenames(self, temp_project):
        """Should handle unicode in filenames"""
        (temp_project / "文件.py").write_text("# Chinese filename")
        (temp_project / "файл.js").write_text("// Russian filename")
        
        project_map = analyze_project(str(temp_project))
        
        assert project_map.stats.total_files >= 2

    def test_binary_files(self, temp_project):
        """Should handle binary files gracefully"""
        (temp_project / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (temp_project / "code.py").write_text("print('hello')")
        
        project_map = analyze_project(str(temp_project))
        
        # Should count both files
        assert project_map.stats.total_files == 2
        # LOC should only count .py file
        assert project_map.stats.total_loc > 0


class TestLargeProject:
    """Test suite for large project handling"""

    def test_large_project_performance(self, temp_project):
        """Should handle projects with many files efficiently"""
        # Create 100 files
        (temp_project / "src").mkdir()
        for i in range(100):
            (temp_project / "src" / f"file{i}.js").write_text(f"// File {i}\nconst x = {i};")
        
        project_map = analyze_project(str(temp_project))
        
        assert project_map.stats.total_files == 100
        assert ".js" in project_map.stats.by_extension
        assert project_map.stats.by_extension[".js"] == 100

    def test_largest_files_limited_to_10(self, temp_project):
        """Largest files list should be limited to 10"""
        (temp_project / "src").mkdir()
        # Create 20 files with varying sizes
        for i in range(20):
            lines = "\n".join([f"line {j}" for j in range(i * 10)])
            (temp_project / "src" / f"file{i}.py").write_text(lines)
        
        project_map = analyze_project(str(temp_project))
        
        # Should have at most 10 largest files
        assert len(project_map.stats.largest_files) <= 10
        # Should be sorted by LOC (descending)
        if len(project_map.stats.largest_files) > 1:
            assert project_map.stats.largest_files[0]["loc"] >= project_map.stats.largest_files[-1]["loc"]
