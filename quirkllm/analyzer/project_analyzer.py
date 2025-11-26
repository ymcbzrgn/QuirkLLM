"""
Project Structure Analyzer for QuirkLLM (Phase 3.3)

Scans project directory, generates statistics, detects important files,
and creates comprehensive project map by integrating package and framework detection.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Generator
from collections import defaultdict

from .package_detector import detect_package_manager, get_dependencies
from .framework_detector import detect_framework


# Directories to ignore during scanning
IGNORE_DIRS = {
    "node_modules", ".venv", "venv", "__pycache__", ".git", ".svn",
    ".hg", "dist", "build", "coverage", ".pytest_cache", ".mypy_cache",
    ".tox", "htmlcov", "site-packages", "env", ".env", "vendor",
    "Pods", ".DS_Store", "out", ".next", ".nuxt", ".cache"
}

# Important file patterns
ENTRY_POINT_FILES = {
    "main.py", "app.py", "index.js", "index.ts", "index.tsx",
    "App.tsx", "App.jsx", "main.ts", "main.tsx", "server.js",
    "server.ts", "__init__.py", "manage.py", "wsgi.py", "asgi.py"
}

CONFIG_FILES = {
    "package.json", "pyproject.toml", "tsconfig.json", "jsconfig.json",
    "webpack.config.js", "vite.config.js", "rollup.config.js",
    "tailwind.config.js", "postcss.config.js", "babel.config.js",
    ".babelrc", ".eslintrc.js", ".prettierrc", "jest.config.js",
    "vitest.config.ts", "playwright.config.ts", "Cargo.toml",
    "go.mod", "pom.xml", "build.gradle", "composer.json",
    "Gemfile", "requirements.txt", "Pipfile", "setup.py",
    "Makefile", "Dockerfile", ".dockerignore", "docker-compose.yml"
}

DOCUMENTATION_FILES = {
    "README.md", "README.rst", "README.txt", "CONTRIBUTING.md",
    "CHANGELOG.md", "LICENSE", "LICENSE.md", "LICENSE.txt",
    "SECURITY.md", "CODE_OF_CONDUCT.md", "AUTHORS", "CONTRIBUTORS"
}

CI_CD_PATTERNS = {
    ".github/workflows", ".gitlab-ci.yml", ".travis.yml",
    "Jenkinsfile", "azure-pipelines.yml", ".circleci/config.yml",
    "bitbucket-pipelines.yml", ".drone.yml"
}

# Source code extensions (for LOC counting)
CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".vue", ".svelte",
    ".go", ".rs", ".java", ".kt", ".swift", ".c", ".cpp",
    ".h", ".hpp", ".cs", ".php", ".rb", ".lua", ".scala"
}


@dataclass
class DirectoryStructure:
    """Directory structure information."""
    src: List[str] = field(default_factory=list)
    tests: List[str] = field(default_factory=list)
    docs: List[str] = field(default_factory=list)
    config: List[str] = field(default_factory=list)
    public: List[str] = field(default_factory=list)
    other: List[str] = field(default_factory=list)


@dataclass
class FileStats:
    """File statistics."""
    total_files: int = 0
    by_extension: Dict[str, int] = field(default_factory=dict)
    total_loc: int = 0
    loc_by_extension: Dict[str, int] = field(default_factory=dict)
    largest_files: List[Dict[str, any]] = field(default_factory=list)


@dataclass
class ProjectMap:
    """Complete project map."""
    root: str
    name: str
    framework: Optional[str] = None
    framework_version: Optional[str] = None
    package_manager: Optional[str] = None
    bundler: Optional[str] = None
    styling: List[str] = field(default_factory=list)
    state_management: Optional[str] = None
    test_framework: Optional[str] = None
    technologies: Set[str] = field(default_factory=set)
    dependencies: Dict[str, str] = field(default_factory=dict)
    dev_dependencies: Dict[str, str] = field(default_factory=dict)
    structure: DirectoryStructure = field(default_factory=DirectoryStructure)
    stats: FileStats = field(default_factory=FileStats)
    entry_points: List[str] = field(default_factory=list)
    important_files: List[str] = field(default_factory=list)


def analyze_project(project_root: str, max_depth: int = 10) -> ProjectMap:
    """
    Analyze project structure and generate comprehensive project map.
    
    Args:
        project_root: Path to project root directory
        max_depth: Maximum directory depth to scan (default: 10)
    
    Returns:
        ProjectMap with all detected information
    """
    root_path = Path(project_root).resolve()
    
    if not root_path.exists():
        return _create_empty_map(str(root_path))
    
    # Initialize project map
    project_map = ProjectMap(
        root=str(root_path),
        name=root_path.name
    )
    
    # Detect package manager and dependencies
    _detect_package_info(root_path, project_map)
    
    # Detect framework and tooling
    _detect_framework_info(root_path, project_map)
    
    # Scan directory structure
    _scan_structure(root_path, project_map, max_depth)
    
    # Calculate file statistics
    _calculate_stats(root_path, project_map, max_depth)
    
    # Detect important files
    _detect_important_files(root_path, project_map)
    
    return project_map


def _create_empty_map(root: str) -> ProjectMap:
    """Create empty project map for non-existent directory."""
    return ProjectMap(
        root=root,
        name=Path(root).name
    )


def _detect_package_info(root_path: Path, project_map: ProjectMap) -> None:
    """Detect package manager and extract dependencies."""
    try:
        managers = detect_package_manager(str(root_path))
        if managers:
            # Use first detected manager
            manager = managers[0]
            project_map.package_manager = manager.value
            
            # Extract dependencies
            package_info = get_dependencies(str(root_path), manager)
            if package_info:
                project_map.dependencies = package_info.dependencies
                project_map.dev_dependencies = package_info.dev_dependencies
    except Exception:
        # Silently continue if detection fails
        pass


def _detect_framework_info(root_path: Path, project_map: ProjectMap) -> None:
    """Detect framework and tooling information."""
    try:
        framework_info = detect_framework(str(root_path))
        
        project_map.framework = framework_info.framework
        project_map.framework_version = framework_info.framework_version
        project_map.bundler = framework_info.bundler
        project_map.styling = framework_info.styling
        project_map.state_management = framework_info.state_management
        project_map.test_framework = framework_info.test_framework
        project_map.technologies = framework_info.technologies
    except Exception:
        # Silently continue if detection fails
        pass


def _scan_structure(root_path: Path, project_map: ProjectMap, max_depth: int) -> None:
    """Scan directory structure and categorize directories."""
    structure = project_map.structure
    
    try:
        for item in root_path.iterdir():
            if not item.is_dir() or item.name in IGNORE_DIRS:
                continue
            
            relative = item.relative_to(root_path)
            relative_str = str(relative)
            
            # Categorize directories
            name_lower = item.name.lower()
            
            if name_lower in {"src", "lib", "app", "source"}:
                structure.src.append(relative_str)
            elif name_lower in {"test", "tests", "__tests__", "spec", "specs"}:
                structure.tests.append(relative_str)
            elif name_lower in {"docs", "doc", "documentation"}:
                structure.docs.append(relative_str)
            elif name_lower in {"config", ".config", "configs"}:
                structure.config.append(relative_str)
            elif name_lower in {"public", "static", "assets"}:
                structure.public.append(relative_str)
            else:
                structure.other.append(relative_str)
    except (PermissionError, OSError):
        # Silently continue if directory is not accessible
        pass


def _calculate_stats(root_path: Path, project_map: ProjectMap, max_depth: int) -> None:
    """Calculate file statistics (counts, LOC, etc.)."""
    stats = project_map.stats
    extension_counts = defaultdict(int)
    loc_counts = defaultdict(int)
    file_locs = []  # List of (file_path, loc) tuples
    
    try:
        for file_path in _walk_files(root_path, max_depth):
            # Count file
            stats.total_files += 1
            
            # Count by extension
            ext = file_path.suffix.lower()
            if ext:
                extension_counts[ext] += 1
            
            # Count LOC for code files
            if ext in CODE_EXTENSIONS:
                try:
                    loc = _count_loc(file_path)
                    stats.total_loc += loc
                    loc_counts[ext] += loc
                    
                    # Track for largest files
                    relative = file_path.relative_to(root_path)
                    file_locs.append((str(relative), loc))
                except (PermissionError, OSError, UnicodeDecodeError):
                    # Skip files that can't be read
                    pass
        
        # Convert defaultdicts to regular dicts
        stats.by_extension = dict(extension_counts)
        stats.loc_by_extension = dict(loc_counts)
        
        # Get top 10 largest files
        file_locs.sort(key=lambda x: x[1], reverse=True)
        stats.largest_files = [
            {"file": path, "loc": loc}
            for path, loc in file_locs[:10]
        ]
    except Exception:
        # Silently continue if scanning fails
        pass


def _walk_files(root_path: Path, max_depth: int, current_depth: int = 0) -> Generator[Path, None, None]:
    """Recursively walk directory tree and yield file paths."""
    if current_depth > max_depth:
        return
    
    try:
        for item in root_path.iterdir():
            # Skip ignored directories
            if item.is_dir():
                if item.name not in IGNORE_DIRS:
                    yield from _walk_files(item, max_depth, current_depth + 1)
            elif item.is_file():
                # Skip symlinks to avoid infinite loops
                if not item.is_symlink():
                    yield item
    except (PermissionError, OSError):
        # Skip directories/files we can't access
        pass


def _count_loc(file_path: Path) -> int:
    """Count lines of code in a file (non-empty, non-comment lines)."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        # Count non-empty lines (simple heuristic)
        # This doesn't perfectly exclude comments but is fast
        loc = sum(1 for line in lines if line.strip())
        return loc
    except (UnicodeDecodeError, PermissionError, OSError):
        return 0


def _detect_important_files(root_path: Path, project_map: ProjectMap) -> None:
    """Detect entry points and important files."""
    important = set()
    entry_points = set()
    
    try:
        # Check root directory for important files
        for item in root_path.iterdir():
            if not item.is_file():
                continue
            
            name = item.name
            
            # Check for entry points
            if name in ENTRY_POINT_FILES:
                entry_points.add(name)
            
            # Check for config files
            if name in CONFIG_FILES:
                important.add(name)
            
            # Check for documentation files
            if name in DOCUMENTATION_FILES:
                important.add(name)
        
        # Check for CI/CD files
        for pattern in CI_CD_PATTERNS:
            ci_path = root_path / pattern
            if ci_path.exists():
                important.add(pattern)
        
        project_map.entry_points = sorted(entry_points)
        project_map.important_files = sorted(important)
    except (PermissionError, OSError):
        # Silently continue if directory is not accessible
        pass


def project_map_to_json(project_map: ProjectMap, indent: int = 2) -> str:
    """
    Convert ProjectMap to JSON string.
    
    Args:
        project_map: ProjectMap to convert
        indent: JSON indentation (default: 2)
    
    Returns:
        JSON string representation
    """
    # Convert dataclass to dict
    data = asdict(project_map)
    
    # Convert sets to lists for JSON serialization
    if "technologies" in data:
        data["technologies"] = sorted(data["technologies"])
    
    return json.dumps(data, indent=indent)


def project_map_from_json(json_str: str) -> ProjectMap:
    """
    Load ProjectMap from JSON string.
    
    Args:
        json_str: JSON string
    
    Returns:
        ProjectMap instance
    """
    data = json.loads(json_str)
    
    # Convert lists back to sets where needed
    if "technologies" in data:
        data["technologies"] = set(data["technologies"])
    
    # Reconstruct nested dataclasses
    if "structure" in data:
        data["structure"] = DirectoryStructure(**data["structure"])
    
    if "stats" in data:
        data["stats"] = FileStats(**data["stats"])
    
    return ProjectMap(**data)
