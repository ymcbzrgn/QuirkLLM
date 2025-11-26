"""
QuirkLLM Project Analyzer Package

Provides intelligent project analysis capabilities including:
- Package manager detection (bun, pnpm, yarn, npm, poetry, pipenv, pip, etc.)
- Framework detection (React, Next.js, Django, FastAPI, etc.)
- Project structure analysis
"""

from quirkllm.analyzer.package_detector import (
    PackageManager,
    PackageInfo,
    detect_package_manager,
    get_dependencies,
)

__all__ = [
    "PackageManager",
    "PackageInfo",
    "detect_package_manager",
    "get_dependencies",
]
