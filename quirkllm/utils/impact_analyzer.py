"""
Impact Analyzer for Ghost Mode.

Analyzes the impact of code changes by detecting dependencies,
breaking changes, and assessing risk levels.
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ChangeImpact:
    """
    Impact analysis result for a code change.
    
    Attributes:
        file_path: Path to the changed file
        risk_level: Risk level (safe, low, medium, high, critical)
        risk_score: Numeric risk score (0-100)
        breaking_changes: List of detected breaking changes
        removed_functions: Functions that were removed
        removed_classes: Classes that were removed
        signature_changes: Function signatures that changed
        added_imports: New imports added
        removed_imports: Imports that were removed
        affected_modules: Modules that may be affected
        recommendations: List of recommended actions
    """
    
    file_path: str
    risk_level: str
    risk_score: int
    breaking_changes: list[str] = field(default_factory=list)
    removed_functions: list[str] = field(default_factory=list)
    removed_classes: list[str] = field(default_factory=list)
    signature_changes: list[str] = field(default_factory=list)
    added_imports: list[str] = field(default_factory=list)
    removed_imports: list[str] = field(default_factory=list)
    affected_modules: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class ImpactAnalyzer:
    """
    Analyze the impact of code changes.
    
    Detects:
    - Breaking changes (removed functions, classes, changed signatures)
    - Dependency changes (imports added/removed)
    - Risk levels (safe, low, medium, high, critical)
    - Affected modules
    
    Usage:
        analyzer = ImpactAnalyzer()
        impact = analyzer.analyze_change(
            file_path="app.py",
            old_content=old_code,
            new_content=new_code,
        )
        print(f"Risk: {impact.risk_level}")
        print(f"Breaking changes: {impact.breaking_changes}")
    """
    
    def __init__(self):
        """Initialize ImpactAnalyzer."""
        pass
    
    def analyze_change(
        self,
        file_path: str,
        old_content: str,
        new_content: str,
    ) -> ChangeImpact:
        """
        Analyze the impact of a code change.
        
        Args:
            file_path: Path to the changed file
            old_content: Original file content
            new_content: Modified file content
            
        Returns:
            ChangeImpact with analysis results
        """
        # Detect changes
        old_functions = self.detect_function_definitions(old_content)
        new_functions = self.detect_function_definitions(new_content)
        
        old_classes = self.detect_class_definitions(old_content)
        new_classes = self.detect_class_definitions(new_content)
        
        old_imports = self.detect_imports(old_content)
        new_imports = self.detect_imports(new_content)
        
        # Find breaking changes
        removed_functions = self.detect_removed_functions(old_functions, new_functions)
        removed_classes = self.detect_removed_classes(old_classes, new_classes)
        signature_changes = self.detect_signature_changes(old_functions, new_functions)
        
        # Import changes
        added_imports = list(set(new_imports) - set(old_imports))
        removed_imports = list(set(old_imports) - set(new_imports))
        
        # Build breaking changes list
        breaking_changes = []
        for func in removed_functions:
            breaking_changes.append(f"Removed function: {func}")
        for cls in removed_classes:
            breaking_changes.append(f"Removed class: {cls}")
        for sig in signature_changes:
            breaking_changes.append(f"Signature changed: {sig}")
        
        # Assess risk
        risk_score, risk_level = self.assess_risk_level(
            removed_functions=removed_functions,
            removed_classes=removed_classes,
            signature_changes=signature_changes,
            removed_imports=removed_imports,
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            risk_level=risk_level,
            breaking_changes=breaking_changes,
            removed_imports=removed_imports,
        )
        
        # Detect affected modules
        affected_modules = self._detect_affected_modules(
            file_path=file_path,
            removed_imports=removed_imports,
        )
        
        return ChangeImpact(
            file_path=file_path,
            risk_level=risk_level,
            risk_score=risk_score,
            breaking_changes=breaking_changes,
            removed_functions=removed_functions,
            removed_classes=removed_classes,
            signature_changes=signature_changes,
            added_imports=added_imports,
            removed_imports=removed_imports,
            affected_modules=affected_modules,
            recommendations=recommendations,
        )
    
    def detect_imports(self, content: str) -> list[str]:
        """
        Detect import statements in Python code.
        
        Args:
            content: Python source code
            
        Returns:
            List of imported module names
        """
        imports = []
        
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except SyntaxError:
            # Fall back to regex if AST parsing fails
            import_pattern = r'^\s*(?:from\s+(\S+)\s+)?import\s+'
            for line in content.split('\n'):
                match = re.match(import_pattern, line)
                if match and match.group(1):
                    imports.append(match.group(1))
        
        return imports
    
    def detect_function_definitions(self, content: str) -> dict[str, str]:
        """
        Detect function definitions with signatures.
        
        Args:
            content: Python source code
            
        Returns:
            Dict of function_name -> signature
        """
        functions = {}
        
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Build signature
                    args = []
                    for arg in node.args.args:
                        args.append(arg.arg)
                    signature = f"{node.name}({', '.join(args)})"
                    functions[node.name] = signature
        except SyntaxError:
            # Fall back to regex
            func_pattern = r'^\s*def\s+(\w+)\s*\(([^)]*)\)'
            for line in content.split('\n'):
                match = re.match(func_pattern, line)
                if match:
                    func_name = match.group(1)
                    params = match.group(2)
                    functions[func_name] = f"{func_name}({params})"
        
        return functions
    
    def detect_class_definitions(self, content: str) -> list[str]:
        """
        Detect class definitions.
        
        Args:
            content: Python source code
            
        Returns:
            List of class names
        """
        classes = []
        
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
        except SyntaxError:
            # Fall back to regex
            class_pattern = r'^\s*class\s+(\w+)'
            for line in content.split('\n'):
                match = re.match(class_pattern, line)
                if match:
                    classes.append(match.group(1))
        
        return classes
    
    def detect_removed_functions(
        self,
        old_functions: dict[str, str],
        new_functions: dict[str, str],
    ) -> list[str]:
        """
        Detect functions that were removed.
        
        Args:
            old_functions: Functions in old code
            new_functions: Functions in new code
            
        Returns:
            List of removed function names
        """
        return [name for name in old_functions if name not in new_functions]
    
    def detect_removed_classes(
        self,
        old_classes: list[str],
        new_classes: list[str],
    ) -> list[str]:
        """
        Detect classes that were removed.
        
        Args:
            old_classes: Classes in old code
            new_classes: Classes in new code
            
        Returns:
            List of removed class names
        """
        return [name for name in old_classes if name not in new_classes]
    
    def detect_signature_changes(
        self,
        old_functions: dict[str, str],
        new_functions: dict[str, str],
    ) -> list[str]:
        """
        Detect function signature changes.
        
        Args:
            old_functions: Functions in old code
            new_functions: Functions in new code
            
        Returns:
            List of functions with changed signatures
        """
        changes = []
        
        for name in old_functions:
            if name in new_functions:
                old_sig = old_functions[name]
                new_sig = new_functions[name]
                if old_sig != new_sig:
                    changes.append(f"{old_sig} -> {new_sig}")
        
        return changes
    
    def assess_risk_level(
        self,
        removed_functions: list[str],
        removed_classes: list[str],
        signature_changes: list[str],
        removed_imports: list[str],
    ) -> tuple[int, str]:
        """
        Assess the risk level of changes.
        
        Risk scoring:
        - Removed function: +15 points
        - Removed class: +20 points
        - Signature change: +10 points
        - Removed import: +5 points
        
        Risk levels:
        - 0: safe
        - 1-20: low
        - 21-40: medium
        - 41-60: high
        - 61+: critical
        
        Args:
            removed_functions: List of removed functions
            removed_classes: List of removed classes
            signature_changes: List of signature changes
            removed_imports: List of removed imports
            
        Returns:
            Tuple of (risk_score, risk_level)
        """
        score = 0
        
        score += len(removed_functions) * 15
        score += len(removed_classes) * 20
        score += len(signature_changes) * 10
        score += len(removed_imports) * 5
        
        # Determine level
        if score == 0:
            level = "safe"
        elif score <= 20:
            level = "low"
        elif score <= 40:
            level = "medium"
        elif score <= 60:
            level = "high"
        else:
            level = "critical"
        
        return score, level
    
    def _generate_recommendations(
        self,
        risk_level: str,
        breaking_changes: list[str],
        removed_imports: list[str],
    ) -> list[str]:
        """
        Generate recommendations based on analysis.
        
        Args:
            risk_level: Assessed risk level
            breaking_changes: List of breaking changes
            removed_imports: List of removed imports
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if risk_level in ("high", "critical"):
            recommendations.append("⚠️ High risk changes detected - thorough testing recommended")
        
        if breaking_changes:
            recommendations.append("🔍 Search codebase for usage of removed/changed functions")
            recommendations.append("📝 Update documentation for API changes")
        
        if removed_imports:
            recommendations.append("🔗 Verify no code depends on removed imports")
        
        if risk_level == "safe":
            recommendations.append("✅ Changes appear safe - low risk of breaking existing code")
        
        return recommendations
    
    def _detect_affected_modules(
        self,
        file_path: str,
        removed_imports: list[str],
    ) -> list[str]:
        """
        Detect modules that may be affected by changes.
        
        Args:
            file_path: Path to changed file
            removed_imports: List of removed imports
            
        Returns:
            List of potentially affected module names
        """
        affected = []
        
        # Add the changed file's module
        path = Path(file_path)
        if path.suffix == ".py":
            module_name = path.stem
            affected.append(module_name)
        
        # Add removed imports as potentially affected
        affected.extend(removed_imports)
        
        return affected
