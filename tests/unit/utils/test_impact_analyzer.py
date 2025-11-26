"""Tests for Impact Analyzer utility."""

import pytest

from quirkllm.utils.impact_analyzer import ImpactAnalyzer, ChangeImpact


@pytest.fixture
def analyzer():
    """Create an ImpactAnalyzer instance."""
    return ImpactAnalyzer()


class TestImportDetection:
    """Test import detection."""
    
    def test_detects_simple_imports(self, analyzer):
        """Test detection of simple import statements."""
        content = """
import os
import sys
import json
"""
        imports = analyzer.detect_imports(content)
        
        assert "os" in imports
        assert "sys" in imports
        assert "json" in imports
    
    def test_detects_from_imports(self, analyzer):
        """Test detection of from...import statements."""
        content = """
from pathlib import Path
from typing import List, Dict
from collections.abc import Callable
"""
        imports = analyzer.detect_imports(content)
        
        assert "pathlib" in imports
        assert "typing" in imports
        assert "collections.abc" in imports
    
    def test_handles_syntax_errors_gracefully(self, analyzer):
        """Test that syntax errors don't crash import detection."""
        content = """
import os
def broken_function(
    # Missing closing paren
"""
        # Should not raise exception
        imports = analyzer.detect_imports(content)
        assert isinstance(imports, list)


class TestFunctionDetection:
    """Test function definition detection."""
    
    def test_detects_function_definitions(self, analyzer):
        """Test detection of function definitions."""
        content = """
def hello():
    pass

def greet(name):
    return f"Hello {name}"

def calculate(x, y, z):
    return x + y + z
"""
        functions = analyzer.detect_function_definitions(content)
        
        assert "hello" in functions
        assert "greet" in functions
        assert "calculate" in functions
        assert "hello()" == functions["hello"]
        assert "greet(name)" == functions["greet"]
        assert "calculate(x, y, z)" == functions["calculate"]
    
    def test_detects_functions_with_defaults(self, analyzer):
        """Test detection of functions with default arguments."""
        content = """
def send_email(to, subject, body=""):
    pass
"""
        functions = analyzer.detect_function_definitions(content)
        
        assert "send_email" in functions
        # Should capture parameter names
        assert "to" in functions["send_email"]
        assert "subject" in functions["send_email"]
    
    def test_handles_syntax_errors_in_functions(self, analyzer):
        """Test graceful handling of syntax errors."""
        content = """
def broken_function(
    # Missing closing paren and body
"""
        # Should not raise exception
        functions = analyzer.detect_function_definitions(content)
        assert isinstance(functions, dict)


class TestClassDetection:
    """Test class definition detection."""
    
    def test_detects_class_definitions(self, analyzer):
        """Test detection of class definitions."""
        content = """
class User:
    pass

class Product:
    def __init__(self):
        pass

class Order(BaseOrder):
    pass
"""
        classes = analyzer.detect_class_definitions(content)
        
        assert "User" in classes
        assert "Product" in classes
        assert "Order" in classes
    
    def test_handles_nested_classes(self, analyzer):
        """Test detection of nested classes."""
        content = """
class Outer:
    class Inner:
        pass
"""
        classes = analyzer.detect_class_definitions(content)
        
        # ast.walk finds all classes
        assert "Outer" in classes
        assert "Inner" in classes


class TestBreakingChangeDetection:
    """Test breaking change detection."""
    
    def test_detects_removed_functions(self, analyzer):
        """Test detection of removed functions."""
        old_functions = {
            "hello": "hello()",
            "greet": "greet(name)",
            "calculate": "calculate(x, y)",
        }
        new_functions = {
            "hello": "hello()",
            "greet": "greet(name)",
        }
        
        removed = analyzer.detect_removed_functions(old_functions, new_functions)
        
        assert "calculate" in removed
        assert "hello" not in removed
        assert "greet" not in removed
    
    def test_detects_removed_classes(self, analyzer):
        """Test detection of removed classes."""
        old_classes = ["User", "Product", "Order"]
        new_classes = ["User", "Product"]
        
        removed = analyzer.detect_removed_classes(old_classes, new_classes)
        
        assert "Order" in removed
        assert "User" not in removed
    
    def test_detects_signature_changes(self, analyzer):
        """Test detection of function signature changes."""
        old_functions = {
            "greet": "greet(name)",
            "calculate": "calculate(x, y)",
        }
        new_functions = {
            "greet": "greet(name, greeting)",  # Added parameter
            "calculate": "calculate(x, y)",    # Unchanged
        }
        
        changes = analyzer.detect_signature_changes(old_functions, new_functions)
        
        assert len(changes) == 1
        assert "greet(name) -> greet(name, greeting)" in changes


class TestRiskAssessment:
    """Test risk level assessment."""
    
    def test_assesses_safe_changes(self, analyzer):
        """Test that no changes results in safe risk level."""
        score, level = analyzer.assess_risk_level(
            removed_functions=[],
            removed_classes=[],
            signature_changes=[],
            removed_imports=[],
        )
        
        assert score == 0
        assert level == "safe"
    
    def test_assesses_low_risk(self, analyzer):
        """Test low risk assessment."""
        score, level = analyzer.assess_risk_level(
            removed_functions=[],
            removed_classes=[],
            signature_changes=["func1"],
            removed_imports=["unused_module"],
        )
        
        # 1 sig change (10) + 1 import (5) = 15
        assert score == 15
        assert level == "low"
    
    def test_assesses_medium_risk(self, analyzer):
        """Test medium risk assessment."""
        score, level = analyzer.assess_risk_level(
            removed_functions=["old_func"],
            removed_classes=[],
            signature_changes=["func1"],
            removed_imports=["module1"],
        )
        
        # 1 func (15) + 1 sig (10) + 1 import (5) = 30
        assert score == 30
        assert level == "medium"
    
    def test_assesses_high_risk(self, analyzer):
        """Test high risk assessment."""
        score, level = analyzer.assess_risk_level(
            removed_functions=["func1", "func2"],
            removed_classes=["OldClass"],
            signature_changes=[],
            removed_imports=[],
        )
        
        # 2 funcs (30) + 1 class (20) = 50
        assert score == 50
        assert level == "high"
    
    def test_assesses_critical_risk(self, analyzer):
        """Test critical risk assessment."""
        score, level = analyzer.assess_risk_level(
            removed_functions=["f1", "f2", "f3"],
            removed_classes=["C1", "C2"],
            signature_changes=["s1", "s2"],
            removed_imports=["m1"],
        )
        
        # 3 funcs (45) + 2 classes (40) + 2 sigs (20) + 1 import (5) = 110
        assert score == 110
        assert level == "critical"


class TestChangeAnalysis:
    """Test full change analysis."""
    
    def test_analyzes_simple_change(self, analyzer):
        """Test analysis of a simple code change."""
        old_content = """
import os

def hello():
    pass

def greet(name):
    return f"Hello {name}"
"""
        
        new_content = """
import os
import sys

def hello():
    pass

def greet(name, greeting="Hello"):
    return f"{greeting} {name}"
"""
        
        impact = analyzer.analyze_change("app.py", old_content, new_content)
        
        assert impact.file_path == "app.py"
        assert "sys" in impact.added_imports
        assert len(impact.signature_changes) == 1
        assert "greet" in impact.signature_changes[0]
    
    def test_detects_removed_function(self, analyzer):
        """Test detection of removed function."""
        old_content = """
def old_function():
    pass

def kept_function():
    pass
"""
        
        new_content = """
def kept_function():
    pass
"""
        
        impact = analyzer.analyze_change("app.py", old_content, new_content)
        
        assert "old_function" in impact.removed_functions
        assert "Removed function: old_function" in impact.breaking_changes
        assert impact.risk_level in ("low", "medium")
    
    def test_detects_removed_class(self, analyzer):
        """Test detection of removed class."""
        old_content = """
class OldClass:
    pass

class KeptClass:
    pass
"""
        
        new_content = """
class KeptClass:
    pass
"""
        
        impact = analyzer.analyze_change("models.py", old_content, new_content)
        
        assert "OldClass" in impact.removed_classes
        assert "Removed class: OldClass" in impact.breaking_changes
    
    def test_detects_removed_imports(self, analyzer):
        """Test detection of removed imports."""
        old_content = """
import os
import sys
import json
"""
        
        new_content = """
import os
import json
"""
        
        impact = analyzer.analyze_change("utils.py", old_content, new_content)
        
        assert "sys" in impact.removed_imports
    
    def test_generates_recommendations_for_safe_changes(self, analyzer):
        """Test that safe changes get appropriate recommendations."""
        old_content = "x = 1"
        new_content = "x = 2"
        
        impact = analyzer.analyze_change("config.py", old_content, new_content)
        
        assert impact.risk_level == "safe"
        assert any("safe" in rec.lower() for rec in impact.recommendations)
    
    def test_generates_recommendations_for_breaking_changes(self, analyzer):
        """Test recommendations for breaking changes."""
        old_content = """
def important_function():
    pass
"""
        
        new_content = ""
        
        impact = analyzer.analyze_change("api.py", old_content, new_content)
        
        assert len(impact.breaking_changes) > 0
        # Should recommend searching codebase
        assert any("search" in rec.lower() or "usage" in rec.lower() 
                  for rec in impact.recommendations)
    
    def test_includes_affected_modules(self, analyzer):
        """Test that affected modules are detected."""
        old_content = """
import requests
import json
"""
        
        new_content = """
import requests
"""
        
        impact = analyzer.analyze_change("api/client.py", old_content, new_content)
        
        # Should include the module itself
        assert "client" in impact.affected_modules
        # Should include removed imports
        assert "json" in impact.affected_modules
    
    def test_handles_complex_refactoring(self, analyzer):
        """Test analysis of complex refactoring."""
        old_content = """
from typing import List

class User:
    pass

class Product:
    pass

def get_users():
    pass

def get_products(limit):
    pass

def delete_user(user_id):
    pass
"""
        
        new_content = """
from typing import List, Optional

class User:
    pass

class Product:
    pass

def get_users(include_inactive=False):
    pass

def get_products(limit, offset=0):
    pass
"""
        
        impact = analyzer.analyze_change("models.py", old_content, new_content)
        
        # Should detect removed function
        assert "delete_user" in impact.removed_functions
        
        # Should detect signature changes
        assert len(impact.signature_changes) == 2
        
        # Should have medium/high risk
        assert impact.risk_level in ("medium", "high")
        
        # Should have recommendations
        assert len(impact.recommendations) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_handles_empty_files(self, analyzer):
        """Test analysis of empty files."""
        impact = analyzer.analyze_change("empty.py", "", "")
        
        assert impact.risk_level == "safe"
        assert len(impact.breaking_changes) == 0
    
    def test_handles_syntax_errors(self, analyzer):
        """Test graceful handling of syntax errors."""
        old_content = """
def broken_function(
    # Missing closing paren
"""
        new_content = """
def fixed_function():
    pass
"""
        
        # Should not crash
        impact = analyzer.analyze_change("broken.py", old_content, new_content)
        
        assert isinstance(impact, ChangeImpact)
    
    def test_handles_identical_content(self, analyzer):
        """Test that identical content results in safe assessment."""
        content = """
def hello():
    pass
"""
        
        impact = analyzer.analyze_change("same.py", content, content)
        
        assert impact.risk_level == "safe"
        assert impact.risk_score == 0
        assert len(impact.breaking_changes) == 0
