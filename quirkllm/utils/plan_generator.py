"""
Plan Generator Utility.

Generates structured plans for refactoring, architecture, and feature development.
Used by Plan Mode to create comprehensive markdown documentation.
"""

from datetime import datetime
from pathlib import Path
from typing import Any


class PlanGenerator:
    """
    Generate structured plans in markdown format.
    
    Provides templates for:
    - Refactoring plans (before/after, steps, risks)
    - Architecture documentation (components, data flow, tech stack)
    - Feature plans (requirements, implementation, testing)
    - Mermaid diagrams (flowchart, sequence, class)
    
    All plans are formatted with:
    - Metadata headers (generated timestamp, type, tags)
    - Structured sections
    - Markdown formatting
    - Optional Mermaid diagrams
    
    Usage:
        generator = PlanGenerator()
        plan = generator.generate_refactoring_plan(
            title="Refactor Auth System",
            current_state="Scattered auth logic",
            target_state="Centralized AuthService",
            steps=["Extract methods", "Create service", "Migrate callers"],
            risks=["Potential breaking changes"],
        )
    """
    
    def __init__(self):
        """Initialize PlanGenerator."""
        pass
    
    def generate_refactoring_plan(
        self,
        title: str,
        current_state: str,
        target_state: str,
        steps: list[str],
        risks: list[str] | None = None,
        affected_files: list[str] | None = None,
        estimated_time: str | None = None,
    ) -> str:
        """
        Generate a refactoring plan.
        
        Args:
            title: Refactoring title
            current_state: Description of current code state
            target_state: Description of desired state
            steps: List of refactoring steps
            risks: Optional list of potential risks
            affected_files: Optional list of files to be modified
            estimated_time: Optional time estimate
            
        Returns:
            Formatted markdown plan
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        plan = f"""# {title}

**Generated:** {timestamp}  
**Type:** Refactoring Plan  
**Status:** Draft

## Current State

{current_state}

## Target State

{target_state}

## Implementation Steps

"""
        
        for i, step in enumerate(steps, 1):
            plan += f"{i}. {step}\n"
        
        if risks:
            plan += "\n## Risks and Considerations\n\n"
            for risk in risks:
                plan += f"- ⚠️ {risk}\n"
        
        if affected_files:
            plan += "\n## Affected Files\n\n"
            for file in affected_files:
                plan += f"- `{file}`\n"
        
        if estimated_time:
            plan += f"\n## Estimated Time\n\n{estimated_time}\n"
        
        plan += "\n## Checklist\n\n"
        for step in steps:
            plan += f"- [ ] {step}\n"
        
        return plan
    
    def generate_architecture_doc(
        self,
        title: str,
        overview: str,
        components: dict[str, str],
        data_flow: list[str] | None = None,
        tech_stack: dict[str, str] | None = None,
        diagram: str | None = None,
    ) -> str:
        """
        Generate an architecture documentation.
        
        Args:
            title: Documentation title
            overview: System overview
            components: Dict of component name -> description
            data_flow: Optional list of data flow descriptions
            tech_stack: Optional dict of technology -> purpose
            diagram: Optional Mermaid diagram code
            
        Returns:
            Formatted markdown documentation
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        doc = f"""# {title}

**Generated:** {timestamp}  
**Type:** Architecture Documentation  

## Overview

{overview}

## Components

"""
        
        for name, description in components.items():
            doc += f"### {name}\n\n{description}\n\n"
        
        if data_flow:
            doc += "## Data Flow\n\n"
            for i, flow in enumerate(data_flow, 1):
                doc += f"{i}. {flow}\n"
            doc += "\n"
        
        if tech_stack:
            doc += "## Tech Stack\n\n"
            for tech, purpose in tech_stack.items():
                doc += f"- **{tech}**: {purpose}\n"
            doc += "\n"
        
        if diagram:
            doc += "## Architecture Diagram\n\n```mermaid\n"
            doc += diagram
            doc += "\n```\n"
        
        return doc
    
    def generate_feature_plan(
        self,
        title: str,
        description: str,
        requirements: list[str],
        implementation_steps: list[str],
        testing_strategy: str | None = None,
        dependencies: list[str] | None = None,
        acceptance_criteria: list[str] | None = None,
    ) -> str:
        """
        Generate a feature development plan.
        
        Args:
            title: Feature title
            description: Feature description
            requirements: List of functional requirements
            implementation_steps: List of implementation steps
            testing_strategy: Optional testing approach
            dependencies: Optional list of dependencies
            acceptance_criteria: Optional list of acceptance criteria
            
        Returns:
            Formatted markdown plan
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        plan = f"""# {title}

**Generated:** {timestamp}  
**Type:** Feature Plan  
**Status:** Planning

## Description

{description}

## Requirements

"""
        
        for i, req in enumerate(requirements, 1):
            plan += f"{i}. {req}\n"
        
        if dependencies:
            plan += "\n## Dependencies\n\n"
            for dep in dependencies:
                plan += f"- {dep}\n"
        
        plan += "\n## Implementation Steps\n\n"
        for i, step in enumerate(implementation_steps, 1):
            plan += f"{i}. {step}\n"
        
        if testing_strategy:
            plan += f"\n## Testing Strategy\n\n{testing_strategy}\n"
        
        if acceptance_criteria:
            plan += "\n## Acceptance Criteria\n\n"
            for criterion in acceptance_criteria:
                plan += f"- [ ] {criterion}\n"
        
        plan += "\n## Implementation Checklist\n\n"
        for step in implementation_steps:
            plan += f"- [ ] {step}\n"
        
        return plan
    
    def create_mermaid_flowchart(
        self,
        nodes: list[tuple[str, str]],
        edges: list[tuple[str, str, str | None]],
    ) -> str:
        """
        Create a Mermaid flowchart diagram.
        
        Args:
            nodes: List of (node_id, label) tuples
            edges: List of (from_id, to_id, label) tuples
            
        Returns:
            Mermaid flowchart code
        """
        diagram = "flowchart TD\n"
        
        for node_id, label in nodes:
            diagram += f"    {node_id}[{label}]\n"
        
        diagram += "\n"
        
        for from_id, to_id, label in edges:
            if label:
                diagram += f"    {from_id} -->|{label}| {to_id}\n"
            else:
                diagram += f"    {from_id} --> {to_id}\n"
        
        return diagram
    
    def create_mermaid_sequence(
        self,
        participants: list[str],
        interactions: list[tuple[str, str, str]],
    ) -> str:
        """
        Create a Mermaid sequence diagram.
        
        Args:
            participants: List of participant names
            interactions: List of (from, to, message) tuples
            
        Returns:
            Mermaid sequence diagram code
        """
        diagram = "sequenceDiagram\n"
        
        for participant in participants:
            diagram += f"    participant {participant}\n"
        
        diagram += "\n"
        
        for from_p, to_p, message in interactions:
            diagram += f"    {from_p}->> {to_p}: {message}\n"
        
        return diagram
    
    def create_mermaid_class(
        self,
        classes: dict[str, dict[str, Any]],
        relationships: list[tuple[str, str, str]],
    ) -> str:
        """
        Create a Mermaid class diagram.
        
        Args:
            classes: Dict of class_name -> {attributes: [...], methods: [...]}
            relationships: List of (class1, class2, relation_type) tuples
                          relation_type can be: "inherits", "implements", "has", "uses"
            
        Returns:
            Mermaid class diagram code
        """
        diagram = "classDiagram\n"
        
        for class_name, members in classes.items():
            diagram += f"    class {class_name} {{\n"
            
            if "attributes" in members:
                for attr in members["attributes"]:
                    diagram += f"        +{attr}\n"
            
            if "methods" in members:
                for method in members["methods"]:
                    diagram += f"        +{method}()\n"
            
            diagram += "    }\n\n"
        
        for class1, class2, relation in relationships:
            if relation == "inherits":
                diagram += f"    {class1} <|-- {class2}\n"
            elif relation == "implements":
                diagram += f"    {class1} <|.. {class2}\n"
            elif relation == "has":
                diagram += f"    {class1} *-- {class2}\n"
            elif relation == "uses":
                diagram += f"    {class1} --> {class2}\n"
        
        return diagram
    
    def save_plan(self, plan: str, filename: str, output_dir: Path | None = None) -> Path:
        """
        Save plan to file.
        
        Args:
            plan: Plan content
            filename: Output filename (without path)
            output_dir: Optional output directory (defaults to .quirkllm/plans/)
            
        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = Path.home() / ".quirkllm" / "plans"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize filename
        safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
        if not safe_filename.endswith('.md'):
            safe_filename += '.md'
        
        filepath = output_dir / safe_filename
        filepath.write_text(plan, encoding="utf-8")
        
        return filepath
