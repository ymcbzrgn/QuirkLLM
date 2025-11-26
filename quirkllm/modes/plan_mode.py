"""
Plan Mode - Read-only architecture and planning mode.

Plan Mode is designed for pre-implementation analysis, architecture
documentation, and strategic planning without making any code changes.
All outputs are saved as markdown files to .quirkllm/plans/.

Behavior:
- Read-only mode (no file edits allowed)
- Generates structured plans and documentation
- Saves outputs to .quirkllm/plans/ directory
- Supports multiple plan types (refactoring, architecture, features, etc.)

Use Cases:
- Pre-refactoring analysis and planning
- Architecture decision documentation
- Feature specification and design
- Technical debt assessment
- Project documentation generation

Output Formats:
- Markdown files with metadata headers
- Timestamped filenames for version tracking
- Structured sections (Problem, Analysis, Approach, Steps, etc.)
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from quirkllm.modes.base import (
    ModeBase,
    ModeType,
    ModeConfig,
    ActionRequest,
    ActionResult,
)


# Plan Mode System Instruction - Expert-level planning guidance
PLAN_MODE_SYSTEM_INSTRUCTION = """You are QuirkLLM Plan Mode - an expert software architect and technical planner.

Your role is to analyze codebases, understand requirements, and generate comprehensive, actionable plans without making any code changes. You operate in read-only mode, focusing on strategic thinking and detailed documentation.

## Core Principles

1. **Think Before Acting**: Always analyze thoroughly before suggesting changes
2. **Structured Approach**: Use clear sections and hierarchical organization
3. **Risk Awareness**: Identify potential issues, edge cases, and technical debt
4. **Pragmatic Planning**: Balance ideal architecture with practical constraints
5. **Evidence-Based**: Ground recommendations in codebase analysis

## Planning Framework

When generating any plan, follow this structure:

### 1. Executive Summary (2-3 sentences)
- What is the goal?
- Why is it important?
- What's the expected outcome?

### 2. Current State Analysis
- What exists now?
- What patterns/architectures are in use?
- What are the pain points?
- What technical debt exists?

### 3. Problem Definition
- Core problem statement
- Root causes (not just symptoms)
- Impact assessment (who/what is affected?)
- Constraints and dependencies

### 4. Proposed Solution
- High-level approach
- Key design decisions
- Architecture patterns to use
- Trade-offs and alternatives considered

### 5. Implementation Plan
- Phased approach (break into logical steps)
- For each phase:
  * Goal of this phase
  * Files to modify (with reasons)
  * New files to create (with purposes)
  * Tests to write/update
  * Estimated complexity
  * Dependencies on other phases

### 6. Risk Assessment
- Technical risks (compatibility, performance, scalability)
- Implementation risks (complexity, time, team knowledge)
- Mitigation strategies for each risk

### 7. Testing Strategy
- Unit tests (what to test)
- Integration tests (how components interact)
- Edge cases to cover
- Regression testing considerations

### 8. Success Criteria
- How do we know when it's done?
- Measurable outcomes
- Quality gates

### 9. Rollback Plan
- What if something goes wrong?
- How to revert safely?
- Data/state considerations

## Plan Type Guidelines

### Refactoring Plans
- Focus: Code quality, maintainability, performance
- Include: Before/after comparisons, dependency graphs
- Consider: Breaking changes, backwards compatibility
- Emphasize: Incremental approach, continuous testing

### Architecture Plans
- Focus: System design, component relationships, scalability
- Include: Diagrams (suggest Mermaid), data flow, APIs
- Consider: Future growth, maintenance burden, team skills
- Emphasize: Separation of concerns, loose coupling

### Feature Plans
- Focus: New functionality, user experience, integration
- Include: Requirements, acceptance criteria, user stories
- Consider: Existing architecture, API design, data models
- Emphasize: MVP approach, iterative delivery

### Bugfix Plans
- Focus: Root cause, reliable fix, prevention
- Include: Reproduction steps, debugging strategy
- Consider: Similar bugs, underlying patterns
- Emphasize: Testing to prevent regression

### Technical Debt Plans
- Focus: Long-term health, modernization, cleanup
- Include: Debt inventory, priority matrix
- Consider: ROI of fixes, opportunity cost
- Emphasize: Sustainable pace, quick wins

## Code Analysis Techniques

When analyzing code for planning:

1. **Pattern Recognition**: Identify design patterns, anti-patterns
2. **Dependency Mapping**: Understand module relationships
3. **Complexity Assessment**: Spot overly complex areas
4. **Test Coverage**: Note gaps in testing
5. **Performance Hotspots**: Identify potential bottlenecks
6. **Security Concerns**: Flag potential vulnerabilities
7. **Maintainability**: Assess code clarity and documentation

## Best Practices

- **Be Specific**: "Refactor UserService.authenticate()" not "improve auth"
- **Show Trade-offs**: Explain why chosen approach beats alternatives
- **Provide Context**: Why this matters, what problem it solves
- **Think Long-term**: Consider maintenance, not just initial development
- **Stay Pragmatic**: Perfect is enemy of good; ship value iteratively
- **Include Examples**: Show concrete code snippets in plans when helpful
- **Reference Standards**: Cite relevant patterns, principles (SOLID, DRY, etc.)

## Language & Tone

- **Professional yet accessible**: Technical but clear
- **Confident but humble**: Strong opinions, loosely held
- **Action-oriented**: Focus on next steps
- **Empathetic**: Consider developer experience

## Output Format

All plans should be:
- **Well-structured**: Clear headings, logical flow
- **Scannable**: Bullet points, numbered lists, tables
- **Comprehensive**: Cover all important aspects
- **Actionable**: Developers can execute directly from the plan

## Metadata to Include

Always generate plans with:
- Title (clear, descriptive)
- Plan type (refactoring, architecture, feature, bugfix, debt)
- Estimated scope (small/medium/large)
- Priority (low/medium/high/critical)
- Key stakeholders
- Related files/modules

Remember: You don't write code in Plan Mode. You create the roadmap that others will follow. Be the architect, strategist, and technical guide. Your plans should inspire confidence and provide clarity."""


class PlanMode(ModeBase):
    """
    Plan Mode - Generate plans and documentation without modifying code.
    
    Plan Mode enforces read-only access to the codebase while generating
    structured planning documents. All plans are saved to .quirkllm/plans/
    with timestamps and metadata.
    
    Read-Only Enforcement:
    - Blocks: file_edit, delete, command, write operations
    - Allows: read_file, analyze, list_files, generate_plan
    
    Plan Types Supported:
    - refactoring: Code refactoring plans
    - architecture: System architecture documentation
    - feature: New feature specifications
    - bugfix: Bug analysis and fix plans
    - general: General purpose planning
    
    Attributes:
        console: Rich console for UI output
        plans_dir: Path to plan output directory
        session_stats: Statistics tracking for the session
    """
    
    def __init__(self, output_dir: str = ".quirkllm/plans", **kwargs: Any) -> None:
        """
        Initialize Plan mode.
        
        Args:
            output_dir: Directory for plan file output
            **kwargs: Additional configuration options
        """
        # Plan config: read-only, no confirmations needed
        config = ModeConfig(
            auto_confirm=True,  # No confirmations (read-only)
            allow_file_edits=False,
            allow_destructive=False,
            background_watch=False,
            diff_display=False,
            plan_output_dir=output_dir,
        )
        super().__init__(ModeType.PLAN, config)
        self.console = Console()
        self.plans_dir = Path(output_dir)
        
        # Create plans directory if it doesn't exist
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        
        # Session statistics
        self.session_stats: Dict[str, Any] = {
            "plans_generated": 0,
            "plan_files": [],
        }
    
    def activate(self) -> None:
        """
        Activate Plan mode and display welcome message.
        
        Shows plan output directory and read-only status.
        Resets session statistics.
        """
        self.active = True
        
        # Reset session stats
        self.session_stats = {
            "plans_generated": 0,
            "plan_files": [],
        }
        
        # Display activation panel
        welcome_panel = Panel(
            "[bold blue]PLAN MODE ACTIVATED[/]\n\n"
            "[blue]Read-only planning and documentation mode[/]\n\n"
            "📋 [green]Generate plans[/] for refactoring, features, architecture\n"
            "📁 [green]Output directory[/]: {}\n"
            "🔒 [yellow]Read-only[/] - No file modifications allowed\n\n"
            "[dim]Switch to Chat/YAMI mode to execute plans[/]".format(
                self.plans_dir.absolute()
            ),
            title="📋 Plan Mode",
            border_style="blue",
        )
        self.console.print(welcome_panel)
    
    def deactivate(self) -> None:
        """
        Deactivate Plan mode and display session summary.
        
        Shows list of generated plans with file paths.
        """
        self.active = False
        
        if self.session_stats["plans_generated"] > 0:
            # Display session summary
            summary_table = Table(title="Plan Mode Session Summary", show_header=True)
            summary_table.add_column("Plan File", style="cyan")
            summary_table.add_column("Type", style="green")
            
            for plan_info in self.session_stats["plan_files"]:
                summary_table.add_row(
                    plan_info["filename"],
                    plan_info.get("plan_type", "general")
                )
            
            self.console.print("\n")
            self.console.print(summary_table)
            self.console.print(
                f"\n[green]✓ {self.session_stats['plans_generated']} plan(s) generated[/]"
            )
            self.console.print(f"[dim]Saved to: {self.plans_dir.absolute()}[/]")
        else:
            self.console.print("\n[dim]No plans generated in this session[/]")
        
        self.console.print("[dim]Plan mode deactivated[/]\n")
    
    def handle_action(self, request: ActionRequest) -> ActionResult:
        """
        Handle action with read-only enforcement.
        
        Flow:
        1. Block any write operations (file_edit, delete, command)
        2. Allow read operations (read_file, analyze, list_files)
        3. Handle plan generation requests
        4. Track statistics
        
        Args:
            request: The action to handle
            
        Returns:
            ActionResult with execution status
        """
        # Block write operations
        if request.action_type in [
            "file_edit",
            "delete",
            "command",
            "write_file",
            "create_file",
            "move_file",
        ]:
            self._display_read_only_message(request)
            return ActionResult(
                success=False,
                message=(
                    f"Plan Mode is read-only. Action '{request.action_type}' blocked. "
                    "Use /mode chat or /mode yami to make changes."
                ),
                warnings=[
                    "Read-only mode enforced",
                    "Switch modes to execute plans",
                ],
            )
        
        # Handle plan generation
        if request.action_type == "generate_plan":
            result = self._generate_plan(request)
            if result.success:
                self.session_stats["plans_generated"] += 1
            return result
        
        # Allow read operations
        if request.action_type in ["read_file", "analyze", "list_files", "search"]:
            self.console.print(
                f"[dim]✓ Read operation: {request.action_type}[/]"
            )
            return ActionResult(
                success=True,
                message=f"Read operation '{request.action_type}' allowed",
                details=request.details,
            )
        
        # Unknown action type
        return ActionResult(
            success=False,
            message=f"Action '{request.action_type}' not supported in Plan Mode",
            warnings=["Plan mode supports: generate_plan, read_file, analyze, list_files"],
        )
    
    def get_prompt_indicator(self) -> str:
        """
        Get the prompt indicator for Plan mode.
        
        Returns:
            Clipboard/document emoji indicating planning mode
        """
        return "📋"
    
    def get_system_prompt(self) -> str:
        """
        Get the system instruction for Plan mode.
        
        This prompt guides the LLM to behave as an expert software architect,
        generating comprehensive, actionable plans with proper structure,
        risk assessment, and implementation guidance.
        
        The system prompt emphasizes:
        - Structured planning methodology
        - Read-only analysis focus
        - Risk-aware recommendations
        - Evidence-based decisions
        - Actionable outputs
        
        Returns:
            Complete system instruction for planning behavior
        """
        return PLAN_MODE_SYSTEM_INSTRUCTION
    
    def _display_read_only_message(self, request: ActionRequest) -> None:
        """
        Display read-only blocking message.
        
        Args:
            request: The blocked action request
        """
        blocked_panel = Panel(
            f"[bold yellow]READ-ONLY MODE[/]\n\n"
            f"Action: [red]{request.action_type}[/]\n"
            f"Target: [red]{request.target}[/]\n\n"
            f"[yellow]Plan Mode does not modify files.[/]\n"
            f"Plans are saved to: [cyan]{self.plans_dir}[/]\n\n"
            f"[dim]To execute plans:[/]\n"
            f"  • Use [cyan]/mode chat[/] for interactive execution\n"
            f"  • Use [cyan]/mode yami[/] for fast execution",
            title="🔒 Read-Only",
            border_style="yellow",
        )
        self.console.print(blocked_panel)
    
    def _generate_plan(self, request: ActionRequest) -> ActionResult:
        """
        Generate and save a structured plan.
        
        Args:
            request: The plan generation request
            
        Returns:
            ActionResult with plan file path
        """
        # Extract plan details
        details = request.details or {}
        plan_type = details.get("plan_type", "general")
        title = details.get("title", "Plan")
        content = details.get("content", "")
        
        # Generate sanitized filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = self._sanitize_filename(title)
        filename = f"{plan_type}_{safe_title}_{timestamp}.md"
        filepath = self.plans_dir / filename
        
        # Format plan with metadata
        plan_content = self._format_plan(title, content, plan_type, details)
        
        # Save plan to file
        try:
            filepath.write_text(plan_content, encoding="utf-8")
            
            # Track in session
            self.session_stats["plan_files"].append({
                "filename": filename,
                "filepath": str(filepath),
                "plan_type": plan_type,
                "title": title,
            })
            
            # Display success message
            self.console.print(
                f"[green]✓ Plan saved:[/] [cyan]{filename}[/]"
            )
            self.console.print(
                f"[dim]  Location: {filepath.absolute()}[/]"
            )
            
            return ActionResult(
                success=True,
                message=f"Plan saved to {filename}",
                details={
                    "filepath": str(filepath),
                    "filename": filename,
                    "content": plan_content,
                },
            )
        
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to save plan: {str(e)}",
                errors=[str(e)],
            )
    
    def _format_plan(
        self, title: str, content: str, plan_type: str, details: Dict[str, Any]
    ) -> str:
        """
        Format plan with metadata and structure.
        
        Args:
            title: Plan title
            content: Plan content
            plan_type: Type of plan (refactoring, architecture, etc.)
            details: Additional metadata
            
        Returns:
            Formatted markdown plan
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Build metadata section
        metadata = f"""# {title}

**Generated:** {timestamp}  
**Type:** {plan_type.title()}  
**Mode:** Plan Mode (Read-only)
"""
        
        # Add optional metadata
        if "author" in details:
            metadata += f"**Author:** {details['author']}  \n"
        if "tags" in details:
            tags = ", ".join(details["tags"]) if isinstance(details["tags"], list) else details["tags"]
            metadata += f"**Tags:** {tags}  \n"
        
        metadata += "\n---\n\n"
        
        # Format content
        formatted_content = content
        
        # Add footer
        footer = "\n\n---\n\n*Generated by QuirkLLM Plan Mode*\n"
        
        return metadata + formatted_content + footer
    
    def _sanitize_filename(self, title: str) -> str:
        """
        Sanitize title for use in filename.
        
        Args:
            title: Original title
            
        Returns:
            Sanitized filename-safe string
        """
        # Remove or replace unsafe characters
        safe = re.sub(r'[^\w\s-]', '', title.lower())
        safe = re.sub(r'[-\s]+', '_', safe)
        # Limit length
        return safe[:50] if len(safe) > 50 else safe
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get current session statistics.
        
        Returns:
            Dictionary with plans_generated and plan_files
        """
        return self.session_stats.copy()
