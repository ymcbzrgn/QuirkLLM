"""Tests for Plan Generator utility."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from quirkllm.utils.plan_generator import PlanGenerator


@pytest.fixture
def generator():
    """Create a PlanGenerator instance."""
    return PlanGenerator()


class TestRefactoringPlans:
    """Test refactoring plan generation."""
    
    def test_generates_basic_refactoring_plan(self, generator):
        """Test basic refactoring plan generation."""
        plan = generator.generate_refactoring_plan(
            title="Refactor Auth System",
            current_state="Scattered authentication logic across controllers",
            target_state="Centralized AuthService with clean API",
            steps=["Extract auth methods", "Create AuthService", "Update controllers"],
        )
        
        assert "Refactor Auth System" in plan
        assert "Current State" in plan
        assert "Target State" in plan
        assert "Implementation Steps" in plan
        assert "Scattered authentication" in plan
        assert "Centralized AuthService" in plan
        assert "1. Extract auth methods" in plan
        assert "2. Create AuthService" in plan
        assert "3. Update controllers" in plan
        assert "Checklist" in plan
    
    def test_refactoring_plan_with_risks(self, generator):
        """Test refactoring plan with risks."""
        plan = generator.generate_refactoring_plan(
            title="Database Migration",
            current_state="SQLite database",
            target_state="PostgreSQL database",
            steps=["Backup data", "Setup Postgres", "Migrate schema", "Import data"],
            risks=["Data loss possible", "Downtime required", "Schema incompatibilities"],
        )
        
        assert "Risks and Considerations" in plan
        assert "⚠️ Data loss possible" in plan
        assert "⚠️ Downtime required" in plan
        assert "⚠️ Schema incompatibilities" in plan
    
    def test_refactoring_plan_with_affected_files(self, generator):
        """Test refactoring plan with affected files list."""
        plan = generator.generate_refactoring_plan(
            title="Extract Utility Functions",
            current_state="Utility functions scattered",
            target_state="Centralized utils module",
            steps=["Identify utils", "Create module", "Move functions"],
            affected_files=["app.py", "models.py", "utils.py"],
        )
        
        assert "Affected Files" in plan
        assert "`app.py`" in plan
        assert "`models.py`" in plan
        assert "`utils.py`" in plan
    
    def test_refactoring_plan_with_time_estimate(self, generator):
        """Test refactoring plan with time estimate."""
        plan = generator.generate_refactoring_plan(
            title="Performance Optimization",
            current_state="Slow queries",
            target_state="Optimized with caching",
            steps=["Profile queries", "Add indexes", "Implement caching"],
            estimated_time="2-3 days",
        )
        
        assert "Estimated Time" in plan
        assert "2-3 days" in plan
    
    def test_refactoring_plan_includes_metadata(self, generator):
        """Test refactoring plan includes metadata."""
        plan = generator.generate_refactoring_plan(
            title="Test Plan",
            current_state="A",
            target_state="B",
            steps=["Step 1"],
        )
        
        assert "**Generated:**" in plan
        assert "**Type:** Refactoring Plan" in plan
        assert "**Status:** Draft" in plan


class TestArchitectureDocs:
    """Test architecture documentation generation."""
    
    def test_generates_basic_architecture_doc(self, generator):
        """Test basic architecture doc generation."""
        doc = generator.generate_architecture_doc(
            title="API Architecture",
            overview="RESTful API with microservices architecture",
            components={
                "API Gateway": "Routes requests to services",
                "Auth Service": "Handles authentication and authorization",
                "Data Service": "Manages database operations",
            },
        )
        
        assert "API Architecture" in doc
        assert "Overview" in doc
        assert "RESTful API with microservices" in doc
        assert "Components" in doc
        assert "### API Gateway" in doc
        assert "Routes requests to services" in doc
        assert "### Auth Service" in doc
        assert "### Data Service" in doc
    
    def test_architecture_doc_with_data_flow(self, generator):
        """Test architecture doc with data flow."""
        doc = generator.generate_architecture_doc(
            title="System Architecture",
            overview="Layered architecture",
            components={"UI": "User interface", "API": "Backend API"},
            data_flow=[
                "User interacts with UI",
                "UI sends request to API",
                "API processes request",
                "Response sent to UI",
            ],
        )
        
        assert "Data Flow" in doc
        assert "1. User interacts with UI" in doc
        assert "2. UI sends request to API" in doc
        assert "3. API processes request" in doc
        assert "4. Response sent to UI" in doc
    
    def test_architecture_doc_with_tech_stack(self, generator):
        """Test architecture doc with tech stack."""
        doc = generator.generate_architecture_doc(
            title="Tech Stack",
            overview="Modern web stack",
            components={"Frontend": "User interface"},
            tech_stack={
                "React": "UI framework",
                "Node.js": "Backend runtime",
                "PostgreSQL": "Database",
                "Redis": "Caching layer",
            },
        )
        
        assert "Tech Stack" in doc
        assert "**React**: UI framework" in doc
        assert "**Node.js**: Backend runtime" in doc
        assert "**PostgreSQL**: Database" in doc
        assert "**Redis**: Caching layer" in doc
    
    def test_architecture_doc_with_diagram(self, generator):
        """Test architecture doc with Mermaid diagram."""
        diagram = generator.create_mermaid_flowchart(
            nodes=[("A", "Client"), ("B", "Server")],
            edges=[("A", "B", "Request")],
        )
        
        doc = generator.generate_architecture_doc(
            title="System Flow",
            overview="Client-server architecture",
            components={"Client": "Frontend", "Server": "Backend"},
            diagram=diagram,
        )
        
        assert "Architecture Diagram" in doc
        assert "```mermaid" in doc
        assert "flowchart TD" in doc
        assert "A[Client]" in doc
        assert "B[Server]" in doc
    
    def test_architecture_doc_includes_metadata(self, generator):
        """Test architecture doc includes metadata."""
        doc = generator.generate_architecture_doc(
            title="Test Doc",
            overview="Test overview",
            components={"A": "Component A"},
        )
        
        assert "**Generated:**" in doc
        assert "**Type:** Architecture Documentation" in doc


class TestFeaturePlans:
    """Test feature plan generation."""
    
    def test_generates_basic_feature_plan(self, generator):
        """Test basic feature plan generation."""
        plan = generator.generate_feature_plan(
            title="User Authentication",
            description="Implement user login and registration",
            requirements=["User can register", "User can login", "Password hashing"],
            implementation_steps=["Create User model", "Add auth routes", "Implement JWT"],
        )
        
        assert "User Authentication" in plan
        assert "Description" in plan
        assert "Implement user login" in plan
        assert "Requirements" in plan
        assert "1. User can register" in plan
        assert "2. User can login" in plan
        assert "3. Password hashing" in plan
        assert "Implementation Steps" in plan
        assert "1. Create User model" in plan
        assert "Implementation Checklist" in plan
    
    def test_feature_plan_with_testing_strategy(self, generator):
        """Test feature plan with testing strategy."""
        plan = generator.generate_feature_plan(
            title="Payment Integration",
            description="Add Stripe payment processing",
            requirements=["Accept credit cards", "Handle webhooks"],
            implementation_steps=["Setup Stripe", "Create payment endpoint"],
            testing_strategy="Unit tests for payment logic, integration tests with Stripe test mode",
        )
        
        assert "Testing Strategy" in plan
        assert "Unit tests for payment logic" in plan
        assert "integration tests with Stripe" in plan
    
    def test_feature_plan_with_dependencies(self, generator):
        """Test feature plan with dependencies."""
        plan = generator.generate_feature_plan(
            title="Email Notifications",
            description="Send email notifications to users",
            requirements=["Send welcome email", "Send password reset"],
            implementation_steps=["Setup email service", "Create templates"],
            dependencies=["User model", "Email service account", "Template engine"],
        )
        
        assert "Dependencies" in plan
        assert "- User model" in plan
        assert "- Email service account" in plan
        assert "- Template engine" in plan
    
    def test_feature_plan_with_acceptance_criteria(self, generator):
        """Test feature plan with acceptance criteria."""
        plan = generator.generate_feature_plan(
            title="Search Feature",
            description="Add search functionality",
            requirements=["Search by keyword"],
            implementation_steps=["Add search endpoint"],
            acceptance_criteria=[
                "User can search by keyword",
                "Results returned in <200ms",
                "Pagination works correctly",
            ],
        )
        
        assert "Acceptance Criteria" in plan
        assert "- [ ] User can search by keyword" in plan
        assert "- [ ] Results returned in <200ms" in plan
        assert "- [ ] Pagination works correctly" in plan
    
    def test_feature_plan_includes_metadata(self, generator):
        """Test feature plan includes metadata."""
        plan = generator.generate_feature_plan(
            title="Test Feature",
            description="Test description",
            requirements=["Req 1"],
            implementation_steps=["Step 1"],
        )
        
        assert "**Generated:**" in plan
        assert "**Type:** Feature Plan" in plan
        assert "**Status:** Planning" in plan


class TestMermaidDiagrams:
    """Test Mermaid diagram generation."""
    
    def test_creates_flowchart(self, generator):
        """Test flowchart creation."""
        diagram = generator.create_mermaid_flowchart(
            nodes=[
                ("start", "Start"),
                ("process", "Process Data"),
                ("end", "End"),
            ],
            edges=[
                ("start", "process", None),
                ("process", "end", "Success"),
            ],
        )
        
        assert "flowchart TD" in diagram
        assert "start[Start]" in diagram
        assert "process[Process Data]" in diagram
        assert "end[End]" in diagram
        assert "start --> process" in diagram
        assert "process -->|Success| end" in diagram
    
    def test_creates_sequence_diagram(self, generator):
        """Test sequence diagram creation."""
        diagram = generator.create_mermaid_sequence(
            participants=["User", "Frontend", "Backend"],
            interactions=[
                ("User", "Frontend", "Click button"),
                ("Frontend", "Backend", "API call"),
                ("Backend", "Frontend", "Response"),
            ],
        )
        
        assert "sequenceDiagram" in diagram
        assert "participant User" in diagram
        assert "participant Frontend" in diagram
        assert "participant Backend" in diagram
        assert "User->> Frontend: Click button" in diagram
        assert "Frontend->> Backend: API call" in diagram
        assert "Backend->> Frontend: Response" in diagram
    
    def test_creates_class_diagram(self, generator):
        """Test class diagram creation."""
        diagram = generator.create_mermaid_class(
            classes={
                "User": {
                    "attributes": ["id", "name", "email"],
                    "methods": ["login", "logout"],
                },
                "Post": {
                    "attributes": ["id", "title", "content"],
                    "methods": ["publish", "archive"],
                },
            },
            relationships=[
                ("User", "Post", "has"),
            ],
        )
        
        assert "classDiagram" in diagram
        assert "class User {" in diagram
        assert "+id" in diagram
        assert "+name" in diagram
        assert "+login()" in diagram
        assert "class Post {" in diagram
        assert "+title" in diagram
        assert "+publish()" in diagram
        assert "User *-- Post" in diagram
    
    def test_class_diagram_relationships(self, generator):
        """Test different class diagram relationship types."""
        diagram = generator.create_mermaid_class(
            classes={
                "Animal": {"attributes": ["name"]},
                "Dog": {"attributes": ["breed"]},
                "IFlyable": {"methods": ["fly"]},
                "Bird": {"attributes": ["wingspan"]},
            },
            relationships=[
                ("Animal", "Dog", "inherits"),
                ("IFlyable", "Bird", "implements"),
                ("Animal", "Bird", "uses"),
            ],
        )
        
        assert "Animal <|-- Dog" in diagram  # Inheritance
        assert "IFlyable <|.. Bird" in diagram  # Implementation
        assert "Animal --> Bird" in diagram  # Usage


class TestPlanSaving:
    """Test plan saving functionality."""
    
    def test_saves_plan_to_default_directory(self, generator):
        """Test saving plan to default .quirkllm/plans/ directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / ".quirkllm" / "plans"
            
            plan = "# Test Plan\n\nThis is a test."
            filepath = generator.save_plan(plan, "test-plan", output_dir=output_dir)
            
            assert filepath.exists()
            assert filepath.name == "test-plan.md"
            assert filepath.parent == output_dir
            assert filepath.read_text(encoding="utf-8") == plan
    
    def test_saves_plan_with_sanitized_filename(self, generator):
        """Test filename sanitization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            plan = "# Test"
            filepath = generator.save_plan(plan, "test/plan*with:invalid<chars>", output_dir=output_dir)
            
            # Should remove invalid characters
            assert filepath.exists()
            assert "/" not in filepath.name
            assert "*" not in filepath.name
            assert "<" not in filepath.name
    
    def test_creates_output_directory_if_missing(self, generator):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "new" / "nested" / "path"
            
            plan = "# Test"
            filepath = generator.save_plan(plan, "test", output_dir=output_dir)
            
            assert output_dir.exists()
            assert filepath.exists()
    
    def test_adds_md_extension_if_missing(self, generator):
        """Test that .md extension is added if missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            plan = "# Test"
            filepath = generator.save_plan(plan, "test-plan-no-ext", output_dir=output_dir)
            
            assert filepath.name == "test-plan-no-ext.md"
