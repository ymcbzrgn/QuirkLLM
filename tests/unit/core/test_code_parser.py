"""Unit tests for CodeBlockParser.

Tests cover:
- Basic code block parsing
- Multiple code blocks
- Language detection
- Filename extraction (inline and from comments)
- Content analysis for filename suggestion
- Non-code text extraction
"""

import pytest

from quirkllm.core.code_parser import CodeBlock, CodeBlockParser


class TestCodeBlockParser:
    """Tests for CodeBlockParser class."""

    @pytest.fixture
    def parser(self) -> CodeBlockParser:
        """Create a parser instance."""
        return CodeBlockParser()

    def test_parse_simple_code_block(self, parser: CodeBlockParser) -> None:
        """Test parsing a simple code block with language."""
        text = """Here's some Python code:

```python
def hello():
    print("Hello, World!")
```

That's it!"""

        blocks = parser.parse(text)

        assert len(blocks) == 1
        assert blocks[0].language == "python"
        assert "def hello():" in blocks[0].code
        assert blocks[0].filename is None

    def test_parse_code_block_without_language(self, parser: CodeBlockParser) -> None:
        """Test parsing code block without language specification."""
        text = """Some code:

```
echo "hello"
```"""

        blocks = parser.parse(text)

        assert len(blocks) == 1
        assert blocks[0].language == ""
        assert 'echo "hello"' in blocks[0].code

    def test_parse_multiple_code_blocks(self, parser: CodeBlockParser) -> None:
        """Test parsing multiple code blocks."""
        text = """First block:

```python
x = 1
```

Second block:

```javascript
const y = 2;
```

Third block:

```rust
let z = 3;
```"""

        blocks = parser.parse(text)

        assert len(blocks) == 3
        assert blocks[0].language == "python"
        assert blocks[1].language == "javascript"
        assert blocks[2].language == "rust"

    def test_parse_inline_filename(self, parser: CodeBlockParser) -> None:
        """Test parsing code block with inline filename."""
        text = """```python:utils.py
def helper():
    pass
```"""

        blocks = parser.parse(text)

        assert len(blocks) == 1
        assert blocks[0].filename == "utils.py"
        assert blocks[0].language == "python"

    def test_parse_filename_from_comment_python(self, parser: CodeBlockParser) -> None:
        """Test extracting filename from Python comment."""
        text = """```python
# filename: config.py
DEBUG = True
```"""

        blocks = parser.parse(text)

        assert len(blocks) == 1
        assert blocks[0].filename == "config.py"

    def test_parse_filename_from_comment_javascript(
        self, parser: CodeBlockParser
    ) -> None:
        """Test extracting filename from JavaScript comment."""
        text = """```javascript
// filename: app.js
console.log("Hello");
```"""

        blocks = parser.parse(text)

        assert len(blocks) == 1
        assert blocks[0].filename == "app.js"

    def test_parse_filename_from_comment_html(self, parser: CodeBlockParser) -> None:
        """Test extracting filename from HTML comment."""
        text = """```html
<!-- filename: index.html -->
<div>Hello</div>
```"""

        blocks = parser.parse(text)

        assert len(blocks) == 1
        assert blocks[0].filename == "index.html"

    def test_parse_preserves_indentation(self, parser: CodeBlockParser) -> None:
        """Test that code indentation is preserved."""
        text = """```python
class Foo:
    def bar(self):
        if True:
            return 42
```"""

        blocks = parser.parse(text)

        assert len(blocks) == 1
        assert "    def bar(self):" in blocks[0].code
        assert "        if True:" in blocks[0].code
        assert "            return 42" in blocks[0].code

    def test_parse_line_numbers(self, parser: CodeBlockParser) -> None:
        """Test that line numbers are tracked correctly."""
        text = """Line 1
Line 2
```python
code here
```
Line 6"""

        blocks = parser.parse(text)

        assert len(blocks) == 1
        assert blocks[0].start_line == 3
        assert blocks[0].end_line == 5

    def test_parse_empty_response(self, parser: CodeBlockParser) -> None:
        """Test parsing response with no code blocks."""
        text = "Just some text without code."

        blocks = parser.parse(text)

        assert len(blocks) == 0

    def test_parse_nested_backticks(self, parser: CodeBlockParser) -> None:
        """Test parsing code that contains backticks in content."""
        text = '''```markdown
Use `code` for inline code.
```'''

        blocks = parser.parse(text)

        assert len(blocks) == 1
        assert "`code`" in blocks[0].code


class TestFilenamesuggestion:
    """Tests for filename suggestion logic."""

    @pytest.fixture
    def parser(self) -> CodeBlockParser:
        """Create a parser instance."""
        return CodeBlockParser()

    def test_suggest_uses_existing_filename(self, parser: CodeBlockParser) -> None:
        """Test that existing filename is returned."""
        block = CodeBlock(
            language="python",
            code="x = 1",
            filename="custom.py",
            start_line=1,
            end_line=3,
        )

        filename = parser.suggest_filename(block)

        assert filename == "custom.py"

    def test_suggest_python_class(self, parser: CodeBlockParser) -> None:
        """Test filename suggestion for Python class."""
        block = CodeBlock(
            language="python",
            code="""class UserManager:
    def __init__(self):
        pass""",
            filename=None,
            start_line=1,
            end_line=4,
        )

        filename = parser.suggest_filename(block)

        assert filename == "user_manager.py"

    def test_suggest_python_main(self, parser: CodeBlockParser) -> None:
        """Test filename suggestion for Python main block."""
        block = CodeBlock(
            language="python",
            code="""def main():
    pass

if __name__ == "__main__":
    main()""",
            filename=None,
            start_line=1,
            end_line=5,
        )

        filename = parser.suggest_filename(block)

        assert filename == "main.py"

    def test_suggest_python_function(self, parser: CodeBlockParser) -> None:
        """Test filename suggestion for Python function."""
        block = CodeBlock(
            language="python",
            code="""def calculate_sum(numbers):
    return sum(numbers)""",
            filename=None,
            start_line=1,
            end_line=2,
        )

        filename = parser.suggest_filename(block)

        assert filename == "calculate_sum.py"

    def test_suggest_react_component(self, parser: CodeBlockParser) -> None:
        """Test filename suggestion for React component."""
        block = CodeBlock(
            language="jsx",
            code="""function Counter() {
    const [count, setCount] = useState(0);
    return <div>{count}</div>;
}""",
            filename=None,
            start_line=1,
            end_line=4,
        )

        filename = parser.suggest_filename(block)

        assert filename == "Counter.jsx"

    def test_suggest_typescript_component(self, parser: CodeBlockParser) -> None:
        """Test filename suggestion for TypeScript component."""
        block = CodeBlock(
            language="tsx",
            code="""const Header = () => {
    return <header>Title</header>;
};""",
            filename=None,
            start_line=1,
            end_line=3,
        )

        filename = parser.suggest_filename(block)

        assert filename == "Header.tsx"

    def test_suggest_javascript_export(self, parser: CodeBlockParser) -> None:
        """Test filename suggestion for JS default export."""
        block = CodeBlock(
            language="javascript",
            code="""export default function utils() {
    return {};
}""",
            filename=None,
            start_line=1,
            end_line=3,
        )

        filename = parser.suggest_filename(block)

        assert filename == "utils.js"

    def test_suggest_rust_main(self, parser: CodeBlockParser) -> None:
        """Test filename suggestion for Rust main."""
        block = CodeBlock(
            language="rust",
            code="""fn main() {
    println!("Hello");
}""",
            filename=None,
            start_line=1,
            end_line=3,
        )

        filename = parser.suggest_filename(block)

        assert filename == "main.rs"

    def test_suggest_go_main(self, parser: CodeBlockParser) -> None:
        """Test filename suggestion for Go main."""
        block = CodeBlock(
            language="go",
            code="""package main

func main() {
    fmt.Println("Hello")
}""",
            filename=None,
            start_line=1,
            end_line=5,
        )

        filename = parser.suggest_filename(block)

        assert filename == "main.go"

    def test_suggest_java_class(self, parser: CodeBlockParser) -> None:
        """Test filename suggestion for Java class."""
        block = CodeBlock(
            language="java",
            code="""public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello");
    }
}""",
            filename=None,
            start_line=1,
            end_line=5,
        )

        filename = parser.suggest_filename(block)

        assert filename == "HelloWorld.java"

    def test_suggest_unknown_language_fallback(self, parser: CodeBlockParser) -> None:
        """Test fallback for unknown language."""
        block = CodeBlock(
            language="unknown",
            code="some code",
            filename=None,
            start_line=1,
            end_line=1,
        )

        filename = parser.suggest_filename(block)

        assert filename == "code.txt"

    def test_suggest_dockerfile(self, parser: CodeBlockParser) -> None:
        """Test filename suggestion for Dockerfile."""
        block = CodeBlock(
            language="dockerfile",
            code="""FROM python:3.11
COPY . /app
CMD ["python", "main.py"]""",
            filename=None,
            start_line=1,
            end_line=3,
        )

        filename = parser.suggest_filename(block)

        assert filename == "Dockerfile"

    def test_suggest_makefile(self, parser: CodeBlockParser) -> None:
        """Test filename suggestion for Makefile."""
        block = CodeBlock(
            language="makefile",
            code="""all:
	echo "Building..."

clean:
	rm -rf build/""",
            filename=None,
            start_line=1,
            end_line=5,
        )

        filename = parser.suggest_filename(block)

        assert filename == "Makefile"


class TestNonCodeTextExtraction:
    """Tests for non-code text extraction."""

    @pytest.fixture
    def parser(self) -> CodeBlockParser:
        """Create a parser instance."""
        return CodeBlockParser()

    def test_get_text_before_and_after(self, parser: CodeBlockParser) -> None:
        """Test extracting text before and after code block."""
        text = """Here's the explanation.

```python
code
```

And here's more text."""

        segments = parser.get_non_code_text(text)

        assert len(segments) == 2
        assert "explanation" in segments[0]
        assert "more text" in segments[1]

    def test_get_text_between_blocks(self, parser: CodeBlockParser) -> None:
        """Test extracting text between multiple code blocks."""
        text = """First:

```python
a = 1
```

Second:

```python
b = 2
```

Third."""

        segments = parser.get_non_code_text(text)

        assert len(segments) == 3
        assert "First" in segments[0]
        assert "Second" in segments[1]
        assert "Third" in segments[2]

    def test_get_text_only_code(self, parser: CodeBlockParser) -> None:
        """Test response with only code (no text)."""
        text = """```python
code
```"""

        segments = parser.get_non_code_text(text)

        assert len(segments) == 0

    def test_get_text_no_code(self, parser: CodeBlockParser) -> None:
        """Test response with no code blocks."""
        text = "Just plain text here."

        segments = parser.get_non_code_text(text)

        assert len(segments) == 1
        assert segments[0] == "Just plain text here."


class TestLanguageExtensions:
    """Tests for language to extension mapping."""

    @pytest.fixture
    def parser(self) -> CodeBlockParser:
        """Create a parser instance."""
        return CodeBlockParser()

    def test_common_languages(self, parser: CodeBlockParser) -> None:
        """Test extension mapping for common languages."""
        test_cases = [
            ("python", ".py"),
            ("py", ".py"),
            ("javascript", ".js"),
            ("js", ".js"),
            ("typescript", ".ts"),
            ("ts", ".ts"),
            ("jsx", ".jsx"),
            ("tsx", ".tsx"),
            ("rust", ".rs"),
            ("go", ".go"),
            ("java", ".java"),
            ("c", ".c"),
            ("cpp", ".cpp"),
            ("ruby", ".rb"),
            ("php", ".php"),
            ("swift", ".swift"),
            ("kotlin", ".kt"),
            ("html", ".html"),
            ("css", ".css"),
            ("json", ".json"),
            ("yaml", ".yaml"),
            ("sql", ".sql"),
            ("bash", ".sh"),
            ("shell", ".sh"),
        ]

        for lang, expected_ext in test_cases:
            assert parser.LANGUAGE_EXTENSIONS.get(lang) == expected_ext


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.fixture
    def parser(self) -> CodeBlockParser:
        """Create a parser instance."""
        return CodeBlockParser()

    def test_empty_code_block(self, parser: CodeBlockParser) -> None:
        """Test parsing empty code block."""
        text = """```python
```"""

        blocks = parser.parse(text)

        assert len(blocks) == 1
        assert blocks[0].code == ""

    def test_whitespace_only_code(self, parser: CodeBlockParser) -> None:
        """Test parsing whitespace-only code block."""
        text = """```python

```"""

        blocks = parser.parse(text)

        assert len(blocks) == 1

    def test_language_case_insensitive(self, parser: CodeBlockParser) -> None:
        """Test that language parsing is case insensitive."""
        text = """```PYTHON
code
```"""

        blocks = parser.parse(text)

        assert len(blocks) == 1
        assert blocks[0].language == "python"

    def test_multiline_code_block(self, parser: CodeBlockParser) -> None:
        """Test parsing multi-line code block."""
        text = """```python
def foo():
    pass

def bar():
    pass

class Baz:
    def qux(self):
        return 42
```"""

        blocks = parser.parse(text)

        assert len(blocks) == 1
        assert "def foo():" in blocks[0].code
        assert "def bar():" in blocks[0].code
        assert "class Baz:" in blocks[0].code

    def test_code_with_special_characters(self, parser: CodeBlockParser) -> None:
        """Test parsing code with special characters."""
        text = """```python
regex = r"^\\d+\\.\\d+$"
string = "Hello, 世界! 🌍"
```"""

        blocks = parser.parse(text)

        assert len(blocks) == 1
        assert 'r"^\\d+\\.\\d+$"' in blocks[0].code
        assert "世界" in blocks[0].code
        assert "🌍" in blocks[0].code
