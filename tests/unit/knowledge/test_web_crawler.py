"""
Tests for WebCrawler class.

Test Categories:
1. Initialization Tests (3)
2. Single Page Fetch Tests (3)
3. HTML to Markdown Tests (4)
4. Link Extraction Tests (3)
5. Recursive Crawl Tests (2)
6. Rate Limiting Tests (2)
7. Adaptive Depth Tests (3)

Total: 20 tests
"""

import time
from unittest.mock import Mock, patch, MagicMock
import pytest
import requests

from quirkllm.knowledge.web_crawler import WebCrawler


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def crawler():
    """Create a basic WebCrawler instance."""
    return WebCrawler(
        base_url="https://example.com/docs/",
        max_depth=2,
        max_pages=10,
        rate_limit=0.1,  # Fast for testing
    )


@pytest.fixture
def sample_html():
    """Sample HTML for testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page Title</title>
    </head>
    <body>
        <nav>Navigation to skip</nav>
        <main>
            <h1>Main Heading</h1>
            <p>This is a paragraph with <a href="/docs/page2">internal link</a>.</p>
            <pre><code>def hello():
    print("Hello, World!")
</code></pre>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
            <a href="https://external.com">External Link</a>
            <a href="/docs/page3">Another internal</a>
        </main>
        <footer>Footer to skip</footer>
    </body>
    </html>
    """


@pytest.fixture
def html_with_table():
    """HTML with a table for testing."""
    return """
    <html>
    <head><title>Table Test</title></head>
    <body>
        <table>
            <tr><th>Name</th><th>Value</th></tr>
            <tr><td>Foo</td><td>Bar</td></tr>
        </table>
    </body>
    </html>
    """


# =============================================================================
# 1. Initialization Tests (3)
# =============================================================================


class TestCrawlerInitialization:
    """Tests for WebCrawler initialization."""

    def test_crawler_init_default_values(self):
        """Test crawler initializes with default values."""
        crawler = WebCrawler("https://example.com")

        assert crawler.base_url == "https://example.com"
        assert crawler.max_depth == 2  # Minimum enforced
        assert crawler.max_pages == 100
        assert crawler.rate_limit == 1.0
        assert crawler.adaptive_depth is True
        assert crawler.base_domain == "example.com"
        assert len(crawler.visited) == 0
        assert len(crawler.results) == 0

    def test_crawler_init_custom_values(self):
        """Test crawler initializes with custom values."""
        crawler = WebCrawler(
            base_url="https://docs.python.org/",
            max_depth=5,
            max_pages=50,
            rate_limit=0.5,
            user_agent="CustomAgent/1.0",
            adaptive_depth=False,
        )

        assert crawler.base_url == "https://docs.python.org"  # Trailing slash removed
        assert crawler.max_depth == 5
        assert crawler.max_pages == 50
        assert crawler.rate_limit == 0.5
        assert crawler.user_agent == "CustomAgent/1.0"
        assert crawler.adaptive_depth is False

    def test_crawler_enforces_minimum_depth(self):
        """Test crawler enforces minimum depth of 2."""
        crawler1 = WebCrawler("https://example.com", max_depth=0)
        crawler2 = WebCrawler("https://example.com", max_depth=1)
        crawler3 = WebCrawler("https://example.com", max_depth=2)

        assert crawler1.max_depth == 2  # Enforced minimum
        assert crawler2.max_depth == 2  # Enforced minimum
        assert crawler3.max_depth == 2  # Already meets minimum

    def test_crawler_session_setup(self):
        """Test crawler sets up session with correct headers."""
        crawler = WebCrawler("https://example.com")

        assert "User-Agent" in crawler.session.headers
        assert "QuirkLLM" in crawler.session.headers["User-Agent"]
        assert "Accept" in crawler.session.headers


# =============================================================================
# 2. Single Page Fetch Tests (3)
# =============================================================================


class TestFetchPage:
    """Tests for fetch_page method."""

    def test_fetch_page_success(self, crawler):
        """Test successful page fetch."""
        with patch.object(crawler.session, "get") as mock_get:
            mock_response = Mock()
            mock_response.text = "<html><body>Test content</body></html>"
            mock_response.headers = {"Content-Type": "text/html"}
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            result = crawler.fetch_page("https://example.com/docs/page")

            assert result == "<html><body>Test content</body></html>"
            mock_get.assert_called_once()

    def test_fetch_page_404_error(self, crawler):
        """Test fetch returns None on 404."""
        with patch.object(crawler.session, "get") as mock_get:
            mock_get.side_effect = requests.HTTPError("404 Not Found")

            result = crawler.fetch_page("https://example.com/nonexistent")

            assert result is None

    def test_fetch_page_timeout_error(self, crawler):
        """Test fetch returns None on timeout."""
        with patch.object(crawler.session, "get") as mock_get:
            mock_get.side_effect = requests.Timeout("Connection timed out")

            result = crawler.fetch_page("https://example.com/slow")

            assert result is None

    def test_fetch_page_non_html_content(self, crawler):
        """Test fetch returns None for non-HTML content."""
        with patch.object(crawler.session, "get") as mock_get:
            mock_response = Mock()
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            result = crawler.fetch_page("https://example.com/api/data.json")

            assert result is None


# =============================================================================
# 3. HTML to Markdown Tests (4)
# =============================================================================


class TestHtmlToMarkdown:
    """Tests for html_to_markdown method."""

    def test_html_to_markdown_headings(self, crawler):
        """Test markdown preserves headings."""
        html = "<html><body><h1>Heading 1</h1><h2>Heading 2</h2></body></html>"

        markdown = crawler.html_to_markdown(html)

        assert "# Heading 1" in markdown
        assert "## Heading 2" in markdown

    def test_html_to_markdown_code_blocks(self, crawler, sample_html):
        """Test markdown preserves code blocks."""
        markdown = crawler.html_to_markdown(sample_html)

        # Code should be preserved
        assert "def hello():" in markdown
        assert 'print("Hello, World!")' in markdown

    def test_html_to_markdown_lists(self, crawler, sample_html):
        """Test markdown preserves lists."""
        markdown = crawler.html_to_markdown(sample_html)

        assert "Item 1" in markdown
        assert "Item 2" in markdown

    def test_html_to_markdown_removes_nav_footer(self, crawler, sample_html):
        """Test markdown removes nav and footer."""
        markdown = crawler.html_to_markdown(sample_html)

        assert "Navigation to skip" not in markdown
        assert "Footer to skip" not in markdown

    def test_html_to_markdown_tables(self, crawler, html_with_table):
        """Test markdown handles tables."""
        markdown = crawler.html_to_markdown(html_with_table)

        # Table content should be present (format may vary)
        assert "Name" in markdown
        assert "Value" in markdown
        assert "Foo" in markdown
        assert "Bar" in markdown


# =============================================================================
# 4. Link Extraction Tests (3)
# =============================================================================


class TestLinkExtraction:
    """Tests for extract_links method."""

    def test_extract_links_same_domain(self, crawler, sample_html):
        """Test extracts same-domain links."""
        links = crawler.extract_links(sample_html, "https://example.com/docs/")

        # Should have internal links
        assert any("/docs/page2" in link for link in links)
        assert any("/docs/page3" in link for link in links)

    def test_extract_links_external_filtered(self, crawler, sample_html):
        """Test filters out external domain links."""
        links = crawler.extract_links(sample_html, "https://example.com/docs/")

        # External links should be filtered
        assert not any("external.com" in link for link in links)

    def test_extract_links_relative_resolved(self, crawler):
        """Test relative links are resolved correctly."""
        html = """
        <a href="page2">Relative</a>
        <a href="../other">Parent relative</a>
        <a href="/absolute">Absolute</a>
        """

        links = crawler.extract_links(html, "https://example.com/docs/guide/")

        # All should be resolved to absolute URLs
        assert all(link.startswith("https://example.com") for link in links)

    def test_extract_links_skips_special_protocols(self, crawler):
        """Test skips mailto:, javascript:, etc."""
        html = """
        <a href="mailto:test@example.com">Email</a>
        <a href="javascript:void(0)">JS</a>
        <a href="tel:+1234567890">Phone</a>
        <a href="/valid">Valid</a>
        """

        links = crawler.extract_links(html, "https://example.com/")

        assert len(links) == 1
        assert "/valid" in links[0]


# =============================================================================
# 5. Recursive Crawl Tests (2)
# =============================================================================


class TestRecursiveCrawl:
    """Tests for crawl method with recursive behavior."""

    def test_crawl_single_page_depth_0(self):
        """Test crawl respects max_pages=1."""
        crawler = WebCrawler("https://example.com", max_pages=1)

        with patch.object(crawler, "fetch_page") as mock_fetch:
            mock_fetch.return_value = "<html><body><h1>Test</h1></body></html>"

            results = crawler.crawl(show_progress=False)

            assert len(results) == 1
            assert results[0]["url"] == "https://example.com"

    def test_crawl_recursive_depth_2(self):
        """Test crawl follows links to depth 2."""
        crawler = WebCrawler(
            "https://example.com/docs/",
            max_depth=2,
            max_pages=10,
            rate_limit=0,  # No delay for testing
        )

        # Mock responses for different pages
        def mock_fetch(url, timeout=10):
            if url == "https://example.com/docs":
                return """
                <html><head><title>Docs Index</title></head>
                <body>
                    <a href="/docs/page1">Page 1</a>
                    <a href="/docs/page2">Page 2</a>
                </body></html>
                """
            elif "page1" in url:
                return """
                <html><head><title>Page 1</title></head>
                <body><h1>Page 1 Content</h1></body></html>
                """
            elif "page2" in url:
                return """
                <html><head><title>Page 2</title></head>
                <body><h1>Page 2 Content</h1></body></html>
                """
            return None

        with patch.object(crawler, "fetch_page", side_effect=mock_fetch):
            results = crawler.crawl(show_progress=False)

            # Should have multiple pages
            assert len(results) >= 1
            urls = [r["url"] for r in results]
            assert "https://example.com/docs" in urls


# =============================================================================
# 6. Rate Limiting Tests (2)
# =============================================================================


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    def test_crawl_rate_limiting(self):
        """Test crawl respects rate limit."""
        crawler = WebCrawler(
            "https://example.com",
            rate_limit=0.1,  # 100ms
            max_pages=2,
        )

        call_times = []

        def mock_fetch(url, timeout=10):
            call_times.append(time.time())
            return f"<html><head><title>{url}</title></head><body><a href='/page{len(call_times)}'>Link</a></body></html>"

        with patch.object(crawler, "fetch_page", side_effect=mock_fetch):
            crawler.crawl(show_progress=False)

            # Check time between calls (should be at least rate_limit apart)
            if len(call_times) >= 2:
                time_diff = call_times[1] - call_times[0]
                assert time_diff >= 0.05  # Allow some tolerance

    def test_crawl_max_pages_limit(self):
        """Test crawl respects max_pages limit."""
        crawler = WebCrawler(
            "https://example.com",
            max_pages=3,
            rate_limit=0,
        )

        # Generate infinite pages
        def mock_fetch(url, timeout=10):
            page_num = len(crawler.results)
            return f"""
            <html><head><title>Page {page_num}</title></head>
            <body><a href='/page{page_num + 1}'>Next</a></body></html>
            """

        with patch.object(crawler, "fetch_page", side_effect=mock_fetch):
            results = crawler.crawl(show_progress=False)

            assert len(results) <= 3


# =============================================================================
# 7. Adaptive Depth Tests (3)
# =============================================================================


class TestAdaptiveDepth:
    """Tests for adaptive depth feature."""

    def test_adaptive_depth_doc_pattern(self):
        """Test doc patterns get +1 depth bonus."""
        crawler = WebCrawler(
            "https://example.com",
            max_depth=2,
            adaptive_depth=True,
        )

        # Regular URL
        regular_depth = crawler._calculate_effective_depth("https://example.com/page")
        assert regular_depth == 2

        # Doc URL
        doc_depth = crawler._calculate_effective_depth("https://example.com/docs/guide/")
        assert doc_depth == 3  # +1 bonus

    def test_adaptive_depth_api_reference_bonus(self):
        """Test API/reference patterns get +2 depth bonus."""
        crawler = WebCrawler(
            "https://example.com",
            max_depth=2,
            adaptive_depth=True,
        )

        # API URL
        api_depth = crawler._calculate_effective_depth("https://example.com/api/v1/")
        assert api_depth == 4  # +2 bonus

        # Reference URL
        ref_depth = crawler._calculate_effective_depth("https://example.com/reference/classes/")
        assert ref_depth == 4  # +2 bonus

    def test_adaptive_depth_max_cap_5(self):
        """Test adaptive depth is capped at 5."""
        crawler = WebCrawler(
            "https://example.com",
            max_depth=4,  # High base
            adaptive_depth=True,
        )

        # Should be capped at 5 even with +2 bonus
        depth = crawler._calculate_effective_depth("https://example.com/api/reference/")
        assert depth == 5  # Capped

    def test_adaptive_depth_disabled(self):
        """Test adaptive depth can be disabled."""
        crawler = WebCrawler(
            "https://example.com",
            max_depth=2,
            adaptive_depth=False,
        )

        # Should NOT get bonus when disabled
        depth = crawler._calculate_effective_depth("https://example.com/api/reference/")
        assert depth == 2  # No bonus


# =============================================================================
# Additional Tests
# =============================================================================


class TestTitleExtraction:
    """Tests for title extraction."""

    def test_extract_title_from_title_tag(self, crawler):
        """Test title extracted from <title> tag."""
        html = "<html><head><title>Page Title</title></head><body></body></html>"

        title = crawler.extract_title(html)

        assert title == "Page Title"

    def test_extract_title_from_h1(self, crawler):
        """Test title extracted from <h1> when no <title>."""
        html = "<html><body><h1>Heading Title</h1></body></html>"

        title = crawler.extract_title(html)

        assert title == "Heading Title"

    def test_extract_title_fallback(self, crawler):
        """Test title fallback to 'Untitled'."""
        html = "<html><body><p>No title here</p></body></html>"

        title = crawler.extract_title(html)

        assert title == "Untitled"


class TestUrlHelpers:
    """Tests for URL helper methods."""

    def test_is_same_domain_true(self, crawler):
        """Test same domain detection."""
        assert crawler._is_same_domain("https://example.com/page")
        assert crawler._is_same_domain("https://example.com/docs/guide")

    def test_is_same_domain_false(self, crawler):
        """Test different domain detection."""
        assert not crawler._is_same_domain("https://other.com/page")
        assert not crawler._is_same_domain("https://subdomain.example.com/page")

    def test_normalize_url(self, crawler):
        """Test URL normalization."""
        # Remove fragment
        assert crawler._normalize_url("https://example.com/page#section") == "https://example.com/page"

        # Remove trailing slash
        assert crawler._normalize_url("https://example.com/page/") == "https://example.com/page"


class TestCrawlerStats:
    """Tests for get_stats method."""

    def test_get_stats_empty(self, crawler):
        """Test stats when no pages crawled."""
        stats = crawler.get_stats()

        assert stats["pages_crawled"] == 0
        assert stats["total_content_length"] == 0
        assert stats["avg_depth"] == 0

    def test_get_stats_with_results(self):
        """Test stats after crawling."""
        crawler = WebCrawler("https://example.com", max_pages=2, rate_limit=0)

        # Manually add results for testing
        crawler.results = [
            {"url": "https://example.com", "title": "Test", "content": "A" * 100,
             "metadata": {"depth": 0, "effective_depth": 2, "domain": "example.com", "content_length": 100}},
            {"url": "https://example.com/page", "title": "Test 2", "content": "B" * 200,
             "metadata": {"depth": 1, "effective_depth": 2, "domain": "example.com", "content_length": 200}},
        ]

        stats = crawler.get_stats()

        assert stats["pages_crawled"] == 2
        assert stats["total_content_length"] == 300
        assert stats["avg_depth"] == 0.5
