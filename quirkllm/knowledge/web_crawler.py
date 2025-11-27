"""
Web Crawler - Crawl and convert web documentation for RAG ingestion.

Features:
- Single page or recursive crawling
- Adaptive depth based on URL patterns
- Rate limiting (respect servers)
- HTML to Markdown conversion
- Code block preservation
- Same-domain filtering

Example:
    >>> crawler = WebCrawler("https://docs.python.org/3/tutorial/")
    >>> pages = crawler.crawl()
    >>> for page in pages:
    ...     print(f"{page['title']}: {len(page['content'])} chars")
"""

from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, urlparse
from collections import deque
import time
import requests
from bs4 import BeautifulSoup
import html2text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console


class WebCrawler:
    """
    Crawls web pages and converts to markdown for RAG ingestion.

    The crawler supports adaptive depth that increases for documentation
    paths like /docs/, /api/, /tutorial/ etc.

    Attributes:
        base_url: Starting URL to crawl
        max_depth: Maximum link depth to follow (minimum 2)
        max_pages: Maximum number of pages to crawl
        rate_limit: Seconds between requests
        adaptive_depth: Enable smart depth based on URL patterns
        visited: Set of already visited URLs
        results: List of crawled page results
    """

    DEFAULT_USER_AGENT = "QuirkLLM/1.0 (+https://github.com/yamac/quirkllm)"

    # Patterns that suggest documentation content worth crawling deeper
    DOC_PATTERNS = [
        "/docs/",
        "/documentation/",
        "/guide/",
        "/api/",
        "/tutorial/",
        "/reference/",
        "/manual/",
        "/learn/",
        "/getting-started/",
        "/quickstart/",
    ]

    def __init__(
        self,
        base_url: str,
        max_depth: int = 2,
        max_pages: int = 100,
        rate_limit: float = 1.0,
        user_agent: Optional[str] = None,
        adaptive_depth: bool = True,
    ) -> None:
        """
        Initialize WebCrawler.

        Args:
            base_url: Starting URL to crawl
            max_depth: Maximum link depth (minimum enforced: 2)
            max_pages: Maximum pages to crawl (safety limit)
            rate_limit: Seconds between requests (respect servers)
            user_agent: Custom user agent string
            adaptive_depth: Enable smart depth for doc patterns
        """
        self.base_url = base_url.rstrip("/")
        self.max_depth = max(2, max_depth)  # Enforce minimum 2
        self.max_pages = max_pages
        self.rate_limit = rate_limit
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT
        self.adaptive_depth = adaptive_depth
        self.visited: set = set()
        self.results: List[Dict[str, Any]] = []

        # Parse base URL for domain matching
        parsed = urlparse(self.base_url)
        self.base_domain = parsed.netloc

        # HTML to Markdown converter setup
        self.converter = html2text.HTML2Text()
        self.converter.ignore_links = False
        self.converter.ignore_images = True
        self.converter.body_width = 0  # No line wrapping
        self.converter.protect_links = True
        self.converter.wrap_links = False
        self.converter.skip_internal_links = False

        # Session for connection pooling and headers
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })

        # Console for Rich output
        self.console = Console()

    def crawl(self, show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Crawl starting from base_url using BFS.

        Uses breadth-first search with adaptive depth limits.
        Respects rate limiting between requests.

        Args:
            show_progress: Show Rich progress bar

        Returns:
            List of dicts with keys: url, title, content, metadata
        """
        # Reset state
        self.visited.clear()
        self.results.clear()

        # BFS queue: (url, depth)
        queue: deque = deque([(self.base_url, 0)])
        self.visited.add(self._normalize_url(self.base_url))

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.completed}/{task.total}"),
                console=self.console,
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Crawling {self.base_domain}...",
                    total=self.max_pages,
                )
                self._crawl_loop(queue, progress, task)
        else:
            self._crawl_loop(queue, None, None)

        return self.results

    def _crawl_loop(
        self,
        queue: deque,
        progress: Optional[Progress],
        task: Optional[int],
    ) -> None:
        """
        Main crawl loop with BFS traversal.

        Args:
            queue: BFS queue of (url, depth) tuples
            progress: Rich progress bar (optional)
            task: Progress task ID (optional)
        """
        while queue and len(self.results) < self.max_pages:
            url, depth = queue.popleft()

            # Calculate effective depth limit for this URL
            effective_depth = self._calculate_effective_depth(url)

            # Skip if beyond effective depth
            if depth > effective_depth:
                continue

            # Fetch and process page
            html = self.fetch_page(url)
            if html is None:
                continue

            # Extract content
            title = self.extract_title(html)
            content = self.html_to_markdown(html)

            # Store result
            self.results.append({
                "url": url,
                "title": title,
                "content": content,
                "metadata": {
                    "depth": depth,
                    "effective_depth": effective_depth,
                    "domain": self.base_domain,
                    "content_length": len(content),
                },
            })

            # Update progress
            if progress and task is not None:
                progress.update(task, completed=len(self.results))
                progress.update(
                    task,
                    description=f"[cyan]Crawling: {title[:40]}...",
                )

            # Extract and queue links (only if not at max depth)
            if depth < effective_depth:
                links = self.extract_links(html, url)
                for link in links:
                    normalized = self._normalize_url(link)
                    if normalized not in self.visited:
                        self.visited.add(normalized)
                        queue.append((link, depth + 1))

            # Rate limiting
            if queue:
                time.sleep(self.rate_limit)

    def fetch_page(self, url: str, timeout: int = 10) -> Optional[str]:
        """
        Fetch single page HTML content.

        Args:
            url: URL to fetch
            timeout: Request timeout in seconds

        Returns:
            HTML content string or None on error
        """
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return None

            return response.text

        except requests.RequestException:
            # Silently skip failed requests
            return None

    def extract_links(self, html: str, current_url: str) -> List[str]:
        """
        Extract same-domain links from HTML.

        Filters out:
        - External domain links
        - Anchor-only links (#)
        - Non-HTTP links (mailto:, javascript:, etc.)

        Args:
            html: HTML content
            current_url: Current page URL for resolving relative links

        Returns:
            List of absolute URLs within same domain
        """
        soup = BeautifulSoup(html, "html.parser")
        links: List[str] = []

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]

            # Skip empty, anchor-only, or special protocol links
            if not href or href.startswith("#"):
                continue
            if href.startswith(("mailto:", "javascript:", "tel:", "ftp:")):
                continue

            # Resolve relative URLs
            absolute_url = urljoin(current_url, href)

            # Filter to same domain only
            if self._is_same_domain(absolute_url):
                # Normalize and add
                normalized = self._normalize_url(absolute_url)
                if normalized not in links:
                    links.append(normalized)

        return links

    def html_to_markdown(self, html: str) -> str:
        """
        Convert HTML to clean Markdown.

        Preserves:
        - Headings (h1-h6)
        - Code blocks (pre, code)
        - Lists (ul, ol)
        - Tables (basic)
        - Links (a)

        Removes:
        - Navigation elements (nav)
        - Header/footer
        - Scripts and styles

        Args:
            html: HTML content

        Returns:
            Markdown string
        """
        soup = BeautifulSoup(html, "html.parser")

        # Remove unwanted elements
        for tag in soup.find_all(["nav", "header", "footer", "script", "style", "aside"]):
            tag.decompose()

        # Find main content (prefer article, main, or content divs)
        main_content = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", {"class": ["content", "main-content", "documentation"]})
            or soup.find("div", {"id": ["content", "main-content", "documentation"]})
            or soup.body
            or soup
        )

        # Convert to markdown
        markdown = self.converter.handle(str(main_content))

        # Clean up excessive whitespace
        lines = markdown.split("\n")
        cleaned_lines = []
        prev_empty = False

        for line in lines:
            stripped = line.strip()
            is_empty = not stripped

            # Skip multiple consecutive empty lines
            if is_empty and prev_empty:
                continue

            cleaned_lines.append(stripped if is_empty else line)
            prev_empty = is_empty

        return "\n".join(cleaned_lines).strip()

    def extract_title(self, html: str) -> str:
        """
        Extract page title from HTML.

        Tries in order:
        1. <title> tag
        2. <h1> tag
        3. og:title meta tag
        4. Fallback to "Untitled"

        Args:
            html: HTML content

        Returns:
            Page title string
        """
        soup = BeautifulSoup(html, "html.parser")

        # Try <title> tag
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return title_tag.string.strip()

        # Try <h1> tag
        h1_tag = soup.find("h1")
        if h1_tag:
            return h1_tag.get_text(strip=True)

        # Try og:title meta
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return og_title["content"].strip()

        return "Untitled"

    def _is_same_domain(self, url: str) -> bool:
        """
        Check if URL is same domain as base_url.

        Args:
            url: URL to check

        Returns:
            True if same domain
        """
        try:
            parsed = urlparse(url)
            return parsed.netloc == self.base_domain
        except Exception:
            return False

    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL for deduplication.

        - Removes fragments (#section)
        - Removes trailing slashes
        - Lowercases the path

        Args:
            url: URL to normalize

        Returns:
            Normalized URL string
        """
        try:
            parsed = urlparse(url)
            # Reconstruct without fragment
            normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            # Remove trailing slash
            normalized = normalized.rstrip("/")
            return normalized
        except Exception:
            return url

    def _calculate_effective_depth(self, url: str) -> int:
        """
        Calculate effective depth limit based on URL patterns.

        Adaptive logic:
        - API/Reference pages: +2 depth bonus
        - Documentation pages: +1 depth bonus
        - Max cap: 5 (safety limit)

        Args:
            url: URL to evaluate

        Returns:
            Effective depth limit for this URL branch
        """
        if not self.adaptive_depth:
            return self.max_depth

        bonus = 0
        url_lower = url.lower()

        # API and reference pages get highest bonus
        if "/api/" in url_lower or "/reference/" in url_lower:
            bonus = 2
        # Other doc patterns get +1
        elif any(pattern in url_lower for pattern in self.DOC_PATTERNS):
            bonus = 1

        # Cap at 5 for safety
        return min(self.max_depth + bonus, 5)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get crawl statistics.

        Returns:
            Dict with pages_crawled, total_content_length, avg_depth
        """
        if not self.results:
            return {
                "pages_crawled": 0,
                "total_content_length": 0,
                "avg_depth": 0,
            }

        total_length = sum(r["metadata"]["content_length"] for r in self.results)
        avg_depth = sum(r["metadata"]["depth"] for r in self.results) / len(self.results)

        return {
            "pages_crawled": len(self.results),
            "total_content_length": total_length,
            "avg_depth": round(avg_depth, 2),
        }
