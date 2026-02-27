"""
Web fetch tool for Multi-Agent Dashboard.

Fetches webpage content, extracts main content, and converts to markdown.
Registered as "web_fetch" in the tool registry.
"""

import logging
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)

# Try to import LangChain BaseTool (optional)
try:
    from langchain.tools import BaseTool
    _LANGCHAIN_TOOLS_AVAILABLE = True
except ImportError:
    # Fallback for environments without LangChain
    class BaseTool:  # type: ignore
        pass
    _LANGCHAIN_TOOLS_AVAILABLE = False

# Try to import required libraries for web fetching
try:
    import requests
    from bs4 import BeautifulSoup
    from markdownify import markdownify as md
    _WEB_FETCH_DEPS_AVAILABLE = True
except ImportError:
    requests = None
    BeautifulSoup = None
    md = None
    _WEB_FETCH_DEPS_AVAILABLE = False
    logger.warning(f"Required libraries for web fetching missing. (Install beautifulsoup4 / markdownify)")

from .registry import register_tool


# JSON Schema matching the adapter's expected format
TOOL_SCHEMA: Dict[str, Any] = {
    "name": "web_fetch",
    "description": "Fetch the content of a webpage and convert it to markdown.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL of the webpage to fetch",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    },
}


if _LANGCHAIN_TOOLS_AVAILABLE and _WEB_FETCH_DEPS_AVAILABLE:
    @register_tool(
        name="web_fetch",
        description="Fetch the content of a webpage and convert it to markdown.",
        schema=TOOL_SCHEMA,
        tags=["web", "fetch", "scraping"],
    )
    class WebFetchTool(BaseTool):
        """Web Fetch Tool.
        
        Fetches webpage content, extracts main content, and converts to markdown.
        """
        
        name: str = "web_fetch"
        description: str = "Fetch the content of a webpage and convert it to markdown."
        args_schema: Type = None  # Will use JSON schema from registration
        
        def _run(
            self,
            url: str,
            **kwargs: Any,
        ) -> str:
            """
            Fetch webpage content and convert to markdown.
            
            Args:
                url: URL of the webpage to fetch
                **kwargs: Additional arguments (ignored)
            
            Returns:
                Markdown content with title
            """
            try:
                logger.info(f"Trying to fetch content from: {url}")
                headers = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                }
                response = requests.get(url, headers=headers)
                response.raise_for_status()

                html = response.text
                soup = BeautifulSoup(html, "html.parser")

                # Remove unnecessary elements
                for tag in soup(["script", "style", "nav", "footer", "iframe"]):
                    tag.decompose()
                for ad in soup.select(".ads"):
                    ad.decompose()

                title = (soup.title.string if soup.title else None) or (soup.find("h1").get_text(strip=True) if soup.find("h1") else "")
                main_content = (
                    soup.find("article") or
                    soup.find("main") or
                    soup.select_one(".content") or
                    soup.select_one("#content") or
                    soup.select_one(".post") or
                    soup.body
                )
                content_html = str(main_content) if main_content else ""
                content = md(content_html, heading_style="atx", code_language=None)

                result = f"Title: {title.strip() if title else ''}\n\n{content}"
                logger.info(f"Web fetch succeeded for {url}, title: {title}")
                return result
            except Exception as e:
                logger.error(f"Failed to fetch {url}: {e}")
                return f"Error fetching webpage: {str(e)}"
        
        async def _arun(self, *args, **kwargs):
            """Async version (not implemented)."""
            raise NotImplementedError("Async web fetch not supported")

else:
    # Create a dummy class for environments without required dependencies
    class WebFetchTool:
        """Dummy class when dependencies are not available."""
        
        def __init__(self, **kwargs):
            missing = []
            if not _LANGCHAIN_TOOLS_AVAILABLE:
                missing.append("langchain.tools")
            if not _WEB_FETCH_DEPS_AVAILABLE:
                missing.append("requests, beautifulsoup4, markdownify")
            raise ImportError(
                f"WebFetchTool requires {', '.join(missing)}. "
                "Install with: pip install langchain requests beautifulsoup4 markdownify"
            )


# Export for easy import
__all__ = ["WebFetchTool", "TOOL_SCHEMA"]