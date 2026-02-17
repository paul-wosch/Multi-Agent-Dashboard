"""
DuckDuckGo Search Tool for Multi-Agent Dashboard.

Provides a LangChain BaseTool wrapper for DuckDuckGoSearchRun with optional domain filtering.
Registered as "web_search_ddg" in the tool registry.
"""
# ---------------------------------------------------------------------
# Customized LangChain Duckduckgo API wrapper
# - accepts backend argument
# - default: "duckduckgo"; only used for text search)
#
# Source code copied from original library:
# .venv/lib/python3.14/site-packages/langchain_community/utilities/duckduckgo_search.py
#
# """Util that calls DuckDuckGo Search.
#
# No setup required. Free.
# https://pypi.org/project/duckduckgo-search/
# """
# ---------------------------------------------------------------------
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, model_validator


class DuckDuckGoSearchAPIWrapper(BaseModel):
    """Wrapper for DuckDuckGo Search API.

    Free and does not require any setup.
    """

    region: Optional[str] = "wt-wt"
    """
    See https://pypi.org/project/duckduckgo-search/#regions
    """
    safesearch: str = "moderate"
    """
    Options: strict, moderate, off
    """
    time: Optional[str] = "y"
    """
    Options: d, w, m, y
    """
    max_results: int = 5
    backend: str = "duckduckgo"
    """
    Options: auto, duckduckgo
    """
    source: str = "text"
    """
    Options: text, news, images
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that python package exists in environment."""
        try:
            from ddgs import DDGS  # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import ddgs python package. "
                "Please install it with `pip install -U ddgs`."
            )
        return values

    def _ddgs_text(
        self, query: str, max_results: Optional[int] = None, backend: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Run query through DuckDuckGo text search and return results."""
        from ddgs import DDGS

        with DDGS() as ddgs:
            ddgs_gen = ddgs.text(
                query,
                region=self.region,
                safesearch=self.safesearch,
                timelimit=self.time,
                max_results=max_results or self.max_results,
                backend=backend or self.backend,
            )
            if ddgs_gen:
                return [r for r in ddgs_gen]
        return []

    def _ddgs_news(
        self, query: str, max_results: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Run query through DuckDuckGo news search and return results."""
        from ddgs import DDGS

        with DDGS() as ddgs:
            ddgs_gen = ddgs.news(
                query,
                region=self.region,
                safesearch=self.safesearch,
                timelimit=self.time,
                max_results=max_results or self.max_results,
            )
            if ddgs_gen:
                return [r for r in ddgs_gen]
        return []

    def _ddgs_images(
        self, query: str, max_results: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Run query through DuckDuckGo image search and return results."""
        from ddgs import DDGS

        with DDGS() as ddgs:
            ddgs_gen = ddgs.images(
                query,
                region=self.region,
                safesearch=self.safesearch,
                max_results=max_results or self.max_results,
            )
            if ddgs_gen:
                return [r for r in ddgs_gen]
        return []

    def run(self, query: str) -> str:
        """Run query through DuckDuckGo and return concatenated results."""
        if self.source == "text":
            results = self._ddgs_text(query)
        elif self.source == "news":
            results = self._ddgs_news(query)
        elif self.source == "images":
            results = self._ddgs_images(query)
        else:
            results = []

        if not results:
            return "No good DuckDuckGo Search Result was found"
        return " ".join(r["body"] for r in results)

    def results(
        self, query: str, max_results: int, source: Optional[str] = None, backend: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Run query through DuckDuckGo and return metadata.

        Args:
            query: The query to search for.
            max_results: The number of results to return.
            source: The source to look from.

        Returns:
            A list of dictionaries with the following keys:
                snippet - The description of the result.
                title - The title of the result.
                link - The link to the result.
        """
        source = source or self.source
        if source == "text":
            results = [
                {"snippet": r["body"], "title": r["title"], "link": r["href"]}
                for r in self._ddgs_text(query, max_results=max_results, backend=backend)
            ]
        elif source == "news":
            results = [
                {
                    "snippet": r["body"],
                    "title": r["title"],
                    "link": r["url"],
                    "date": r["date"],
                    "source": r["source"],
                }
                for r in self._ddgs_news(query, max_results=max_results)
            ]
        elif source == "images":
            results = [
                {
                    "title": r["title"],
                    "thumbnail": r["thumbnail"],
                    "image": r["image"],
                    "url": r["url"],
                    "height": r["height"],
                    "width": r["width"],
                    "source": r["source"],
                }
                for r in self._ddgs_images(query, max_results=max_results)
            ]
        else:
            results = []

        if results is None:
            results = [{"Result": "No good DuckDuckGo Search Result was found"}]

        return results


# ---------------------------------------------------------------------
# Customized LangChain Duckduckgo tool
# Source code copied from original library:
# .venv/lib/python3.14/site-packages/langchain_community/utilities/duckduckgo_search.py
#
# """Tool for the DuckDuckGo search API."""
# ---------------------------------------------------------------------
import json
import warnings
from typing import Any, List, Literal, Optional, Type, Union

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# Import disabled to force use of patched Duckduckgo API wrapper
# from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper


class DDGInput(BaseModel):
    """Input for the DuckDuckGo search tool."""

    query: str = Field(description="search query to look up")


class DuckDuckGoSearchRun(BaseTool):
    """DuckDuckGo tool.

    Setup:
        Install ``duckduckgo-search`` and ``langchain-community``.

        .. code-block:: bash

            pip install -U duckduckgo-search langchain-community

    Instantiation:
        .. code-block:: python

            from langchain_community.tools import DuckDuckGoSearchResults

            tool = DuckDuckGoSearchResults()

    Invocation with args:
        .. code-block:: python

            tool.invoke("Obama")

        .. code-block:: python

            '[snippet: Users on X have been widely comparing the boost of support felt for Kamala Harris\' campaign to Barack Obama\'s in 2008., title: Surging Support For Kamala Harris Compared To Obama-Era Energy, link: https://www.msn.com/en-us/news/politics/surging-support-for-kamala-harris-compared-to-obama-era-energy/ar-BB1qzdC0, date: 2024-07-24T18:27:01+00:00, source: Newsweek on MSN.com], [snippet: Harris tried to emulate Obama\'s coalition in 2020 and failed. She may have a better shot at reaching young, Black, and Latino voters this time around., title: Harris May Follow Obama\'s Path to the White House After All, link: https://www.msn.com/en-us/news/politics/harris-may-follow-obama-s-path-to-the-white-house-after-all/ar-BB1qv9d4, date: 2024-07-23T22:42:00+00:00, source: Intelligencer on MSN.com], [snippet: The Republican presidential candidate said in an interview on Fox News that he "wouldn\'t be worried" about Michelle Obama running., title: Donald Trump Responds to Michelle Obama Threat, link: https://www.msn.com/en-us/news/politics/donald-trump-responds-to-michelle-obama-threat/ar-BB1qqtu5, date: 2024-07-22T18:26:00+00:00, source: Newsweek on MSN.com], [snippet: H eading into the weekend at his vacation home in Rehoboth Beach, Del., President Biden was reportedly stewing over Barack Obama\'s role in the orchestrated campaign to force him, title: Opinion | Barack Obama Strikes Again, link: https://www.msn.com/en-us/news/politics/opinion-barack-obama-strikes-again/ar-BB1qrfiy, date: 2024-07-22T21:28:00+00:00, source: The Wall Street Journal on MSN.com]'

    Invocation with ToolCall:

        .. code-block:: python

            tool.invoke({"args": {"query":"Obama"}, "id": "1", "name": tool.name, "type": "tool_call"})

        .. code-block:: python

            ToolMessage(content="[snippet: Biden, Obama and the Clintons Will Speak at the Democratic Convention. The president, two of his predecessors and the party's 2016 nominee are said to be planning speeches at the party's ..., title: Biden, Obama and the Clintons Will Speak at the Democratic Convention ..., link: https://www.nytimes.com/2024/08/12/us/politics/dnc-speakers-biden-obama-clinton.html], [snippet: Barack Obama—with his wife, Michelle—being sworn in as the 44th president of the United States, January 20, 2009. Key events in the life of Barack Obama. Barack Obama (born August 4, 1961, Honolulu, Hawaii, U.S.) is the 44th president of the United States (2009-17) and the first African American to hold the office., title: Barack Obama | Biography, Parents, Education, Presidency, Books ..., link: https://www.britannica.com/biography/Barack-Obama], [snippet: Former President Barack Obama released a letter about President Biden's decision to drop out of the 2024 presidential race. Notably, Obama did not name or endorse Vice President Kamala Harris., title: Read Obama's full statement on Biden dropping out - CBS News, link: https://www.cbsnews.com/news/barack-obama-biden-dropping-out-2024-presidential-race-full-statement/], [snippet: Many of the marquee names in Democratic politics began quickly lining up behind Vice President Kamala Harris on Sunday, but one towering presence in the party held back: Barack Obama. The former ..., title: Why Obama Hasn't Endorsed Harris - The New York Times, link: https://www.nytimes.com/2024/07/21/us/politics/why-obama-hasnt-endorsed-harris.html]", name='duckduckgo_results_json', tool_call_id='1')
    """  # noqa: E501

    name: str = "duckduckgo_search"
    description: str = (
        "A wrapper around DuckDuckGo Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    api_wrapper: DuckDuckGoSearchAPIWrapper = Field(
        default_factory=DuckDuckGoSearchAPIWrapper
    )
    args_schema: Type[BaseModel] = DDGInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)


class DuckDuckGoSearchResults(BaseTool):
    """Tool that queries the DuckDuckGo search API and
    returns the results in `output_format`."""

    name: str = "duckduckgo_results_json"
    description: str = (
        "A wrapper around Duck Duck Go Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    max_results: int = Field(alias="num_results", default=4)
    api_wrapper: DuckDuckGoSearchAPIWrapper = Field(
        default_factory=DuckDuckGoSearchAPIWrapper
    )
    backend: str = "text"
    args_schema: Type[BaseModel] = DDGInput
    keys_to_include: Optional[List[str]] = None
    """Which keys from each result to include. If None all keys are included."""
    results_separator: str = ", "
    """Character for separating results."""
    output_format: Literal["string", "json", "list"] = "string"
    """Output format of the search results.

    - 'string': Return a concatenated string of the search results.
    - 'json': Return a JSON string of the search results.
    - 'list': Return a list of dictionaries of the search results.
    """
    response_format: Literal["content_and_artifact"] = "content_and_artifact"

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> tuple[Union[List[dict], str], List[dict]]:
        """Use the tool."""
        raw_results = self.api_wrapper.results(
            query, self.max_results, backend=self.backend # source=self.backend
        )
        results = [
            {
                k: v
                for k, v in d.items()
                if not self.keys_to_include or k in self.keys_to_include
            }
            for d in raw_results
        ]

        if self.output_format == "list":
            return results, raw_results
        elif self.output_format == "json":
            return json.dumps(results), raw_results
        elif self.output_format == "string":
            res_strs = [", ".join([f"{k}: {v}" for k, v in d.items()]) for d in results]
            return self.results_separator.join(res_strs), raw_results
        else:
            raise ValueError(
                f"Invalid output_format: {self.output_format}. "
                "Needs to be one of 'string', 'json', 'list'."
            )

# ---------------------------------------------------------------------
# Duckduckgo custom tool integration
# ---------------------------------------------------------------------

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

# Try to import DuckDuckGoSearchRun from langchain_community
try:
    # Import disabled to force use of patched Duckduckgo tool classes
    # from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
    _DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DuckDuckGoSearchRun = None      # type: ignore
    DuckDuckGoSearchResults = None  # type: ignore
    _DUCKDUCKGO_AVAILABLE = False

from .registry import register_tool


# JSON Schema matching the adapter's expected format
# This must match the schema in provider_tool_adapter.py:_convert_web_search_ddg_tool()
TOOL_SCHEMA: Dict[str, Any] = {
    "name": "duckduckgo_search",
    "description": "Search the web using DuckDuckGo. Returns relevant web pages with snippets.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            },
            "domain_filter": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of domains to restrict results to",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


if _LANGCHAIN_TOOLS_AVAILABLE and _DUCKDUCKGO_AVAILABLE:
    @register_tool(
        name="web_search_ddg",
        description="Search the web using DuckDuckGo. Returns relevant web pages with snippets.",
        schema=TOOL_SCHEMA,
        tags=["search", "web", "duckduckgo"],
    )
    class DuckDuckGoSearchTool(BaseTool):
        """DuckDuckGo Search Tool with optional domain filtering."""
        
        name: str = "duckduckgo_search"
        description: str = "Search the web using DuckDuckGo. Returns relevant web pages with snippets."
        args_schema: Type = None  # Will use JSON schema from registration
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._search_tool = DuckDuckGoSearchResults(output_format="list", backend="duckduckgo")
        
        def _run(
            self,
            query: str,
            domain_filter: Optional[list] = None,
            **kwargs: Any,
        ) -> str:
            """
            Execute DuckDuckGo search.
            
            Args:
                query: Search query
                domain_filter: Optional list of domains to restrict results to
                **kwargs: Additional arguments
            
            Returns:
                Search results as text
            """

            # Use instance attribute if set (from UI domain filters), else fallback to parameter
            effective_filter = getattr(self, '_domain_filter', None) or domain_filter
            if effective_filter:
                logger.info(f"DuckDuckGo search filter: {effective_filter}")
                # Normalize domain filter with multiple domains for query
                domain_clauses = [f"site:{domain}" for domain in effective_filter]
                domain_clauses = [f"OR {clause}" if i > 0 else clause for i, clause in enumerate(domain_clauses)]
                # Inject domain filter to query
                domain_filter_string = " ".join(domain_clauses)
                query = f"{query} {domain_filter_string}"
                logger.info(f"DuckDuckGo search query: {query}")

            try:
                result = self._search_tool.invoke(query)
                return str(result)
            except Exception as e:
                logger.error(f"DuckDuckGo search failed: {e}")
                return f"Error performing DuckDuckGo search: {str(e)}"
        
        async def _arun(self, *args, **kwargs):
            """Async version (not implemented)."""
            raise NotImplementedError("Async DuckDuckGo search not supported")
        
else:
    # Create a dummy class for environments without required dependencies
    class DuckDuckGoSearchTool:
        """Dummy class when dependencies are not available."""
        
        def __init__(self, **kwargs):
            raise ImportError(
                "DuckDuckGoSearchTool requires langchain.tools and langchain_community.tools. "
                "Install with: pip install langchain langchain-community"
            )


# Export for easy import
__all__ = ["DuckDuckGoSearchTool", "TOOL_SCHEMA"]