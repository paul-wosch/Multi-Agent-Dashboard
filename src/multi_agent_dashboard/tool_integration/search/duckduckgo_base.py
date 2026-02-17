"""
Patched LangChain DuckDuckGo search utilities.

This file contains modified copies of two components
from LangChain Community:

1. `langchain_community/utilities/duckduckgo_search.py`
   – the source of `DuckDuckGoSearchAPIWrapper`
2  `langchain_community/tools/ddg_search/tool.py`
   – the source of `DuckDuckGoSearchRun` and `DuckDuckGoSearchResults`

Modifications:
- `DuckDuckGoSearchAPIWrapper` and `DuckDuckGoSearchResults`
  have been patched to accept a `backend` argument.
- `DuckDuckGoSearchRun` is included for completeness
  but does not expose the `backend` parameter.

The custom `DuckDuckGoSearchTool` in `duckduckgo_tool.py`
uses `DuckDuckGoSearchResults` and adds domain filtering
and registry registration.
"""

import json
import warnings
from typing import Any, Dict, List, Optional, Literal, Type, Union

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, model_validator


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
