from typing import Any, Dict

import youdotcom
from youdotcom.models import LiveCrawl, LiveCrawlFormats, ResearchEffort

from evals.samplers.base_samplers.base_api_sampler import BaseAPISampler
from evals.samplers.base_samplers.base_sdk_sampler import (
    BaseSDKSampler,
)


class YouSampler(BaseSDKSampler):
    def __init__(
        self,
        sampler_name: str,
        api_key: str = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        needs_synthesis: bool = True,
    ):
        super().__init__(
            sampler_name=sampler_name,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
            needs_synthesis=needs_synthesis,
        )

    def _initialize_client(self):
        self.client = youdotcom.You(self.api_key, timeout_ms=int(self.timeout * 1000))

    def _get_search_results_impl(self, query: str) -> Any:
        pass


class YouSearchSampler(YouSampler):
    def __init__(
        self,
        sampler_name: str,
        api_key: str = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        needs_synthesis: bool = True,
        include_news_results: bool = False,
    ):
        super().__init__(
            sampler_name=sampler_name,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
            needs_synthesis=needs_synthesis,
        )
        self.include_news_results = include_news_results

    def format_results(self, results: Any) -> list[str]:
        formatted_results = []
        raw_results = []
        if results.results and results.results.web:
            raw_results.extend(results.results.web)
        if self.include_news_results and results.results and results.results.news:
            raw_results.extend(results.results.news)

        for result in raw_results:
            title = result.title
            url = result.url

            contents = result.contents
            if contents and contents.markdown:
                formatted_result = f"[{title}]({url})\n{contents.markdown}"
            else:
                highlights = " ".join(contents.highlights or []) if contents else ""
                formatted_result = f"[{title}]({url})\n highlights: {highlights}\n description: {result.description}"
            formatted_results.append(formatted_result)

        return formatted_results


class YouSearchHighlightsSampler(YouSearchSampler):
    def __init__(
        self,
        sampler_name: str,
        api_key: str = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        needs_synthesis: bool = True,
        include_news_results: bool = False,
    ):
        super().__init__(
            sampler_name=sampler_name,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
            needs_synthesis=needs_synthesis,
            include_news_results=include_news_results,
        )

    def _get_search_results_impl(self, query: str) -> Any:
        return self.client.search.unified(
            query=query,
            count=10,
            extraction={"extraction_mode": "highlights"},
        )


class YouLivecrawlSampler(YouSearchSampler):
    def __init__(
        self,
        sampler_name: str,
        api_key: str = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        needs_synthesis: bool = True,
        include_news_results: bool = False,
    ):
        super().__init__(
            sampler_name=sampler_name,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
            needs_synthesis=needs_synthesis,
            include_news_results=include_news_results,
        )

    def _get_search_results_impl(self, query: str) -> Any:
        if self.include_news_results:
            livecrawl = LiveCrawl.ALL
        else:
            livecrawl = LiveCrawl.WEB

        return self.client.search.unified(
            query=query,
            count=10,
            livecrawl=livecrawl,
            livecrawl_formats=[LiveCrawlFormats.MARKDOWN],
        )


class YouResearchSampler(YouSampler):
    def __init__(
        self,
        sampler_name: str,
        api_key: str = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        needs_synthesis: bool = False,
        research_effort: ResearchEffort = ResearchEffort.STANDARD,
    ):
        super().__init__(
            sampler_name=sampler_name,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
            needs_synthesis=needs_synthesis,
        )
        self.max_concurrency = 5
        self.research_effort = research_effort

    def _get_search_results_impl(self, query: str) -> Any:
        return self.client.research(
            input=query,
            research_effort=self.research_effort,
        )

    def format_results(self, results: Any) -> list[str]:
        return results.output.content


class YouFinanceResearchSampler(BaseAPISampler):
    """Sampler for the You.com finance research API.

    Calls POST /v1/finance_research with {"input": query, "research_effort": ...}
    and returns output.content directly (no synthesis step needed).
    """

    def __init__(
        self,
        sampler_name: str,
        api_key: str = None,
        research_effort: str = "deep",
        timeout: float = 300.0,
        max_retries: int = 3,
        base_url: str | None = None,
    ):
        self.research_effort = research_effort
        self._base_url = base_url or "https://api.you.com"
        super().__init__(
            sampler_name=sampler_name,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            needs_synthesis=False,
        )
        self.max_concurrency = 5

    def _get_base_url(self) -> str:
        return self._base_url

    @staticmethod
    def _get_endpoint() -> str:
        return "/v1/finance_research"

    @staticmethod
    def _get_method() -> str:
        return "POST"

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-API-Key": self.api_key,
        }

    def _get_payload(self, query: str) -> Dict[str, Any]:
        return {"input": query, "research_effort": self.research_effort}

    def format_results(self, results: Any) -> str:
        output = results.get("output", {}) if isinstance(results, dict) else {}
        return output.get("content", "")
