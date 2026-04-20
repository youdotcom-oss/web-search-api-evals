from typing import Any

import youdotcom
from youdotcom.models import LiveCrawl, LiveCrawlFormats, ResearchEffort

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
        self.client = youdotcom.You(self.api_key)

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

            if result.contents and "markdown" in result.contents.__dict__.keys():
                contents = result.contents.markdown
                formatted_result = f"[{title}]({url})\n{contents}"
                formatted_results.append(formatted_result)
            else:
                description = result.description
                snippet = result.snippets
                if snippet and isinstance(snippet, list):
                    snippet = " ".join(snippet)
                formatted_result = f"[{title}]({url})\n snippet: {snippet}\n description: {description}"
                formatted_results.append(formatted_result)

        return formatted_results


class YouSearchSnippetsSampler(YouSearchSampler):
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
            livecrawl_formats=LiveCrawlFormats.MARKDOWN,
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
        self.research_effort = research_effort

    def _get_search_results_impl(self, query: str) -> Any:
        return self.client.research(
            input=query,
            research_effort=self.research_effort,
        )

    def format_results(self, results: Any) -> list[str]:
        return results.output.content
