"""Run evals using the Tavily SDK"""

from typing import Any

from tavily import TavilyClient

from evals.samplers.base_samplers.base_sdk_sampler import BaseSDKSampler


class TavilySampler(BaseSDKSampler):
    def __init__(
        self,
        sampler_name: str,
        api_key: str = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        needs_synthesis: bool = True,
        search_depth: str = None,
    ):
        self.search_depth = search_depth
        super().__init__(
            sampler_name=sampler_name,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
            needs_synthesis=needs_synthesis,
        )

    def _initialize_client(self):
        self.client = TavilyClient(self.api_key)

    def _get_search_results_impl(self, query: str) -> Any:
        return self.client.search(
            query=query,
            max_results=10,
            search_depth=self.search_depth,
        )

    def format_results(self, results: Any) -> list[str]:
        formatted_results = []
        raw_results = results["results"]

        for result in raw_results:
            if isinstance(result, dict):
                title = result.get("title", "")
                url = result.get("url", "")
                content = result.get("content", "")
                if content:
                    formatted_results.append(f"[{title}]({url})\ncontent: {content}\n")

        return formatted_results
