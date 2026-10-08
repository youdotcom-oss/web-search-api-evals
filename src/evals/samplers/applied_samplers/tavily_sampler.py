"""Run evals using the Tavily SDK"""

import time
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
    ):
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


class TavilyResearchSampler(BaseSDKSampler):
    """Tavily Research SDK sampler."""

    def __init__(
        self,
        sampler_name: str,
        api_key: str = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        research_model: str = "",
        needs_synthesis: bool = False,
    ):
        self.research_model = research_model
        super().__init__(
            sampler_name=sampler_name,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
            needs_synthesis=needs_synthesis,
        )

    def _initialize_client(self):
        self.client = TavilyClient(api_key=self.api_key)

    def _start_research_task(self, query: str) -> str:
        response = self.client.research(query, model=self.research_model)
        return response.get("request_id")

    def _get_task_result(self, research_request_id: str) -> Any:
        while True:
            result = self.client.get_research(research_request_id)
            status = result.get("status")
            if status == "completed":
                return result
            if status == "failed":
                raise ValueError(
                    f"Task {research_request_id} failed with error {result}"
                )
            time.sleep(10)

    def _get_search_results_impl(self, query: str) -> Any:
        research_request_id = self._start_research_task(query)
        return self._get_task_result(research_request_id)

    def format_results(self, results: Any) -> list[str]:
        return [results.get("content", "")]
