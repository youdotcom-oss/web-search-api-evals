from typing import Any, Dict

from evals.samplers.base_samplers.base_api_sampler import BaseAPISampler


class GoogleSampler(BaseAPISampler):
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

    @staticmethod
    def _get_base_url():
        return "https://serpapi.com"

    @staticmethod
    def _get_endpoint() -> str:
        return "/search"

    @staticmethod
    def _get_method() -> str:
        return "GET"

    def _get_headers(self) -> Dict[str, str]:
        return {}

    def _get_payload(self, query: str) -> Dict[str, Any]:
        return {
            "q": query,
            "engine": "google",
            "num": 10,
            "api_key": self.api_key,
        }

    def format_results(self, results: Any) -> list[str]:
        formatted_results = []
        if "organic_results" in results:
            for result in results["organic_results"]:
                if isinstance(result, dict):
                    title = result.get("title", "")
                    link = result.get("link", "")
                    snippet = result.get("snippet", "")
                    if snippet and isinstance(snippet, list):
                        snippet = " ".join(snippet)
                    formatted_results.append(f"[{title}]({link})\n snippet: {snippet}")
        return formatted_results
