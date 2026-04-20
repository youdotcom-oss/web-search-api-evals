from typing import Any

from parallel import Parallel

from evals.samplers.base_samplers.base_sdk_sampler import BaseSDKSampler


class ParallelSearchSampler(BaseSDKSampler):
    """Parallel sampler using the Search API"""

    def __init__(
        self,
        sampler_name: str,
        api_key: str = None,
        timeout: float = 60.0,
        max_characters: int | None = None,
        mode: str = "one-shot",
    ):
        self.max_characters = max_characters
        self.mode = mode

        super().__init__(
            sampler_name=sampler_name,
            api_key=api_key,
            timeout=timeout,
        )

    def _initialize_client(self):
        """Initialize Parallel SDK client"""
        self.client = Parallel(api_key=self.api_key)

    def _get_search_results_impl(self, query):
        search_params = {
            "mode": self.mode,
            "objective": query,
            "max_results": 10,
        }

        if self.max_characters is not None:
            search_params["excerpts"] = {"max_chars_per_result": self.max_characters}

        response = self.client.beta.search(**search_params)
        return response

    def format_results(self, results: Any) -> list[str]:
        formatted_results = []
        if results and results.results:
            for result in results.results:
                title = result.title
                url = result.url
                content = "\n".join(result.excerpts)
                formatted_results.append(f"[{title}]({url})\n{content}\n")
        return formatted_results
