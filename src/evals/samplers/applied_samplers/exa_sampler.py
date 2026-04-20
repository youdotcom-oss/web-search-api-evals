"""Run evals using the Exa SDK"""

from typing import Any

from exa_py import Exa

from evals.samplers.base_samplers.base_sdk_sampler import BaseSDKSampler


class ExaSampler(BaseSDKSampler):
    def __init__(
        self,
        sampler_name: str,
        api_key: str = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        needs_synthesis: bool = True,
        text: Any = False,
    ):
        self.text: bool = text
        super().__init__(
            sampler_name=sampler_name,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
            needs_synthesis=needs_synthesis,
        )

    def _initialize_client(self):
        self.client = Exa(self.api_key)

    def _get_search_results_impl(self, query: str) -> Any:
        return self.client.search(
            query=query, num_results=10, contents={"text": self.text}
        )

    def format_results(self, results: Any) -> list[str]:
        formatted_results = []

        raw_results = getattr(results, "results", None)
        for result in raw_results:
            title = getattr(result, "title", "")
            url = getattr(result, "url", "")
            text = getattr(result, "text", "")
            if text:
                formatted_results.append(f'[{title}]({url})\ntext: "{text}"\n')

        return formatted_results
