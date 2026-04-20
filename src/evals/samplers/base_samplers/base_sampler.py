from abc import ABC, abstractmethod
import logging
import time
from typing import Any, Dict

from evals.configs import datasets
from evals.processing import synthesizer_utils


class BaseSampler(ABC):
    """Base class for all samplers with common functionality"""

    def __init__(
        self,
        sampler_name: str,
        api_key: str = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        needs_synthesis: bool = True,
    ):
        self.api_key = api_key
        self.sampler_name = sampler_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.needs_synthesis = needs_synthesis

    @abstractmethod
    async def get_search_results(self, query: str) -> Any:
        """
        Get raw search results from the API or SDK.

        Args:
            query: The search query string

        Returns:
            Raw search results in provider-specific format
        """
        pass

    @abstractmethod
    def format_results(self, results: Any) -> list[str]:
        """
        Format search results.

        Args:
            results: Raw search results from get_search_results

        Returns:
            tuple: (formatted_results) where formatted_results is either:
                - str: Already synthesized answer (no further synthesis needed)
                - list[str]: List of individual search results (needs synthesis)
        """
        pass

    @staticmethod
    def __extract_query_from_messages__(message_list: list[dict]) -> str:
        """Extract query from message list"""
        if isinstance(message_list, list) and len(message_list) > 0:
            last_message = message_list[-1]
            if isinstance(last_message, dict) and "content" in last_message:
                return last_message["content"]
        return str(message_list)

    @staticmethod
    async def __evaluate_response(
        query: str, ground_truth: str, generated_answer: str, dataset: datasets.Dataset
    ) -> Dict[str, Any]:
        """Evaluate the generated response against ground truth"""
        return await dataset.grader(query, ground_truth, generated_answer)

    async def __call__(
        self,
        query_input,
        dataset: datasets.Dataset,
        ground_truth: str = "",
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Main execution pipeline"""
        if not self.api_key:
            raise ValueError(
                f"API key not provided for sampler {self.sampler_name}. Ensure .env file is configured and contains necessary API keys"
            )

        internal_response_time_ms = None
        request_response_time_ms = None
        if isinstance(query_input, list):
            query = self.__extract_query_from_messages__(query_input)
        else:
            query = str(query_input)

        # Get raw results
        try:
            # For providers that do not report internal latency, calculate the latency of the request
            response_start_time = time.time()
            raw_results = await self.get_search_results(query)
            response_end_time_ms = time.time()
            if self.sampler_name == "you_search_with_livecrawl":
                internal_response_time_ms = round(
                    raw_results.metadata.latency * 1000, 2
                )  # Convert to ms
            elif self.sampler_name == "you_search":
                internal_response_time_ms = round(
                    raw_results.metadata.latency * 1000, 2
                )  # Convert to ms
            elif "tavily" in self.sampler_name:
                internal_response_time_ms = round(
                    raw_results["response_time"] * 1000, 2
                )  # Convert to ms

            if response_end_time_ms is not None:
                request_response_time_ms = round(
                    (response_end_time_ms - response_start_time) * 1000, 2
                )

            formatted_results = self.format_results(raw_results)
        except Exception as e:
            (
                raw_results,
                internal_response_time_ms,
                request_response_time_ms,
                formatted_results,
            ) = (
                "FAILED",
                "FAILED",
                "FAILED",
                "FAILED",
            )
            logging.exception(e)

        # Synthesize raw results
        try:
            if raw_results == "FAILED":
                generated_answer = "FAILED"
            elif self.needs_synthesis:
                generated_answer = await synthesizer_utils.synthesize_response(
                    query, formatted_results
                )
            else:
                generated_answer = formatted_results  # Already synthesized by API
        except Exception as e:
            generated_answer = "FAILED"
            logging.exception(e)

        # Evaluated synthesized results against ground truth
        try:
            if generated_answer == "FAILED":
                # Failed to get an answer, do not grade
                evaluation_result = "FAILED"
            elif ground_truth:
                evaluation_result_dict = await self.__evaluate_response(
                    query, ground_truth, generated_answer, dataset
                )
                evaluation_result = evaluation_result_dict["score_name"]
            else:
                raise ValueError("Ground truth is missing")
        except Exception as e:
            evaluation_result = "FAILED"
            logging.exception(e)

        # Format result
        result = {
            "query": query,
            "internal_response_time_ms": internal_response_time_ms,
            "request_response_time_ms": request_response_time_ms,
            "evaluation_result": evaluation_result,
            "generated_answer": generated_answer,
            "ground_truth": ground_truth,
            # Commenting these out because they are bloating the results files. Feel free to uncomment if you want extra metadata.
            # "raw_results": raw_results,
            # "formatted_results": formatted_results,
        }
        return result
