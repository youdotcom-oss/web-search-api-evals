from abc import abstractmethod
import asyncio
import logging
import sys
import traceback
from typing import Any

from evals.samplers.base_samplers.base_sampler import BaseSampler


class BaseSDKSampler(BaseSampler):
    """Base class for SDK-based samplers that use provider SDKs"""

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
        self.client = None
        if self.api_key:
            self._initialize_client()

    @abstractmethod
    def _initialize_client(self):
        """
        Initialize the SDK client with the API key.

        Returns:
            Initialized SDK client instance
        """
        pass

    @abstractmethod
    def _get_search_results_impl(self, query: str) -> Any:
        """
        Implementation of getting raw search results using the SDK client.
        This method should be implemented by derived classes.

        Args:
            query: The search query string

        Returns:
            Raw search results in provider-specific format
        """
        pass

    async def get_search_results(self, query: str) -> Any:
        """
        Get raw search results using the SDK client asynchronously.
        This method wraps _get_search_results_impl with error handling and timeout.

        Args:
            query: The search query string

        Returns:
            Raw search results in provider-specific format

        Raises:
            TimeoutError: If the search operation exceeds the timeout
            Exception: Re-raises any exception encountered during search
        """
        try:
            # Run synchronous SDK call in thread pool with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(self._get_search_results_impl, query),
                timeout=self.timeout,
            )
            return result
        except asyncio.TimeoutError:
            error_msg = f"{self.sampler_name} timed out after {self.timeout} seconds"
            logging.error(error_msg)
            raise TimeoutError(error_msg)
        except Exception as e:
            logging.error(f"{self.sampler_name} failed with error {e}")
            raise e

    async def _retry_with_backoff_async(self, func, *args, **kwargs):
        """Generic async retry logic with exponential backoff"""
        trial = 0
        while True:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                _, _, traceback_ = sys.exc_info()
                if trial >= self.max_retries:
                    logging.error(f"Failed after {self.max_retries} retries: {str(e)}")
                    raise

                trial += 1
                backoff_time = 2**trial
                logging.warning(
                    f"Attempt {trial}/{self.max_retries} failed: {traceback.print_tb(traceback_)}. Retrying in {backoff_time}s..."
                )
                await asyncio.sleep(backoff_time)
