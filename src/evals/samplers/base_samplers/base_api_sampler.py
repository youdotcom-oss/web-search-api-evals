import asyncio
import logging
import random
from abc import abstractmethod
from typing import Any, Dict

import aiohttp

from evals.samplers.base_samplers.base_sampler import BaseSampler

RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


class BaseAPISampler(BaseSampler):
    """Base class for API-based samplers that make HTTP requests"""

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

    def _set_params(self):
        """Set API parameters before making a request"""
        self.base_url = self._get_base_url()
        self.method = self._get_method()
        self.headers = self._get_headers()
        self.endpoint = self._get_endpoint()

    @staticmethod
    @abstractmethod
    def _get_base_url() -> str:
        """Get provider specific base url"""
        pass

    @abstractmethod
    def _get_headers(self) -> Dict[str, str]:
        """Get provider specific headers"""
        pass

    @abstractmethod
    def _get_payload(self, query: str) -> Dict[str, Any]:
        """Get provider specific request payload"""
        pass

    @staticmethod
    @abstractmethod
    def _get_endpoint() -> str:
        """Get provider specific API endpoint"""
        pass

    @staticmethod
    @abstractmethod
    def _get_method() -> str:
        """Get provider specific HTTP method"""
        pass

    async def _request(self, payload: Dict[str, Any]) -> Any:
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if self.method not in ("GET", "POST"):
                raise ValueError(
                    'Unsupported method, please select between ["POST", "GET"]'
                )
            kwargs = {"json": payload} if self.method == "POST" else {"params": payload}
            async with session.request(
                self.method,
                self.base_url + self.endpoint,
                headers=self.headers,
                **kwargs,
            ) as response:
                response.raise_for_status()
                return await response.json()

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        if isinstance(error, aiohttp.ClientResponseError):
            return error.status in RETRYABLE_STATUSES
        return isinstance(error, (aiohttp.ClientConnectionError, asyncio.TimeoutError))

    async def get_search_results(self, query: str) -> Any:
        """Get raw search results from the API, retrying transient failures with backoff"""
        self._set_params()
        payload = self._get_payload(query)

        for attempt in range(self.max_retries + 1):
            try:
                return await self._request(payload)
            except Exception as e:
                if attempt == self.max_retries or not self._is_retryable(e):
                    logging.error(f"{self.sampler_name} failed with error {e}")
                    raise
                backoff = 2**attempt + random.random()
                logging.warning(
                    f"{self.sampler_name} attempt {attempt + 1}/{self.max_retries} failed: {e}. Retrying in {backoff:.1f}s"
                )
                await asyncio.sleep(backoff)
