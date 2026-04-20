from abc import abstractmethod
from typing import Any, Dict

import aiohttp

from evals.samplers.base_samplers.base_sampler import BaseSampler


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

    async def get_search_results(self, query: str) -> Any:
        """Get raw search results from the API using async HTTP"""
        try:
            self._set_params()
            payload = self._get_payload(query)

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if self.method == "POST":
                    async with session.post(
                        self.base_url + self.endpoint,
                        json=payload,
                        headers=self.headers,
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                elif self.method == "GET":
                    async with session.get(
                        self.base_url + self.endpoint,
                        params=payload,
                        headers=self.headers,
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                else:
                    raise ValueError(
                        'Unsupported method, please select between ["POST", "GET"]'
                    )

                return data
        except Exception as e:
            print(f"{self.sampler_name} failed with error {e}")
            raise e
