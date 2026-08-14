from abc import abstractmethod
import logging
from typing import Any, Dict

import aiohttp

from evals.samplers.base_samplers.base_sampler import BaseSampler


# Enough of the provider's error body to identify the problem without flooding logs.
MAX_ERROR_BODY_CHARS = 2000


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

    @staticmethod
    async def _decode_response(response: aiohttp.ClientResponse) -> Any:
        """Return the decoded JSON body, or raise with the provider's error text.

        aiohttp's raise_for_status() reports only the status line, e.g.
        "400, message='Bad Request'". Providers put the actual reason in the
        response body -- which field was rejected, which parameter was invalid --
        and discarding it turns a one-line diagnosis into a debugging session.
        """
        if response.status >= 400:
            body = (await response.text())[:MAX_ERROR_BODY_CHARS]
            raise aiohttp.ClientResponseError(
                response.request_info,
                response.history,
                status=response.status,
                message=f"{response.reason}: {body}",
                headers=response.headers,
            )
        return await response.json()

    async def get_search_results(self, query: str) -> Any:
        """Get raw search results from the API using async HTTP"""
        try:
            self._set_params()
            payload = self._get_payload(query)
            url = self.base_url + self.endpoint

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if self.method == "POST":
                    request = session.post(url, json=payload, headers=self.headers)
                elif self.method == "GET":
                    request = session.get(url, params=payload, headers=self.headers)
                else:
                    raise ValueError(
                        'Unsupported method, please select between ["POST", "GET"]'
                    )

                async with request as response:
                    return await self._decode_response(response)
        except Exception as e:
            logging.error(f"{self.sampler_name} failed with error {e}")
            raise e
