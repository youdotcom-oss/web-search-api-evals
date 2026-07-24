"""Generic HTTP people-search sampler (any endpoint that returns people[]).

Request (POST JSON)::

    {
      "query": "<natural language query>",
      "metadata": {
        "benchmark_id": "fp_001",
        "persona": "...",
        "persona_slug": "recruiter",
        "query_type": "enrichment" | "search",
        "person_name": "...",   # enrichment rows
        "company": "..."
      }
    }

Response (JSON)::

    {
      "people": [
        {
          "displayname": "...",
          "current_title": "...",
          "current_company": "...",
          "location": "...",
          "linkedin_url": "...",
          "highlight": "...",
          "best_work_email": "...",
          "phones": ["..."],
          "top_skills": ["..."],
          "insights": {},
          "confidence": {"likelihood": 0.9}
        }
      ],
      "person_count": 1,
      "error": null
    }

Configure with ``PEOPLE_SEARCH_API_URL`` and optional ``PEOPLE_SEARCH_API_KEY``
(sent as ``Authorization: Bearer …``).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import aiohttp

from evals.processing.people_search.schema import normalize_people_payload
from evals.samplers.base_samplers.base_sampler import BaseSampler

logger = logging.getLogger(__name__)


def _parse_metadata(ground_truth: str) -> dict:
    if not ground_truth:
        return {}
    try:
        data = json.loads(ground_truth)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


class HttpPeopleSearchSampler(BaseSampler):
    """Call any people-search HTTP endpoint; score structured people[] output."""

    def __init__(
        self,
        sampler_name: str = "http_people_search",
        api_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        max_concurrency: int = 5,
    ):
        self.api_url = (api_url or os.getenv("PEOPLE_SEARCH_API_URL") or "").rstrip("/")
        # BaseSampler requires a truthy api_key; fall back to the URL as a sentinel
        # when the endpoint needs no auth.
        resolved_key = api_key or os.getenv("PEOPLE_SEARCH_API_KEY") or self.api_url
        super().__init__(
            sampler_name=sampler_name,
            api_key=resolved_key,
            timeout=timeout,
            max_retries=max_retries,
            needs_synthesis=False,
            max_concurrency=max_concurrency,
        )
        self._auth_key = api_key or os.getenv("PEOPLE_SEARCH_API_KEY") or ""
        self._eval_metadata: dict = {}

    async def __call__(
        self,
        query_input,
        dataset,
        ground_truth: str = "",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        self._eval_metadata = _parse_metadata(ground_truth)
        return await super().__call__(
            query_input, dataset, ground_truth=ground_truth, overwrite=overwrite
        )

    async def get_search_results(self, query: str) -> Any:
        if not self.api_url:
            raise ValueError(
                "PEOPLE_SEARCH_API_URL is required for http_people_search. "
                "Point it at any people-search endpoint that returns people[] JSON."
            )

        payload = {
            "query": query,
            "metadata": {
                **self._eval_metadata,
                "query_text": self._eval_metadata.get("query_text") or query,
            },
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._auth_key and self._auth_key != self.api_url:
            headers["Authorization"] = f"Bearer {self._auth_key}"

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.api_url, json=payload, headers=headers
            ) as response:
                text = await response.text()
                if response.status >= 400:
                    return {
                        "people": [],
                        "person_count": 0,
                        "error": f"HTTP {response.status}: {text[:500]}",
                    }
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {
                        "people": [],
                        "person_count": 0,
                        "error": f"Non-JSON response: {text[:500]}",
                    }

    def format_results(self, results: Any) -> str:
        payload = normalize_people_payload(results, provider=self.sampler_name)
        return json.dumps(payload, ensure_ascii=False)
