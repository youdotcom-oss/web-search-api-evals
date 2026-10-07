import asyncio
import types

import aiohttp
import pytest

from evals.samplers.base_samplers import base_api_sampler


class StubSampler(base_api_sampler.BaseAPISampler):
    calls = 0
    failures = []

    @staticmethod
    def _get_base_url():
        return "http://x"

    def _get_headers(self):
        return {}

    def _get_payload(self, query):
        return {"q": query}

    @staticmethod
    def _get_endpoint():
        return "/s"

    @staticmethod
    def _get_method():
        return "GET"

    def format_results(self, raw_results):
        return raw_results

    async def _request(self, payload):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return {"ok": True}


def _status_error(status):
    return aiohttp.ClientResponseError(
        types.SimpleNamespace(real_url="http://x"), (), status=status
    )


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def instant(_):
        pass

    monkeypatch.setattr(base_api_sampler.asyncio, "sleep", instant)


def _run(failures, max_retries=3):
    sampler = StubSampler("stub", api_key="k", max_retries=max_retries)
    sampler.failures = list(failures)
    return sampler, sampler.get_search_results("q")


def test_retries_503_then_succeeds():
    sampler, coro = _run([_status_error(503), _status_error(503)])
    assert asyncio.run(coro) == {"ok": True}
    assert sampler.calls == 3


def test_does_not_retry_client_error():
    sampler, coro = _run([_status_error(401)])
    with pytest.raises(aiohttp.ClientResponseError):
        asyncio.run(coro)
    assert sampler.calls == 1


def test_gives_up_after_max_retries():
    sampler, coro = _run([_status_error(503)] * 5, max_retries=2)
    with pytest.raises(aiohttp.ClientResponseError):
        asyncio.run(coro)
    assert sampler.calls == 3
