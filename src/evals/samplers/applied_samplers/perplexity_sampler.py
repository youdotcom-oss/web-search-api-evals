from typing import Any, Dict

from evals.samplers.base_samplers.base_api_sampler import BaseAPISampler


class PerplexityFinanceSearchSampler(BaseAPISampler):
    """Sampler using Perplexity's Agent API with the finance_search tool.

    Calls POST /v1/agent with finance_search (and optionally web_search + fetch_url)
    and returns the final assistant message directly — no synthesis step needed.

    Recommended model choices (per Perplexity docs):
      - "perplexity/sonar"       -> live quotes / real-time data (fast, cheap)
      - "openai/gpt-5.5"         -> single-company historical lookups (balanced)
      - "anthropic/claude-opus-4-7" -> multi-step cross-company research (thorough)
    """

    def __init__(
        self,
        sampler_name: str,
        api_key: str = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        model: str = "openai/gpt-5.5",
        max_steps: int = 5,
        max_tokens: int = 2048,
        include_web_search: bool = True,
        include_fetch_url: bool = True,
        reasoning_effort: str = None,
    ):
        self.model = model
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.include_web_search = include_web_search
        self.include_fetch_url = include_fetch_url
        self.reasoning_effort = reasoning_effort

        super().__init__(
            sampler_name=sampler_name,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            needs_synthesis=False,
        )

    @staticmethod
    def _get_base_url() -> str:
        return "https://api.perplexity.ai"

    @staticmethod
    def _get_endpoint() -> str:
        return "/v1/agent"

    @staticmethod
    def _get_method() -> str:
        return "POST"

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _get_payload(self, query: str) -> Dict[str, Any]:
        tools = [{"type": "finance_search"}]
        if self.include_web_search:
            tools.append({"type": "web_search"})
        if self.include_fetch_url:
            tools.append({"type": "fetch_url"})
        payload = {
            "model": self.model,
            "input": query,
            "tools": tools,
            "max_steps": self.max_steps,
            "max_output_tokens": self.max_tokens,
        }
        if self.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        return payload

    def format_results(self, results: Any) -> str:
        output = results.get("output", []) if isinstance(results, dict) else []
        for item in reversed(output):
            if item.get("type") == "message":
                for block in item.get("content", []):
                    if block.get("type") == "output_text":
                        return block["text"]
        return ""


class PerplexityDeepSearchSampler(BaseAPISampler):
    """Sampler using Perplexity's deep research chat-completions endpoint."""

    def __init__(
        self,
        sampler_name: str,
        api_key: str = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        needs_synthesis: bool = True,
        model: str = "sonar-deep-research",
        search_effort: str = "medium",
    ):
        self.model = model
        self.search_effort = search_effort
        super().__init__(
            sampler_name=sampler_name,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            needs_synthesis=needs_synthesis,
        )

    @staticmethod
    def _get_base_url() -> str:
        return "https://api.perplexity.ai"

    @staticmethod
    def _get_endpoint() -> str:
        return "/chat/completions"

    @staticmethod
    def _get_method() -> str:
        return "POST"

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _get_payload(self, query: str) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": query}],
            "reasoning_effort": self.search_effort,
        }

    def format_results(self, results: Any) -> list[str]:
        if self.needs_synthesis:
            return [row["message"]["content"] for row in results["choices"]]
        raise ValueError("Need to route to answer if synthesis is not needed")
