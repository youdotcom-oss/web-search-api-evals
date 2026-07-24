import os

from youdotcom.models import ResearchEffort

from evals.samplers.applied_samplers.exa_sampler import ExaResearchSampler, ExaSampler
from evals.samplers.applied_samplers.google_sampler import GoogleSampler
from evals.samplers.applied_samplers.parallel_sampler import (
    ParallelSearchSampler,
    ParallelTaskSampler,
)
from evals.samplers.applied_samplers.perplexity_sampler import (
    PerplexityDeepSearchSampler,
    PerplexityFinanceSearchSampler,
)
from evals.samplers.applied_samplers.tavily_sampler import (
    TavilyResearchSampler,
    TavilySampler,
)
from evals.samplers.applied_samplers.you_search_sampler import (
    YouFinanceResearchSampler,
    YouLivecrawlSampler,
    YouResearchSampler,
    YouSearchSnippetsSampler,
)
from evals.samplers.applied_samplers.people_search_sampler import HttpPeopleSearchSampler


SAMPLERS = [
    YouLivecrawlSampler(
        sampler_name="you_search_with_livecrawl",
        api_key=os.getenv("YOU_API_KEY"),
        include_news_results=False,
    ),
    YouResearchSampler(
        sampler_name="you_research_lite",
        api_key=os.getenv("YOU_API_KEY"),
        research_effort=ResearchEffort.LITE,
    ),
    YouResearchSampler(
        sampler_name="you_research_standard",
        api_key=os.getenv("YOU_API_KEY"),
        research_effort=ResearchEffort.STANDARD,
        timeout=120,
    ),
    YouResearchSampler(
        sampler_name="you_research_deep",
        api_key=os.getenv("YOU_API_KEY"),
        research_effort=ResearchEffort.DEEP,
        timeout=200,
    ),
    YouResearchSampler(
        sampler_name="you_research_exhaustive",
        api_key=os.getenv("YOU_API_KEY"),
        research_effort=ResearchEffort.EXHAUSTIVE,
        timeout=400,
    ),
    YouSearchSnippetsSampler(
        sampler_name="you_search",
        api_key=os.getenv("YOU_API_KEY"),
        include_news_results=False,
    ),
    # You.com finance research samplers
    YouFinanceResearchSampler(
        sampler_name="you_finance_research_deep",
        api_key=os.getenv("YOU_API_KEY"),
        research_effort="deep",
        timeout=300,
    ),
    YouFinanceResearchSampler(
        sampler_name="you_finance_research_exhaustive",
        api_key=os.getenv("YOU_API_KEY"),
        research_effort="exhaustive",
        timeout=600,
    ),
    ExaSampler(
        sampler_name="exa_search_with_text",
        api_key=os.getenv("EXA_API_KEY"),
        text={"max_characters": 20000},
    ),
    ExaResearchSampler(
        sampler_name="exa_research_pro",
        api_key=os.getenv("EXA_API_KEY"),
        research_model="exa-research-pro",
        timeout=3000,
    ),
    GoogleSampler(
        sampler_name="google_search",
        api_key=os.getenv("SERP_API_KEY"),
    ),
    ParallelSearchSampler(
        sampler_name="parallel_search_basic",
        api_key=os.getenv("PARALLEL_API_KEY"),
        mode="basic",
        num_results=10,
    ),
    ParallelTaskSampler(
        sampler_name="parallel_pro",
        api_key=os.getenv("PARALLEL_API_KEY"),
        processor="pro",
        timeout=3000,
    ),
    ParallelTaskSampler(
        sampler_name="parallel_ultra",
        api_key=os.getenv("PARALLEL_API_KEY"),
        processor="ultra",
        timeout=3000,
    ),
    TavilySampler(
        sampler_name="tavily_basic",
        api_key=os.getenv("TAVILY_API_KEY"),
        search_depth="basic",
    ),
    TavilySampler(
        sampler_name="tavily_advanced",
        api_key=os.getenv("TAVILY_API_KEY"),
        search_depth="advanced",
    ),
    TavilyResearchSampler(
        sampler_name="tavily_research_pro",
        api_key=os.getenv("TAVILY_API_KEY"),
        research_model="pro",
        timeout=3000,
    ),
    PerplexityFinanceSearchSampler(
        sampler_name="perplexity_finance_historical_lookup",
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        model="openai/gpt-5.5",
        max_steps=5,
        max_tokens=2048,
        include_web_search=True,
        include_fetch_url=True,
        reasoning_effort="low",
        timeout=120,
    ),
    PerplexityFinanceSearchSampler(
        sampler_name="perplexity_finance_multi_step_research",
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        model="anthropic/claude-opus-4-7",
        max_steps=10,
        max_tokens=4096,
        include_web_search=True,
        include_fetch_url=True,
        timeout=180,
    ),
    PerplexityDeepSearchSampler(
        sampler_name="perplexity_sonar_deep_research_high",
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        model="sonar-deep-research",
        search_effort="high",
        timeout=3000,
    ),
    # Generic people-search HTTP endpoint (scorer-based; excluded from defaults)
    HttpPeopleSearchSampler(
        sampler_name="http_people_search",
        api_url=os.getenv("PEOPLE_SEARCH_API_URL"),
        api_key=os.getenv("PEOPLE_SEARCH_API_KEY"),
        timeout=120,
        max_concurrency=5,
    ),
]

# Samplers excluded from default runs due to high cost, long latency, or special datasets
EXCLUDE_KEYWORDS = [
    "research",
    "parallel_pro",
    "parallel_ultra",
    "perplexity_finance_historical_lookup",
    "http_people_search",
]

NON_RESEARCH_SAMPLERS = [
    sampler.sampler_name
    for sampler in SAMPLERS
    if not any(keyword in sampler.sampler_name for keyword in EXCLUDE_KEYWORDS)
]
