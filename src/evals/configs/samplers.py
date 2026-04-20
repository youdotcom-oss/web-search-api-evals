import os

from youdotcom.models import ResearchEffort

from evals.samplers.applied_samplers.exa_sampler import ExaSampler
from evals.samplers.applied_samplers.google_sampler import GoogleSampler
from evals.samplers.applied_samplers.parallel_sampler import ParallelSearchSampler
from evals.samplers.applied_samplers.tavily_sampler import TavilySampler
from evals.samplers.applied_samplers.you_search_sampler import (
    YouLivecrawlSampler,
    YouResearchSampler,
    YouSearchSnippetsSampler,
)


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
    ExaSampler(
        sampler_name="exa_search_with_text",
        api_key=os.getenv("EXA_API_KEY"),
        text={"max_characters": 20000},
    ),
    GoogleSampler(
        sampler_name="google_search",
        api_key=os.getenv("SERP_API_KEY"),
    ),
    ParallelSearchSampler(
        sampler_name="parallel_search_one_shot",
        api_key=os.getenv("PARALLEL_API_KEY"),
        mode="one-shot",
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
]
