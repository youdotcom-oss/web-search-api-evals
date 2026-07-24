"""LLM judges for people-search (overall + persona rubrics)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from evals import constants
from evals.processing import llm
from evals.processing.people_search.field_fill import extract_people
from evals.processing.people_search.people_preview import (
    format_people_for_scorer,
    judge_persona_from_metadata,
    named_target_from_metadata,
)

CHOICE_SCORES = {
    "high value": 1.0,
    "useful": 0.7,
    "low value": 0.3,
    "failed": 0.0,
}

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_LABEL_RE = re.compile(
    r"LABEL:\s*(High Value|Useful|Low Value|Failed)",
    re.IGNORECASE,
)


def llm_judges_enabled() -> bool:
    """LLM judges run by default; set PEOPLE_SEARCH_LLM_JUDGES=0 to skip."""
    return os.getenv("PEOPLE_SEARCH_LLM_JUDGES", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _parse_label(response: str) -> tuple[str | None, float | None]:
    pretty_map = {
        "high value": "High Value",
        "useful": "Useful",
        "low value": "Low Value",
        "failed": "Failed",
    }
    match = _LABEL_RE.search(response or "")
    if match:
        key = match.group(1).lower()
        return pretty_map[key], CHOICE_SCORES[key]

    lowered = (response or "").lower()
    for key, pretty in pretty_map.items():
        if key in lowered:
            return pretty, CHOICE_SCORES[key]
    return None, None


def _prompt_vars(query: str, output: dict, metadata: dict) -> dict[str, Any]:
    people = extract_people(output)
    person_count = output.get("person_count")
    if person_count is None:
        person_count = len(people)
    return {
        "provider": output.get("provider") or "unknown",
        "query": query,
        "persona": metadata.get("persona") or "",
        "judge_persona": judge_persona_from_metadata(metadata),
        "query_type": metadata.get("query_type") or "",
        "named_target": named_target_from_metadata(metadata) or "(none)",
        "error": output.get("error") or "(none)",
        "person_count": person_count,
        "people_preview": format_people_for_scorer(people),
    }


async def run_people_llm_judges(
    query: str,
    output: dict,
    metadata: dict | None = None,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """Run overall + persona LLM judges; return score fields for the results CSV."""
    if not llm_judges_enabled():
        return {}

    meta = metadata or {}
    vars_ = _prompt_vars(query, output, meta)
    judge_model = model or constants.GRADER_MODEL
    system = (
        "You are a careful evaluator of people-search API outputs. "
        "Follow the rubric exactly and always end with a LABEL line."
    )

    overall_prompt = _load_prompt("overall.md").format(**vars_)
    persona_prompt = _load_prompt("persona.md").format(**vars_)

    overall_raw = await llm.call_llm(judge_model, system, overall_prompt)
    persona_raw = await llm.call_llm(judge_model, system, persona_prompt)

    overall_label, overall_score = _parse_label(overall_raw)
    persona_label, persona_score = _parse_label(persona_raw)

    return {
        "judge_overall_label": overall_label,
        "judge_overall": overall_score,
        "judge_persona_label": persona_label,
        "judge_persona": persona_score,
        "judge_persona_slug": vars_["judge_persona"],
    }
