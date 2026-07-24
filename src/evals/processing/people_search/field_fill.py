"""Deterministic people[] richness scorers (no gold answers)."""

from __future__ import annotations

from typing import Any

from evals.processing.people_search.constants import (
    PERSONA_FIELD_WEIGHTS,
    TRACKED_FIELDS,
)

MAX_FIELDS = len(TRACKED_FIELDS)
DEFAULT_WEIGHTS = {field: 1.0 for field in TRACKED_FIELDS}


def _is_filled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        s = value.strip()
        return bool(s) and s not in ("?", "—", "-", "null", "None")
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    if isinstance(value, bool):
        return value
    return True


def _person_field_values(person: dict) -> dict[str, object]:
    skills = person.get("top_skills") or []
    insights = person.get("insights") or {}
    confidence = person.get("confidence")
    if confidence is None and isinstance(person.get("likelihood"), (int, float)):
        confidence = {"likelihood": person["likelihood"]}

    email = person.get("best_work_email") or person.get("best_personal_email")
    if not email and person.get("has_email"):
        email = "present"
    if not email and person.get("altemails"):
        email = person["altemails"][0]

    phones = person.get("phones") or []
    phone = phones[0] if phones else None
    if not phone and person.get("has_phone"):
        phone = "present"

    url = person.get("linkedin_url") or person.get("url")

    return {
        "displayname": person.get("displayname"),
        "current_title": person.get("current_title") or person.get("headline"),
        "current_company": person.get("current_company"),
        "location": person.get("location"),
        "profile_url": url,
        "highlight": person.get("highlight"),
        "email": email,
        "phone": phone,
        "skills": skills if isinstance(skills, list) and skills else None,
        "insights": insights if isinstance(insights, dict) and insights else None,
        "confidence": confidence,
    }


def person_fill_ratio(person: dict) -> tuple[float, int]:
    values = _person_field_values(person)
    filled = sum(1 for key in TRACKED_FIELDS if _is_filled(values.get(key)))
    return filled / MAX_FIELDS, filled


def persona_weighted_fill_ratio(person: dict, persona: str) -> float:
    weights = PERSONA_FIELD_WEIGHTS.get(persona, DEFAULT_WEIGHTS)
    values = _person_field_values(person)
    total_weight = sum(weights.get(f, 1.0) for f in TRACKED_FIELDS)
    earned = sum(
        weights.get(f, 1.0) for f in TRACKED_FIELDS if _is_filled(values.get(f))
    )
    return earned / total_weight if total_weight else 0.0


def extract_people(output: dict) -> list:
    people = output.get("people") or []
    if not people and isinstance(output.get("summary"), dict):
        summary = output["summary"]
        people = summary.get("people") or []
        if summary.get("person") and not people:
            people = [summary["person"]]
    return people if isinstance(people, list) else []


def row_fill_score(people: list | None, *, max_people: int = 5) -> dict:
    if not people:
        return {
            "score": 0.0,
            "avg_ratio": 0.0,
            "avg_fields_per_person": 0.0,
            "people_scored": 0,
            "total_filled": 0,
        }

    ratios: list[float] = []
    counts: list[int] = []
    for person in people[:max_people]:
        if not isinstance(person, dict):
            continue
        ratio, count = person_fill_ratio(person)
        ratios.append(ratio)
        counts.append(count)

    if not ratios:
        return {
            "score": 0.0,
            "avg_ratio": 0.0,
            "avg_fields_per_person": 0.0,
            "people_scored": 0,
            "total_filled": 0,
        }

    avg_ratio = sum(ratios) / len(ratios)
    return {
        "score": round(avg_ratio, 4),
        "avg_ratio": round(avg_ratio, 4),
        "avg_fields_per_person": round(sum(counts) / len(counts), 2),
        "people_scored": len(ratios),
        "total_filled": sum(counts),
    }


def row_persona_fill_score(
    people: list | None, persona: str, *, max_people: int = 5
) -> dict:
    if not people:
        return {"score": 0.0, "persona": persona, "people_scored": 0}

    ratios = [
        persona_weighted_fill_ratio(p, persona)
        for p in people[:max_people]
        if isinstance(p, dict)
    ]
    if not ratios:
        return {"score": 0.0, "persona": persona, "people_scored": 0}

    avg = sum(ratios) / len(ratios)
    return {
        "score": round(avg, 4),
        "persona": persona,
        "people_scored": len(ratios),
    }


def score_people_output(output: dict, metadata: dict | None = None) -> dict:
    """Run deterministic scorers on a provider output payload."""
    meta = metadata or {}
    if not isinstance(output, dict) or output.get("error"):
        return {
            "has_people": 0.0,
            "person_count": 0,
            "field_fill": 0.0,
            "persona_field_fill": 0.0,
            "persona": meta.get("persona_slug") or "unknown",
        }

    people = extract_people(output)
    person_count = int(output.get("person_count") or len(people) or 0)
    fill = row_fill_score(people)
    persona = (
        str(meta.get("persona_slug") or meta.get("judge_persona") or "unknown")
        .strip()
        .lower()
    )
    persona_fill = row_persona_fill_score(people, persona)

    return {
        "has_people": 1.0 if person_count > 0 else 0.0,
        "person_count": person_count,
        "field_fill": fill["score"],
        "persona_field_fill": persona_fill["score"],
        "persona": persona,
        "people_scored": fill["people_scored"],
        "avg_fields_per_person": fill.get("avg_fields_per_person", 0.0),
    }
