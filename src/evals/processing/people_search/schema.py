"""Normalize arbitrary people-search API responses into the scorer payload."""

from __future__ import annotations

from typing import Any


def normalize_people_payload(result: Any, provider: str = "http_people_search") -> dict:
    """Coerce endpoint JSON into ``{people, person_count, error, provider}``.

    Accepts either the canonical shape or common variants:
    - top-level ``people`` list
    - ``summary.people`` / ``summary.person``
    - ``results`` as a people list
    """
    if not isinstance(result, dict):
        return {
            "provider": provider,
            "people": [],
            "person_count": 0,
            "error": "invalid provider result",
        }

    if result.get("error"):
        people = result.get("people") if isinstance(result.get("people"), list) else []
        return {
            "provider": provider,
            "people": people,
            "person_count": int(result.get("person_count") or len(people) or 0),
            "error": result.get("error"),
        }

    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    people = result.get("people")
    if not isinstance(people, list):
        people = summary.get("people") if isinstance(summary.get("people"), list) else None
    if not people and isinstance(result.get("results"), list):
        people = result["results"]
    if not people and summary.get("person"):
        people = [summary["person"]]
    if not isinstance(people, list):
        people = []

    person_count = result.get("person_count")
    if person_count is None:
        person_count = summary.get("person_count")
    if person_count is None:
        person_count = len(people)

    return {
        "provider": provider,
        "people": people,
        "person_count": int(person_count or 0),
        "error": None,
    }
