"""People preview formatting for LLM judges."""

from __future__ import annotations

from evals.processing.people_search.constants import PERSONA_SLUGS


def named_target_from_metadata(metadata: dict | None) -> str:
    meta = metadata or {}
    name = (meta.get("person_name") or "").strip()
    company = (meta.get("company") or "").strip()
    if name and company:
        return f"{name} at {company}"
    return name


def judge_persona_from_metadata(metadata: dict | None) -> str:
    meta = metadata or {}
    override = meta.get("judge_persona") or meta.get("persona_slug")
    if override:
        return str(override).strip().lower()
    persona = meta.get("persona") or ""
    return PERSONA_SLUGS.get(
        persona, persona.strip().lower().replace(" ", "_") or "unknown"
    )


def format_people_for_scorer(people: list | None, max_people: int = 5) -> str:
    if not people:
        return "(zero results returned)"
    blocks: list[str] = []
    for index, person in enumerate(people[:max_people], start=1):
        if not isinstance(person, dict):
            continue
        name = person.get("displayname") or "?"
        title = person.get("current_title") or person.get("headline") or ""
        company = person.get("current_company") or ""
        location = person.get("location") or ""
        lines = [f"{index}. {name}"]
        if title or company:
            header = title + (f" @ {company}" if company else "")
            lines.append(f"   Title: {header[:220]}")
        if location:
            lines.append(f"   Location: {location[:120]}")
        skills = person.get("top_skills") or []
        if skills:
            lines.append(f"   Skills: {', '.join(str(s) for s in skills[:5])}")
        insights = person.get("insights") or {}
        if isinstance(insights, dict):
            if insights.get("overall_summary"):
                lines.append(f"   Summary: {str(insights['overall_summary'])[:200]}")
            for chip in (insights.get("why_matched") or [])[:3]:
                if not isinstance(chip, dict):
                    continue
                crit = chip.get("criterion", "?")
                phrase = chip.get("matched_phrase") or chip.get("display_text") or ""
                lines.append(f"   Match: {crit} — {phrase[:120]}")
        if person.get("highlight"):
            lines.append(f"   Highlight: {person['highlight'][:280]}")
        if person.get("best_work_email"):
            lines.append(f"   Work email: {person['best_work_email']}")
        if person.get("best_personal_email"):
            lines.append(f"   Personal email: {person['best_personal_email']}")
        phones = person.get("phones") or []
        if phones:
            lines.append(f"   Phones: {', '.join(str(p) for p in phones[:2])}")
        url = person.get("linkedin_url") or person.get("url")
        if url:
            lines.append(f"   URL: {url}")
        confidence = person.get("confidence") or {}
        if isinstance(confidence, dict) and confidence.get("likelihood") is not None:
            lines.append(f"   Match likelihood: {confidence['likelihood']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "(zero results returned)"
