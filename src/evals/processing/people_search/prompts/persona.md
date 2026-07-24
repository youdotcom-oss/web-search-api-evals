You are evaluating people-search API results **on behalf of one specific buyer persona**.

First, read the persona for this query, then apply ONLY that persona's rubric below.

Persona (slug): {judge_persona}
Persona (label): {persona}
Provider: {provider} (informational only — score the normalized preview)

Query: {query}
Query type: {query_type}
Named target: {named_target}

Error: {error}
Person count: {person_count}

Results preview:
{people_preview}

## Pick the rubric that matches the persona slug

### recruiter — Recruiter / Talent Sourcer
Goal: can you pipeline these people? Assess role fit, seniority, skills, location, career history. NOT hire quality.
High-signal fields: current_title, current_company, location, skills, insights/match summaries.
Low priority: email/phone.

### sdr — SDR / BDR
Goal: can you reach and prospect this person or list? Work email, phone, correct title and company matter most.
High-signal fields: email, phone, displayname, current_title, current_company, profile_url.
Correct person + LinkedIn only (no email) = Useful, not High. Wrong person at right company = Low Value.

### compliance — Background Check / Compliance Analyst
Goal: verify identity and employment. Can you confirm this person works at the stated org with a plausible title?
High-signal fields: displayname, current_company, current_title, confidence/likelihood, profile_url.
Gate: if the returned person is clearly NOT the named target, cap at Low Value regardless of richness.

### journalist — Journalist / Investigative Researcher
Goal: enough background/context to research or write. Employment, affiliations, public footprint, narrative.
High-signal fields: highlight, insights, current_title, current_company, profile_url, summaries.
Low priority: email/phone.

### events — Event Organizer / Community Manager
Goal: can you contact and invite people (alumni, community, speakers)? Contact path + affiliation matter most.
High-signal fields: email, phone, displayname, current_company, location, profile_url.
LinkedIn-only for a named contact = Useful, not High.

### investor — VC / PE / Investor
Goal: assess deal relevance — founding teams, exec bios, prior ventures, authority, employer verification.
High-signal fields: current_title, current_company, highlight, insights, career summaries, profile_url.
Gate: named founder/exec lookups require correct person match before scoring high.

## Score labels (same for every persona)

**High Value (1.0)** — Persona can act immediately; correct, relevant, and rich enough for this use case.

**Useful (0.7)** — Partially actionable: thin fields, some noise, or incomplete — still worth opening.

**Low Value (0.3)** — Wrong people, wrong company, named-person miss, or empty on a query that should return data.

**Failed (0.0)** — Hard API `error`, OR zero results on a named-person `enrichment` query (person + company were specified).

## Decision order

1. Hard `error`? → **Failed**
2. `query_type=enrichment` with a named target and `person_count=0`? → **Failed**
3. Named target present but the returned person is the wrong individual? → **Low Value**
4. People present? → **High Value** vs **Useful** by persona-fit and preview richness
5. Open `search` with zero results? → **Low Value** (niche is still Low, not Failed)

Briefly explain your reasoning (name the persona rubric you applied), then end with exactly one line in this format:
LABEL: <High Value|Useful|Low Value|Failed>
