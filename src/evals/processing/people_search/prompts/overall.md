You are evaluating people-search API results for overall data quality and actionability.

Judge whether a buyer could use this output to act on the query — regardless of which persona asked. Focus on relevance, correctness, richness, and whether zero results are acceptable.

Provider: {provider} (informational only — score the normalized preview)

Query: {query}
Persona (context): {persona}
Query type: {query_type}
Named target: {named_target}

Error: {error}
Person count: {person_count}

Results preview:
{people_preview}

## What “good” means (overall)

- **Relevant people** returned for the query intent
- **Structured fields** present: name, title, company, location, profile URL, contact info when appropriate
- **Named-person enrichment** (`query_type=enrichment`): the returned person should match `person_name` at `company`
- **Open search** (`query_type=search`): a useful list of on-topic candidates, not random profiles

## Score labels

**High Value (1.0)** — Correct, relevant results with enough structure to act on immediately.

**Useful (0.7)** — Partially actionable: thin fields, some noise, or incomplete contact info but correct direction.

**Low Value (0.3)** — Wrong people, irrelevant list, named-person miss, or empty on a query that should return data.

**Failed (0.0)** — Hard API `error`, OR zero results on a named-person enrichment where the person and company are specified.

## Decision order

1. Hard `error`? → **Failed**
2. `query_type=enrichment` with `person_name` set and `person_count=0`? → **Failed**
3. Named target present but wrong person in preview? → **Low Value**
4. People present with good relevance + structure? → **High Value** vs **Useful** by richness
5. Open search with zero results? → **Low Value** (niche queries may still be low, not failed)

Briefly explain your reasoning, then end with exactly one line in this format:
LABEL: <High Value|Useful|Low Value|Failed>
