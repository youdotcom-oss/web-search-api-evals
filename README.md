# `web-search-api-evals`: An Evaluation Framework for Web Search APIs

This repository contains evaluation framework for AI-first web search APIs. Each API is integrated as a sampler and evaluated across benchmarks that test accuracy, latency, and information retrieval performance.

The framework supports multiple search providers (You.com, Exa, Tavily, Parallel) and a representative 
Google SERP–based sampler. For each query, search results are fetched from the search API, synthesized into an answer 
using an LLM, then graded against the ground truth.[^1] It also includes a dedicated [finance evaluation](#finance-evaluation)
suite and a [people search evaluation](#people-search-evaluation) for benchmarking people-data APIs (no gold answers;
deterministic field-fill scorers + LLM judges on structured `people[]` output).


To learn more about our evals methodology and system architecture, please read You.com's research articles:
- [How to Evaluate AI Search in the Agentic Era: A Sneak Peek](https://you.com/resources/sneak-peek-how-to-evaluate-ai-search-in-the-agentic-era)
- [How to Evaluate AI Search for the Agentic Era](https://you.com/resources/how-we-evaluate-ai-search)
- [Randomness in AI Benchmarks: What Makes an Eval Trustworthy?](https://you.com/resources/randomness-in-ai-benchmarks)

**We want to hear from you**. If you hit a configuration issue, have questions about your eval setup, want to request a 
benchmark, or just want to talk through how to evaluate search providers for your use case, start a conversation in 
GitHub Discussions. For enterprise or private inquiries, reach out directly at api@you.com. We read it.

## Results

Below are evaluation results across different search samplers and benchmark suites. Grading is performed via an LLM 
judge (GPT 5.4 mini) using prompts from the standard benchmarks (as specified in the original papers or repositories).[^2]
GPT 5.4 nano was used as the synthesis model.

**SimpleQA**

| sampler                   | accuracy | p50_latency_ms* |
|---------------------------|----------|-----------------|
| you_search_with_livecrawl |**92.09%**| 1048.05         |
| exa_search_with_text      | 90.06%   | 1176.05         |
| parallel_search_basic     | 89.78%   | 1901.66         |
| tavily_advanced           | 86.32%   | 3190.00         |
| you_search                | 84.81%   | 538.44          |
| google_search             | 80.17%   | 1347.48         |
| tavily_basic              | 59.11%   | 1340.00         |
* Internal latency as reported by the provider is used when available. When unavailable, the total time taken to complete 
the API request is used. 

**FRAMES**

| sampler                   | accuracy | p50_latency_ms |
|---------------------------|----------|----------------|
| you_research_lite         | 70.75%   | 3939.82        |
| tavily_advanced           | 39.93%   | 3460.00        |
| exa_search_with_text      | 39.81%   | 1351.75        |
| you_search_with_livecrawl | 37.26%   | 1153.78        |
| parallel_search_basic     | 34.83%   | 2118.61        |
| you_search                | 28.03%   | 565.80         |
| google_search             | 22.94%   | 1475.05        |
| tavily_basic              | 19.30%   | 2180.00        |


### Supported Benchmarks

| Benchmark    | Description                                                                                                                                                                                                                                                                                               | Flag / usage              |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------|
| SimpleQA     | Factual question answering ([OpenAI SimpleQA](https://openai.com/index/introducing-simpleqa/))                                                                                                                                                                                                            | `--datasets simpleqa`     |
| FRAMES       | Deep research and multi-hop reasoning ([paper](https://arxiv.org/abs/2409.12941), [dataset](https://huggingface.co/datasets/google/frames-benchmark))                                                                                                                                                     | `--datasets frames`       |
| DeepSearchQA | Challenging multi-step information seeking tasks. Only recommended for use with research endpoints ([paper](https://storage.googleapis.com/deepmind-media/DeepSearchQA/DeepSearchQA_benchmark_paper.pdf), [dataset](https://huggingface.co/datasets/google/deepsearchqa)) | `--datasets deepsearchqa` |
| BrowseComp   | A simple and challenging benchmark that measures the ability of AI agents to locate hard-to-find information. Only recommended for use with research endpoints ([paper](https://arxiv.org/abs/2504.12516), [dataset](https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv)) | `--datasets browsecomp`   |
| FinSearchComp T2 & T3 | Public-company financial lookup benchmarks from filings ([paper](https://arxiv.org/pdf/2509.13160)). T2 covers simple historical lookups; T3 covers complex historical investigations. Grading follows the paper's judge prompt; numbers in different formats (e.g. `12.45%` vs `0.1245`) are treated as equivalent. | `--datasets fin_search_comp_t2_global fin_search_comp_t3_global` |
| People Search | 240 people enrichment / search queries across 6 buyer personas. **No gold answers.** Score structured `people[]` from any HTTP people-search API via `http_people_search` (deterministic field-fill + LLM judges on by default). See [People Search evaluation](#people-search-evaluation). | `--datasets people_search --samplers http_people_search` |


## Installation
Requires Python versions >=3.10 and <3.14 

```bash
# Clone the repository
git clone https://github.com/youdotcom-oss/web-search-api-evals.git
cd web-search-api-evals

# Create a virtual environment, then install
pip install -r requirements.txt
pip install -e .
```

### API keys

Copy the example env file and set the appropriate API keys for the samplers you want to run:

```bash
cp .env.example .env
```

Edit `.env` and set the keys for your chosen providers. To run evaluations for a given search API, set the corresponding environment variable to a valid API key, then pass the sampler name via `--samplers`:

| Sampler                     | Environment variable   |
|-----------------------------|-------------------------|
| Exa                         | `EXA_API_KEY`           |
| Google                      | `SERP_API_KEY`           |
| Parallel                    | `PARALLEL_API_KEY`      |
| Perplexity                  | `PERPLEXITY_API_KEY`    |
| Tavily                      | `TAVILY_API_KEY`        |
| You.com                     | `YOU_API_KEY`           |
| People search (generic HTTP)| `PEOPLE_SEARCH_API_URL` (+ optional `PEOPLE_SEARCH_API_KEY`) |

Grading uses OpenAI models by default, but Gemini models are also supported. Set `OPENAI_API_KEY` or 
`GOOGLE_GEMINI_API_KEY` as appropriate for the LLM judge.

## Usage

### Basic instructions

Run evaluations from the command line via the eval runner:

```bash
# List available samplers and datasets
python src/evals/eval_runner.py --help

# Run SimpleQA and FRAMES on default samplers (does not include You.com Research endpoints)
python src/evals/eval_runner.py

# Run SimpleQA for specific samplers only
python src/evals/eval_runner.py --samplers you_search_with_livecrawl tavily_basic --datasets simpleqa

# Run FRAMES evaluation
python src/evals/eval_runner.py --datasets frames

# Run on a limited number of problems (e.g. 100 for a quick sanity check)
python src/evals/eval_runner.py --samplers you_search_with_livecrawl --datasets simpleqa --limit 100

# Fresh run: clear existing results and re-run
python src/evals/eval_runner.py --clean True --samplers you_search_with_livecrawl --datasets simpleqa --limit 100
```

#### Important Notes
- To avoid unintended high credit usage, You.com's Research endpoints are not included in the default samplers. They can 
be evaluated by calling them explicitly, like `--samplers you_research_standard` or by using `--samplers all`.
- The BrowseComp and Deep Search QA Datasets are not included in the default benchmark dataset list because they are 
intended to evaluate Research endpoints.
- `people_search` / `http_people_search` are also excluded from defaults. Run them explicitly (see [People Search evaluation](#people-search-evaluation)).

### LLM's for synthesis and judging
By default, GPT 5.4 nano is used for synthesis and GPT 5.4 mini via the OpenAI API is used for grading. 
This codebase also supports Gemini models via the Google `genai` library. To use an alternative OpenAI model or a 
Gemini model, simply update the model name in `src/evals/constants.py`. The code will interpret whether you are using a GPT
or Gemini model and route your request appropriately.

### Other configuration options

| Option               | Flag / default              | Description                                                        |
|----------------------|-----------------------------|--------------------------------------------------------------------|
| Samplers             | `--samplers <names>`        | One or more sampler names (default: All except You.com Research).  |
| Datasets             | `--datasets <names>`        | One or more datasets (default: `simpleqa`, `frames`).              |
| Limit                | `--limit <n>`               | Run on at most `n` problems (optional).                            |
| Batch size           | `--batch-size 50`           | Number of problems per batch before writing results (default: 50). |
| Max concurrent tasks | `--max-concurrent-tasks 10` | Concurrency limit (default: 10).                                   |
| Clean                | `--clean True`              | Wipe results and run from scratch (pass the string `True`; default is no clean). |


## Finance evaluation

To learn more about You.com's Finance Research API, read our [blog post](https://you.com/resources/introducing-the-finance-research-api-agentic-research-no-infra-required).

The `fin_search_comp_t2_global` dataset evaluates simple historical lookup of public-company financials (e.g. *"What were Uber's research and development expenses for the full year 2019?"*). Ground truth comes from SEC filings and grading follows the prompt from the FinSearchComp paper[^3], which treats numerically equivalent answers (`12.45%` vs `0.1245`, `120,400,000` vs `120.4 million`) as the same and ignores unit-only differences. The grader model is configurable independently of the default `GRADER_MODEL` via `FIN_SEARCH_GRADER_MODEL` in `src/evals/constants.py`.

### Samplers evaluated against this benchmark

| Sampler                                   | Provider   |
|-------------------------------------------|------------|
| `you_finance_research_deep`               | You.com    |
| `you_finance_research_exhaustive`         | You.com    |
| `perplexity_finance_historical_lookup`    | Perplexity |
| `perplexity_finance_multi_step_research`  | Perplexity |
| `perplexity_sonar_deep_research_high`     | Perplexity |
| `exa_research_pro`                        | Exa        |
| `tavily_research_pro`                     | Tavily     |
| `parallel_pro`                            | Parallel   |
| `parallel_ultra`                          | Parallel   |

### Running the benchmark

```bash
# Quick sanity check on a single sampler
python src/evals/eval_runner.py \
  --samplers you_finance_research_deep \
  --datasets fin_search_comp_t2_global \
  --limit 10

# Full sweep across all finance-capable samplers
python src/evals/eval_runner.py \
  --samplers you_finance_research_deep you_finance_research_exhaustive tavily_research_pro \
  --datasets fin_search_comp_t2_global
```

### Results

**FinSearchComp T2 — Simple historical lookup (global)**

| sampler                                  | accuracy   | p50_latency_ms* |
|------------------------------------------|------------|-----------------|
| you_finance_research_deep                | **87.29%** | 124.0           |
| parallel_ultra                           | 73.11%     | 861.3           |
| perplexity_finance_historical_lookup     | 72.27%     | 32.2            |
| perplexity_sonar_deep_research_high      | 53.78%     | 92.6            |
| exa_research_pro                         | 42.02%     | 366.8           |
| tavily_research_pro                      | 40.34%     | 104.5           |
| parallel_pro                             | 34.45%     | 317.0           |

* Internal latency as reported by the provider is used when available. When unavailable, the total time taken to complete the API request is used.

## People Search evaluation

The `people_search` benchmark evaluates **people-data / people-search APIs** (named-person enrichment and open
candidate search). It does **not** use the web-search path of snippets → LLM synthesis → gold-answer grading.

There is **no gold answer** per row. Your API returns structured `people[]`; the shared `eval_runner.py` scores that
payload with:

1. **Deterministic scorers** (always on) — retrieval + field richness  
2. **LLM judges** (on by default; 2 calls per row) — overall quality + persona-specific quality  

This repo does **not** ship people-search API clients. Point `http_people_search` at **any** people-search HTTP
endpoint that speaks the contract below (or a thin adapter in front of your existing API).

```bash
python src/evals/eval_runner.py \
  --samplers http_people_search \
  --datasets people_search \
  --limit 5
```

### Dataset

| | |
|--|--|
| File | [`data/people_search_full_dataset.csv`](data/people_search_full_dataset.csv) |
| Size | 240 queries |
| Split | 90 `enrichment` · 150 `search` |
| Personas (40 each) | Recruiter / Talent Sourcer · SDR / BDR · Background Check / Compliance Analyst · Journalist / Investigative Researcher · Event Organizer / Community Manager · VC / PE / Investor |

CSV columns: `benchmark_id`, `problem`, `answer`, `persona`, `persona_slug`, `query_type`, `person_name`, `company`.

`problem` is the natural-language query. `answer` is **always empty** in the shipped CSV (no gold / sample answers).
Scoring context (`persona`, `persona_slug`, `query_type`, `person_name`, `company`, `benchmark_id`) lives in those
dedicated columns; at load time the runner assembles them into metadata for `http_people_search` and the graders.

### Pipeline

```
CSV row (problem + persona / query_type / … columns; answer empty)
        │
        ▼
http_people_search  ──POST──►  YOUR_PEOPLE_API  ──►  { people[], person_count }
        │                                              (JSON string becomes generated_answer)
        ▼
deterministic scorers  +  LLM judges (unless PEOPLE_SEARCH_LLM_JUDGES=0)
        │
        ▼
src/evals/results/dataset_people_search_raw_results_http_people_search.csv
analyzed_results.csv  (mean_field_fill / mean_judge_* / has_people_rate; accuracy_score blank)
```

No LLM synthesis step (`needs_synthesis=False`). Scoring operates on the structured people payload only.

### Environment

| Variable | Required | Purpose |
|----------|----------|---------|
| `PEOPLE_SEARCH_API_URL` | Yes | URL of your people-search HTTP endpoint |
| `PEOPLE_SEARCH_API_KEY` | No | If set (and not equal to the URL), sent as `Authorization: Bearer …` |
| `OPENAI_API_KEY` or `GOOGLE_GEMINI_API_KEY` | For LLM judges | Uses `GRADER_MODEL` (`gpt-5.4-mini` by default) |
| `PEOPLE_SEARCH_LLM_JUDGES` | No (default `1`) | Set to `0` / `false` / `off` for deterministic scorers only |

### HTTP endpoint contract

`http_people_search` sends:

```http
POST $PEOPLE_SEARCH_API_URL
Content-Type: application/json
Accept: application/json
Authorization: Bearer $PEOPLE_SEARCH_API_KEY   # only if PEOPLE_SEARCH_API_KEY is set
```

**Request body**

```json
{
  "query": "Find work history and current role for William McKinnerney, who works at CoreWeave.",
  "metadata": {
    "benchmark_id": "fp_001",
    "persona": "Recruiter / Talent Sourcer",
    "persona_slug": "recruiter",
    "query_type": "enrichment",
    "person_name": "William McKinnerney",
    "company": "CoreWeave",
    "query_text": "Find work history and current role for William McKinnerney, who works at CoreWeave."
  }
}
```

`metadata` is built from the CSV’s dedicated columns (`persona`, `persona_slug`, `query_type`, `person_name`,
`company`, `benchmark_id`); the sampler also sets `query_text` to the query string when missing.
`query_type` is `enrichment` (named person ± company) or `search` (open candidate list). **Your API** decides how to
route (enrich vs search, etc.) from that metadata — the eval runner does not call provider-specific endpoints.

**Response body (canonical)**

```json
{
  "people": [
    {
      "displayname": "William McKinnerney",
      "current_title": "...",
      "current_company": "CoreWeave",
      "location": "...",
      "linkedin_url": "https://...",
      "highlight": "...",
      "best_work_email": "...",
      "phones": ["..."],
      "top_skills": ["..."],
      "insights": {},
      "confidence": {"likelihood": 0.9}
    }
  ],
  "person_count": 1,
  "error": null
}
```

On failure, return `"error": "<message>"` (typically with empty `people` and `person_count: 0`). HTTP status ≥400 or
non-JSON bodies are turned into an `error` payload by the sampler.

**Accepted response variants** (normalized in `src/evals/processing/people_search/schema.py`): top-level `people`,
`summary.people` / `summary.person`, or `results` as a people list.

Person field aliases scorers understand (examples): `headline` → title; `linkedin_url` / `url` → profile URL;
`best_work_email` / `best_personal_email` / `has_email` → email; `phones` / `has_phone` → phone; `top_skills` → skills.

Contract source: [`src/evals/samplers/applied_samplers/people_search_sampler.py`](src/evals/samplers/applied_samplers/people_search_sampler.py).

### Scorers

#### Deterministic (always on)

Implemented in `src/evals/processing/people_search/field_fill.py`.

| Metric | Range | Meaning |
|--------|-------|---------|
| `has_people` | 0 or 1 | At least one person (`person_count` > 0). Raw `evaluation_result` is `has_people` or `no_people`. |
| `field_fill` | 0–1 | Mean fill ratio across 11 fields: displayname, current_title, current_company, location, profile_url, highlight, email, phone, skills, insights, confidence |
| `persona_field_fill` | 0–1 | Same fields, weighted by `persona_slug` (`recruiter`, `sdr`, `compliance`, `journalist`, `events`, `investor`) |

Hard rule: if the payload has an `error` key, deterministic scores are **0**. Empty `people` also yields fill scores of **0**.

**Primary quality metrics** for comparing providers: `mean_field_fill`, `mean_persona_field_fill`, and LLM judges below.  
`has_people` / `has_people_rate` is retrieval-only. `accuracy_score` in `analyzed_results.csv` is left **blank** for
`people_search` so it is not confused with gold-answer accuracy on SimpleQA/FRAMES.

#### LLM judges (on by default)

Prompts: [`src/evals/processing/people_search/prompts/overall.md`](src/evals/processing/people_search/prompts/overall.md)
and [`persona.md`](src/evals/processing/people_search/prompts/persona.md).  
Wired in `evaluate_single_people_search` via `src/evals/processing/people_search/llm_judges.py`.  
Model: `GRADER_MODEL` in [`src/evals/constants.py`](src/evals/constants.py) (default `gpt-5.4-mini`).

When enabled, each row runs **two** LLM calls (overall + persona). Labels map to scores:

| Label | Score |
|-------|-------|
| High Value | 1.0 |
| Useful | 0.7 |
| Low Value | 0.3 |
| Failed | 0.0 |

| Metric | Meaning |
|--------|---------|
| `judge_overall` | Cross-persona quality / actionability |
| `judge_persona` | Persona-switched rubric for the row’s `persona_slug` |

Also written per row: `judge_overall_label`, `judge_persona_label`, `judge_persona_slug`.

Disable LLM judges (deterministic only — no OpenAI/Gemini spend for this dataset):

```bash
PEOPLE_SEARCH_LLM_JUDGES=0 python src/evals/eval_runner.py \
  --samplers http_people_search \
  --datasets people_search \
  --limit 5
```

### Running

```bash
cp .env.example .env
# Set PEOPLE_SEARCH_API_URL (+ optional PEOPLE_SEARCH_API_KEY)
# Set OPENAI_API_KEY (or GOOGLE_GEMINI_API_KEY) unless PEOPLE_SEARCH_LLM_JUDGES=0

# Smoke test
python src/evals/eval_runner.py \
  --samplers http_people_search \
  --datasets people_search \
  --limit 5

# Full benchmark (wipe prior people_search results for this sampler)
python src/evals/eval_runner.py \
  --samplers http_people_search \
  --datasets people_search \
  --clean True
```

`http_people_search` is excluded from the default sampler list so it is not run against SimpleQA/FRAMES by accident.

### Results for people_search

Raw CSV includes the usual runner fields (`query`, latencies, `generated_answer`, `ground_truth`, `evaluation_result`)
plus scorer columns:

`has_people`, `person_count`, `field_fill`, `persona_field_fill`, `judge_overall`, `judge_overall_label`,
`judge_persona`, `judge_persona_label`, `judge_persona_slug` (judge columns omitted when LLM judges are disabled).

`generated_answer` is the JSON people payload string. `ground_truth` is scoring metadata assembled at load time from
the dedicated CSV columns (the shipped `answer` column is empty). `evaluation_result` is `has_people` or `no_people`
(not `is_correct` / `is_incorrect`).

`analyzed_results.csv` for this dataset:

| Column | Meaning |
|--------|---------|
| `mean_field_fill` | Primary deterministic quality (0–1) |
| `mean_persona_field_fill` | Persona-weighted field fill (0–1) |
| `mean_judge_overall` / `mean_judge_persona` | Mean LLM judge scores when present |
| `has_people_rate` | Fraction of rows with ≥1 person (retrieval only) |
| `accuracy_score` | **Blank** (N/A — not gold-answer accuracy) |
| `p50_*_latency` / `problem_count` | Same as other datasets |

Within `people_search`, rows are ordered by `mean_field_fill` (if missing, falls back to `mean_judge_overall`, then
`has_people_rate`).

### Key source files

| Path | Role |
|------|------|
| `data/people_search_full_dataset.csv` | Benchmark queries + scoring metadata |
| `src/evals/configs/datasets.py` | Registers `people_search` (`requires_ground_truth=False`) |
| `src/evals/configs/samplers.py` | Registers `http_people_search` (excluded from defaults) |
| `src/evals/samplers/applied_samplers/people_search_sampler.py` | Generic HTTP sampler + request/response contract |
| `src/evals/processing/people_search/` | Schema normalize, field-fill, previews, LLM judges, prompts |
| `src/evals/processing/evaluate_answer.py` | `evaluate_single_people_search` |
| `src/evals/eval_results_analyzer.py` | People-specific analyzed metrics (blank accuracy) |
| `tests/test_people_search.py` | Unit tests (scorers, label parse, metrics; judges off) |

## Output

Results are written to `src/evals/results/` with the following structure:

```
src/evals/results/
├── dataset_<dataset_name>_raw_results_<sampler_name>.csv   # Per-sampler, per-dataset raw results
└── analyzed_results.csv                               # Aggregated metrics (accuracy, latency) updated after each run
```

Raw CSVs contain per-query fields (e.g. query, generated answer, evaluation result, latencies). After a run, 
`write_metrics()` is called automatically and `analyzed_results.csv` is updated. For gold-answer datasets that is
accuracy and latency; for `people_search` it is field-fill / judge means and `has_people_rate` (see above).

## Citation

If you use this repository in your research, please consider citing:

```bibtex
@misc{2026yousearchevals,
  title        = {web-search-api-evals: An Evaluation Framework for AI-first Web Search APIs},
  author       = {You.com},
  year         = {2026},
  journal      = {GitHub repository},
	publisher    = {GitHub},
  howpublished = {\url{https://github.com/youdotcom-oss/web-search-api-evals}}
}
```

## License

This repository is made available under the [MIT License](LICENSE).


[^1]: For web-search benchmarks, search results are fetched from each search API, then synthesized into a single answer using an LLM; the answer is graded by an LLM judge. Synthesis uses GPT 5.4 nano and grading uses GPT 5.4 mini (configurable in `src/evals/constants.py`). People Search skips synthesis and scores structured `people[]` instead.
[^2]: Grading uses prompts aligned with the standard benchmarks as specified in the original papers or repositories (e.g. [SimpleQA](https://openai.com/index/introducing-simpleqa/) and [FRAMES](https://arxiv.org/abs/2409.12941).
[^3]: FinSearchComp grading uses the judge prompt from the [FinSearchComp paper](https://arxiv.org/pdf/2509.13160).
