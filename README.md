# `web-search-api-evals`: An Evaluation Framework for Web Search APIs

This repository contains evaluation framework for AI-first web search APIs. Each API is integrated as a sampler and evaluated across benchmarks that test accuracy, latency, and information retrieval performance.

The framework supports multiple search providers (You.com, Exa, Tavily, Parallel) and a representative 
Google SERP–based sampler. For each query, search results are fetched from the search API, synthesized into an answer 
using an LLM, then graded against the ground truth.[^1] It also includes a dedicated [finance evaluation](#finance-evaluation) suite for benchmarking financial data retrieval.


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
| you_search                | 84.81%   | 538.44          |
| google_search             | 80.17%   | 1347.48         |
| tavily_search              | 59.11%   | 1340.00         |
* Internal latency as reported by the provider is used when available. When unavailable, the total time taken to complete 
the API request is used. 

**FRAMES**

| sampler                   | accuracy | p50_latency_ms |
|---------------------------|----------|----------------|
| you_research_lite         | 70.75%   | 3939.82        |
| exa_search_with_text      | 39.81%   | 1351.75        |
| you_search_with_livecrawl | 37.26%   | 1153.78        |
| parallel_search_basic     | 34.83%   | 2118.61        |
| you_search                | 28.03%   | 565.80         |
| google_search             | 22.94%   | 1475.05        |
| tavily_search              | 19.30%   | 2180.00        |


### Supported Benchmarks

| Benchmark    | Description                                                                                                                                                                                                                                                                                               | Flag / usage              |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------|
| SimpleQA     | Factual question answering ([OpenAI SimpleQA](https://openai.com/index/introducing-simpleqa/))                                                                                                                                                                                                            | `--datasets simpleqa`     |
| FRAMES       | Deep research and multi-hop reasoning ([paper](https://arxiv.org/abs/2409.12941), [dataset](https://huggingface.co/datasets/google/frames-benchmark))                                                                                                                                                     | `--datasets frames`       |
| DeepSearchQA | Challenging multi-step information seeking tasks. Only recommended for use with research endpoints ([paper](https://storage.googleapis.com/deepmind-media/DeepSearchQA/DeepSearchQA_benchmark_paper.pdf), [dataset](https://huggingface.co/datasets/google/deepsearchqa)) | `--datasets deepsearchqa` |
| BrowseComp   | A simple and challenging benchmark that measures the ability of AI agents to locate hard-to-find information. Only recommended for use with research endpoints ([paper](https://arxiv.org/abs/2504.12516), [dataset](https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv)) | `--datasets browsecomp`   |
| FinSearchComp T2 & T3 | Public-company financial lookup benchmarks from filings ([paper](https://arxiv.org/pdf/2509.13160)). T2 covers simple historical lookups; T3 covers complex historical investigations. Grading follows the paper's judge prompt; numbers in different formats (e.g. `12.45%` vs `0.1245`) are treated as equivalent. | `--datasets fin_search_comp_t2_global fin_search_comp_t3_global` |


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
| You.com                     | `YDC_API_KEY`           |

Grading uses OpenAI models by default, but Gemini models are also supported. Set `OPENAI_API_KEY` or 
`GOOGLE_GEMINI_KEY` as appropriate for the LLM judge.

## Usage

### Basic instructions

Run evaluations from the command line via the eval runner:

```bash
# List available samplers and datasets
python src/evals/eval_runner.py --help

# Run SimpleQA and FRAMES on default samplers (does not include You.com Research endpoints)
python src/evals/eval_runner.py

# Run SimpleQA for specific samplers only
python src/evals/eval_runner.py --samplers you_search_with_livecrawl tavily_search --datasets simpleqa

# Run FRAMES evaluation
python src/evals/eval_runner.py --datasets frames

# Run on a limited number of problems (e.g. 100 for a quick sanity check)
python src/evals/eval_runner.py --samplers you_search_with_livecrawl --datasets simpleqa --limit 100

# Fresh run: clear existing results and re-run
python src/evals/eval_runner.py --clean --samplers you_search_with_livecrawl --datasets simpleqa --limit 100
```

#### Important Notes
- To avoid unintended high credit usage, You.com's Research endpoints are not included in the default samplers. They can 
be evaluated by calling them explicitly, like `--samplers you_research_standard` or by using `--samplers all`.
- The BrowseComp and Deep Search QA Datasets are not included in the default benchmark dataset list because they are 
intended to evaluate Research endpoints. 

### LLM's for synthesis and judging
By default, GPT 5.4 nano is used for synthesis and GPT 5.4 mini via the OpenAI API is used for grading. 
This codebase also supports Gemini models via the Google `genai` library. To use an alternative OpenAI model or a 
Gemini model, simply update the model name in `src.constants.py`. The code will interpret whether you are using a GPT
or Gemini model and route your request appropriately.

### Other configuration options

| Option               | Flag / default              | Description                                                        |
|----------------------|-----------------------------|--------------------------------------------------------------------|
| Samplers             | `--samplers <names>`        | One or more sampler names (default: All except You.com Research).  |
| Datasets             | `--datasets <names>`        | One or more datasets (default: `simpleqa`, `frames`).              |
| Limit                | `--limit <n>`               | Run on at most `n` problems (optional).                            |
| Batch size           | `--batch-size 50`           | Number of problems per batch before writing results (default: 50). |
| Max concurrent tasks | `--max-concurrent-tasks 10` | Concurrency limit (default: 10).                                   |
| Clean                | `--clean`                   | Remove existing results and run from scratch. (default False)      |

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

## Output

Results are written to `src/evals/results/` with the following structure:

```
src/evals/results/
├── dataset_<dataset_name>_raw_results_<sampler_name>.csv   # Per-sampler, per-dataset raw results
└── analyzed_results.csv                               # Aggregated metrics (accuracy, latency) updated after each run
```

Raw CSVs contain per-query fields (e.g. query, generated answer, evaluation result, latencies). After a run, 
`write_metrics()` is called automatically and `analyzed_results.csv` is updated with accuracy and average latency per
sampler and dataset.

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


[^1]: Search results are fetched from each search API, then synthesized into a single answer using an LLM; the answer is graded by an LLM judge. Synthesis uses GPT 5.4 nano and grading uses GPT 5.4 mini (configurable in `src/evals/constants.py`).
[^2]: Grading uses prompts aligned with the standard benchmarks as specified in the original papers or repositories (e.g. [SimpleQA](https://openai.com/index/introducing-simpleqa/) and [FRAMES](https://arxiv.org/abs/2409.12941).
[^3]: FinSearchComp grading uses the judge prompt from the [FinSearchComp paper](https://arxiv.org/pdf/2509.13160).
