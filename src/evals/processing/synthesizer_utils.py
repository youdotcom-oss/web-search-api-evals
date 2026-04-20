import math

import tiktoken

from evals.processing.synthesize_answer import SynthesizeAnswer
from evals.constants import (
    SYNTHESIS_PROMPT,
    MAX_SEARCH_RESULT_TOKENS,
    SYNTHESIS_MODEL,
)

_FALLBACK_ENCODING = "cl100k_base"


def _get_encoding(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding(_FALLBACK_ENCODING)


async def trim_results_to_model_limit(
    formatted_results: list[str],
    synthesis_model: str,
) -> list[str]:
    enc = _get_encoding(synthesis_model)
    results_with_tokens = sorted(
        [(r, enc.encode(r)) for r in formatted_results], key=lambda x: len(x[1])
    )

    remaining_search_result_tokens = MAX_SEARCH_RESULT_TOKENS
    trimmed_results = []

    for i, (result, tokens) in enumerate(results_with_tokens):
        remaining_results = len(results_with_tokens) - i
        max_tokens_per_result = math.floor(
            remaining_search_result_tokens / remaining_results
        )

        if len(tokens) <= max_tokens_per_result:
            trimmed_results.append(result)
            remaining_search_result_tokens -= len(tokens)
        else:
            trimmed_results.append(enc.decode(tokens[:max_tokens_per_result]))
            remaining_search_result_tokens -= max_tokens_per_result

    return trimmed_results


async def synthesize_response(
    query: str,
    formatted_results: list[str],
    synthesis_model: str = SYNTHESIS_MODEL,
) -> str:
    trimmed_results = await trim_results_to_model_limit(
        formatted_results, synthesis_model
    )
    concatenated_results = "\n---\n".join(trimmed_results)
    answer_synthesizer = SynthesizeAnswer(
        SYNTHESIS_PROMPT, max_retries=3, synthesis_model=synthesis_model
    )
    result = await answer_synthesizer.process_single(query, concatenated_results)
    return result.response_text if result else f"Synthesis failed for: {query}"
