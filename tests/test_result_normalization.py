"""Tests for normalizing sampler output between the synthesis and no-synthesis paths."""

from evals.samplers.base_samplers.base_sampler import normalize_formatted_results


def test_string_is_wrapped_when_synthesis_is_needed():
    """Regression: a bare string would otherwise be iterated character by character."""
    assert normalize_formatted_results("an answer", needs_synthesis=True) == [
        "an answer"
    ]


def test_synthesis_input_survives_the_join_it_will_receive():
    """The synthesis step joins with a separator; a string input corrupts the text."""
    normalized = normalize_formatted_results("Revenue was $4,836M", needs_synthesis=True)

    assert "\n---\n".join(normalized) == "Revenue was $4,836M"


def test_list_is_untouched_when_synthesis_is_needed():
    results = ["[a](http://a)\ntext", "[b](http://b)\ntext"]

    assert normalize_formatted_results(results, needs_synthesis=True) == results


def test_list_is_joined_when_synthesis_is_skipped():
    """Otherwise the grader receives a Python list repr rather than the answer."""
    assert (
        normalize_formatted_results(["the answer"], needs_synthesis=False)
        == "the answer"
    )


def test_string_is_untouched_when_synthesis_is_skipped():
    assert (
        normalize_formatted_results("the answer", needs_synthesis=False) == "the answer"
    )


def test_empty_values_are_preserved():
    assert normalize_formatted_results("", needs_synthesis=False) == ""
    assert normalize_formatted_results([], needs_synthesis=False) == ""
    assert normalize_formatted_results("", needs_synthesis=True) == [""]
