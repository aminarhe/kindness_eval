"""Offline tests: no API key needed. Inspect's mockllm model returns scripted outputs."""

import math

from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from kindness_eval_inspect.self_report import extract_rating, self_report
from kindness_eval_inspect.sjt import parse_verdict, sjt


def mock(*completions: str):
    return get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", c) for c in completions],
    )


def metrics(log) -> dict[str, float]:
    return {k: m.value for s in log.results.scores for k, m in s.metrics.items()}


# --- parsing (same rules as the notebook) -----------------------------------


def test_extract_rating():
    assert extract_rating("6") == 6.0
    assert extract_rating("I'd say 3, maybe 5") == 5.0  # last digit wins
    assert extract_rating("8 or 9") is None
    assert extract_rating("") is None
    assert extract_rating(None) is None


def test_parse_verdict():
    assert parse_verdict("kind") == "kind"
    assert parse_verdict("Unkind.") == "unkind"  # 'unkind' checked before 'kind'
    assert parse_verdict("The answer is kind") == "kind"
    assert parse_verdict("no idea") == "unclear"
    assert parse_verdict("") == "error"


# --- end-to-end on the mock model -------------------------------------------


def test_self_report_metrics():
    task = self_report(include_controls=False, epochs=1)
    [log] = eval(task, model=mock("7", "5", "no number here"), limit=3, display="none")
    assert log.status == "success"
    m = metrics(log)
    assert m["all"] == 6.0  # unparsed response is skipped, not counted as 0
    assert math.isclose(m["parse_rate"], 2 / 3)


def test_sjt_metrics():
    # 4 responses from the model under test (the last one empty), then the judge's verdicts
    # for the 3 non-empty ones. An empty response is never sent to the judge.
    task = sjt(epochs=1, temperature=0.0)
    [log] = eval(
        task,
        model=mock("Help him.", "Keep walking.", "Hmm.", ""),
        model_roles={"grader": mock("kind", "unkind", "unclear")},
        limit=4,
        max_connections=1,
        display="none",
    )
    assert log.status == "success"
    m = metrics(log)
    assert m["all"] == 0.5  # kind / (kind + unkind)
    assert m["unclear_rate"] == 0.25
    assert m["no_response_rate"] == 0.25
    assert m["judge_error_rate"] == 0.0


def test_sjt_judge_error_is_reported():
    task = sjt(epochs=1, temperature=0.0)
    [log] = eval(
        task,
        model=mock("Help him."),
        model_roles={"grader": mock("")},  # e.g. a reasoning judge that ran out of tokens
        limit=1,
        display="none",
    )
    m = metrics(log)
    assert m["judge_error_rate"] == 1.0
    assert m["unclear_rate"] == 0.0
    assert math.isnan(m["all"])


def test_sjt_pushback_scores_both_answers():
    task = sjt(pushback=True)
    [log] = eval(
        task,
        model=mock("Help him.", "Fine, keep walking."),
        model_roles={"grader": mock("unkind", "kind")},  # final answer judged first
        limit=1,
        display="none",
    )
    assert log.status == "success"
    m = metrics(log)
    assert m["initial_all"] == 1.0
    assert m["all"] == 0.0


def test_filters():
    assert len(sjt(dimensions=["A"], ambiguity=["low"]).dataset) == 5
    assert len(sjt().dataset) == 63
    assert len(self_report().dataset) == 24
