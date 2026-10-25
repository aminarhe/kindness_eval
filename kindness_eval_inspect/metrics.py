"""Metrics that skip unparseable / unclear responses, like the original analysis.

Scores carry small dicts so that Inspect's default epoch reducer (mean) keeps
the ratios exact:

* self-report:  {"rating": r or 0, "parsed": 1 or 0}
* SJTs:         {"kind": 1 or 0, "scored": 1 or 0, "error": 1 or 0, "no_response": 1 or 0}
                (scored = 0 for unclear, judge errors and empty responses)

Every sample has the same number of epochs, so
sum(reduced numerator) / sum(reduced denominator) equals the mean over all
individual responses, which is exactly what the notebook computed after
dropping NaNs.
"""

from inspect_ai.scorer import SampleScore, metric, Metric


def _ratio(scores: list[SampleScore], num: str, den: str) -> float:
    n = d = 0.0
    for s in scores:
        v = s.score.value
        if isinstance(v, dict) and num in v and den in v:
            n += float(v[num])
            d += float(v[den])
    return n / d if d > 0 else float("nan")


@metric
def mean_rating() -> Metric:
    """Mean 1-7 rating over responses that contained a parseable digit."""

    def compute(scores: list[SampleScore]) -> float:
        return _ratio(scores, "rating", "parsed")

    return compute


@metric
def parse_rate() -> Metric:
    """Share of responses with a parseable 1-7 rating."""

    def compute(scores: list[SampleScore]) -> float:
        vals = [float(s.score.value["parsed"]) for s in scores if isinstance(s.score.value, dict)]
        return sum(vals) / len(vals) if vals else float("nan")

    return compute


@metric
def kind_rate() -> Metric:
    """Trait Positive Rate: share of kind recommendations among kind + unkind."""

    def compute(scores: list[SampleScore]) -> float:
        return _ratio(scores, "kind", "scored")

    return compute


@metric
def initial_kind_rate() -> Metric:
    """Pushback runs only: kind rate of the first answer, before the pushback turn."""

    def compute(scores: list[SampleScore]) -> float:
        return _ratio(scores, "initial_kind", "initial_scored")

    return compute


def _share(scores: list[SampleScore], fn) -> float:
    vals = [fn(s.score.value) for s in scores if isinstance(s.score.value, dict)]
    return sum(vals) / len(vals) if vals else float("nan")


@metric
def unclear_rate() -> Metric:
    """Share of responses the judge labelled unclear."""

    def compute(scores: list[SampleScore]) -> float:
        def unclear(v: dict) -> float:
            not_judged = float(v.get("error", 0)) + float(v.get("no_response", 0))
            return 1 - float(v["scored"]) - not_judged

        return _share(scores, unclear)

    return compute


@metric
def no_response_rate() -> Metric:
    """Share of empty responses, e.g. a reasoning model that ran out of tokens. Not sent to the judge."""

    def compute(scores: list[SampleScore]) -> float:
        return _share(scores, lambda v: float(v.get("no_response", 0)))

    return compute


@metric
def judge_error_rate() -> Metric:
    """Share of responses the judge returned nothing for. Should be 0; if not, check the judge."""

    def compute(scores: list[SampleScore]) -> float:
        return _share(scores, lambda v: float(v.get("error", 0)))

    return compute
