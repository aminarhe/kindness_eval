"""Phase 1: models rate themselves on the Youngs et al. kindness scale (1-7).

    inspect eval kindness_eval_inspect/self_report --model groq/openai/gpt-oss-20b
    inspect eval kindness_eval_inspect/self_report -T framing=ai --model ...
"""

import re
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, Target, grouped, scorer
from inspect_ai.solver import TaskState, generate, system_message

from kindness_eval_inspect._data import (
    SR_SYS_AI,
    SR_SYS_HUMAN,
    SR_USR_AI,
    SR_USR_HUMAN,
    kindness_items,
)
from kindness_eval_inspect.metrics import mean_rating, parse_rate


def extract_rating(text: str | None) -> float | None:
    """Same rule as the notebook: the last standalone digit 1-7 in the response."""
    if not text:
        return None
    m = re.findall(r"\b([1-7])\b", text)
    return float(m[-1]) if m else None


@scorer(metrics=[grouped(mean_rating(), "dimension"), parse_rate()])
def likert_rating():
    async def score(state: TaskState, target: Target) -> Score:
        raw = state.output.completion
        rating = extract_rating(raw)
        return Score(
            value={"rating": rating or 0.0, "parsed": 1.0 if rating is not None else 0.0},
            answer=raw,
            explanation=f"rating={rating}" if rating is not None else "no 1-7 digit found",
        )

    return score


@task
def self_report(
    framing: Literal["human", "ai"] = "human",
    include_controls: bool = True,
    epochs: int = 3,
    temperature: float = 0.0,
) -> Task:
    """Self-reported kindness.

    Args:
        framing: "human" (as in the spring runs) or "ai" (rate yourself as an assistant).
        include_controls: include the 3 neutral control items (dimension CONTROL).
        epochs: repetitions per item (spring runs: 3).
        temperature: sampling temperature (spring runs: 0.0).
    """
    sys_prompt = SR_SYS_HUMAN if framing == "human" else SR_SYS_AI
    usr_tmpl = SR_USR_HUMAN if framing == "human" else SR_USR_AI

    dataset = MemoryDataset(
        [
            Sample(
                id=item["id"],
                input=usr_tmpl.format(statement=item["text"]),
                metadata={
                    "dimension": item["dimension"],
                    "item_text": item["text"],
                    "framing": framing,
                },
            )
            for item in kindness_items(include_controls)
        ],
        name="kindness_self_report",
    )

    return Task(
        dataset=dataset,
        solver=[system_message(sys_prompt), generate()],
        scorer=likert_rating(),
        epochs=epochs,
        config=GenerateConfig(temperature=temperature, max_tokens=512),
    )
