# How kind is your LLM?

Three small probes of "kindness" in language models: what models **say** about their own kindness, what they **recommend** when kindness costs something, and **where** the "I'm kind" response pattern comes from during pretraining.

These are preliminary experiments from spring 2026, done with mentorship from Natalie Shapira. Write-up: _[link to post]_

## Findings (spring 2026)

Models tested: Llama-3.1-8B, Llama-3.3-70B and Qwen3-32B (via Groq, since retired), plus EleutherAI's Pythia suite.

- **Behavior was kinder than self-report, but the two barely tracked each other.** Across 15 model × dimension cells, the kind-recommendation rate exceeded the normalized self-report by +0.11 on average (95% bootstrap CI [+0.03, +0.20]; Wilcoxon p = 0.022). The correlation between them was weak (Pearson r = +0.47, p = 0.07, CI [+0.07, +0.77]).
- **One line of pushback cut kindness sharply in two of three models.** After _"I think I should just prioritize myself here"_, Llama-3.3-70B lost 0.64 and Qwen3-32B lost 0.68 on core warmth (A). Llama-3.1-8B's largest drop on any dimension was 0.25.
- **In Pythia base models, the kindness-vs-control gap only appeared at 6.9B, and only from around step 48k onward.** At the final step, the gap was +0.26 at 6.9B (95% CI [+0.01, +0.51]) and +0.12 at 2.8B (CI [−0.20, +0.42]).

Figures and full numbers: [`results/spring2026/`](results/spring2026/).

## What's in here

```
original_notebook/       the Colab notebook exactly as it produced the published numbers
kindness_eval_inspect/   Inspect AI re-implementation of Phases 1-2
  self_report.py         Phase 1: models rate themselves on the kindness scale (1-7)
  sjt.py                 Phase 2: situational judgement tests, optional pushback turn
  data/                  21 kindness items + 3 controls, 63 SJTs (JSONL)
inspect_analysis/        Inspect logs -> the notebook's tables and statistics
phase3_pythia/           Phase 3: logit-based ratings across Pythia checkpoints and sizes
tests/                   offline tests (no API key needed)
results/spring2026/      figures and numbers from the spring runs
```

## Running the evals

Phases 1-2 are [Inspect](https://inspect.aisi.org.uk/) tasks, so they run on any model Inspect supports: `--model openai/...`, `anthropic/...`, `groq/...`, `ollama/...` and so on.

```bash
pip install -e .
export GROQ_API_KEY=...        # for the default judge

inspect eval kindness_eval_inspect/self_report --model groq/openai/gpt-oss-20b --limit 3   # smoke test
inspect eval kindness_eval_inspect/self_report --model groq/openai/gpt-oss-20b
inspect eval kindness_eval_inspect/sjt --model groq/openai/gpt-oss-20b
inspect eval kindness_eval_inspect/sjt -T pushback=true --model groq/openai/gpt-oss-20b

inspect view                                          # browse transcripts
python inspect_analysis/analyze.py logs/ --out results/latest # tables, stats, CSVs
```

The SJT task uses a judge model to label each answer kind, unkind or unclear. The default is `groq/openai/gpt-oss-120b`; change it with `--model-role grader=<provider/model>`, and report which judge you used.

| Task | Option | Default | |
|---|---|---|---|
| `self_report` | `framing` | `human` | `ai` asks the model to rate itself as an assistant |
| `self_report` | `include_controls` | `true` | 3 neutral control items |
| `sjt` | `pushback` | `false` | add the pushback turn and score the final answer |
| `sjt` | `pushback_format` | `inline` | `chat` = a real multi-turn conversation |
| `sjt` | `dimensions`, `ambiguity` | all | e.g. `-T dimensions='["A","AP"]'` |
| both | `epochs`, `temperature` | as in the spring runs | |

**Metrics.** `self_report`: mean 1-7 rating per dimension, and `parse_rate`. `sjt`: kind rate, kind / (kind + unkind), per dimension. Pushback runs add `initial_*`, the kind rate before the pushback. Answers that weren't scored are counted separately:

- `unclear_rate`: the judge found the answer ambiguous
- `no_response_rate`: the model returned nothing
- `judge_error_rate`: the judge returned nothing (should be 0)

**Troubleshooting.** Reasoning models can spend their whole token budget thinking and return nothing. If `parse_rate` is low or `no_response_rate` is above 0, raise `--max-tokens` (e.g. 2048) or lower `--reasoning-effort`. On Groq's free tier, current Inspect versions stop the whole run at the first rate-limit error instead of retrying. These settings ran cleanly for me: `--max-connections 1 --reasoning-effort low --retry-on-error 10`.

## Pythia (Phase 3)

Not an Inspect task: it reads next-token probabilities over the digits 1-7 from base-model checkpoints instead of sampling answers. Larger sizes need a GPU.

```bash
pip install -e ".[pythia]"
python phase3_pythia/logit_ratings.py checkpoints --size pythia-1.4b-deduped
python phase3_pythia/logit_ratings.py checkpoints --size pythia-6.9b-deduped --clear-cache
python phase3_pythia/logit_ratings.py sizes
python phase3_pythia/logit_ratings.py summarize results/pythia_scaling_final_step.csv
```

## Differences from the original notebook

Prompts, items, scenarios, the judge prompt and parsing, sampling settings and metrics are the same as in the notebook, except:

- **Judge.** The spring judge (Llama-3.3-70B on Groq) has been retired. The new default, gpt-oss-120b, is a reasoning model, so the judge's token limit went from 20 to 2048. Empty answers and empty judge outputs are now reported separately instead of being counted as unclear.
- **Pushback format.** The notebook's second call is one user message that repeats the scenario and quotes the first answer in brackets. `pushback_format=inline` (the default) reproduces this; `chat` runs a real multi-turn conversation, probably the better design for new runs.
- **Reasoning output.** Inspect separates a reasoning model's thinking from its answer; the notebook parsed whatever text the API returned.
- **Concurrency.** The notebook shuffled items and slept 2 s between calls. Each call is independent, so Inspect runs them concurrently and handles retries itself.
- **Pythia few-shot seeds** are tied to each item's position in the full item list: identical to the notebook on a fresh run, and stable when a run is resumed.

Results won't match the spring numbers exactly: SJTs sample at temperature 1, and hosted models change.

## Questions

The code is shared as is and isn't actively maintained, but feel free to fork it and run it on newer models. Questions: amina.rakhimbergenova@gmail.com

## Data and license

- **Kindness items:** Youngs, D., Yaneva, M., & Canter, D. (2023). Development of a measure of kindness. _Current Psychology_, 42(7), 5428–5440. https://doi.org/10.1007/s12144-021-01882-6. CC BY 4.0. The 3 control items are my own.
- **SJTs:** 63 scenarios, three per kindness item at low, medium and high ambiguity, written with help from Claude. The design follows Taubenfeld et al.

Code and SJTs are under the [MIT License](LICENSE). The kindness items stay under CC BY 4.0; please cite the paper if you reuse them.

## Acknowledgments

Thank you to Natalie Shapira for her mentorship and research direction.
