# Entry point so `inspect eval kindness_eval_inspect/<task>` works after `pip install -e .`
from kindness_eval_inspect.self_report import self_report  # noqa: F401
from kindness_eval_inspect.sjt import sjt  # noqa: F401
