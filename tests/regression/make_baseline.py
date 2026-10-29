import json
import sys

sys.path[:0] = ["src", "."]
from tests.regression.test_regression import BASELINE, _compute  # noqa: E402

BASELINE.write_text(json.dumps(_compute(), indent=1, sort_keys=True))
print("wrote", BASELINE)
