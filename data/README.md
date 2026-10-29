# Data

| File | What it is | Origin |
|------|-----------|--------|
| `paper_reference_values.json` | Numbers the paper reports (Tables 3, 5 to 9 and the Section IV-C human study), each with a page citation | Transcribed from the paper. **Reference only**: never mixed into results this project computes |
| `sample_instances/tiny_handbuilt.json` | 3-request instance used in the unit tests | Hand-built; every value can be checked with a calculator |
| `sample_instances/synthetic_30_requests.json` | 30-request instance, seed 20251028 | Produced by our seeded generator. **Synthetic**: the paper gives no dataset |

The paper's own instances, prompts and the GPT-4o outputs are NOT AVAILABLE. See `docs/source_audit.md`.
Instances can be loaded in the dashboard (Problem configuration) or edited there and saved again.
