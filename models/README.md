# Reproduced models

These are model-only exports of the best development-selected checkpoints from the frozen three-seed reproduction. They do not contain replay buffers or optimizer state.

| File | Training seed | Selected gain scale | Random | Greedy | Depth 3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `seed_0.pt` | 0 | 50 | 96.4% | 56.4% | 51.7% |
| `seed_1.pt` | 1 | 50 | 97.0% | 57.8% | 51.3% |
| `seed_2.pt` | 2 | 55 | 96.0% | 56.0% | 51.2% |

Each result uses 1,000 paired held-out games per opponent. Apply the listed gain scale with `evaluate.py --gain-scale`.
