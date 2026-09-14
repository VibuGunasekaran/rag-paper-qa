# Generation results

- Generator and judge: `claude-opus-5` / `claude-opus-5`, server-side refusal fallbacks on.
- Retrieval feeding generation: 512 tokens, hybrid, rerank=True, title_prefix=False, top 5 chunks.
- Questions: 150. Refusals: 0. Parse failures: 0. Served by a fallback model: 0.
- Tokens (input/output): generation 554,694/46,359, judge 627,821/13,108; estimated cost $7.4 (includes cached responses from earlier runs).
- Rates carry 95% bootstrap intervals over questions.

| Metric | Rate [95% CI] (n) | Wanted |
|---|---|---|
| Gold evidence in the top 5 (answerable) | 0.79 [0.72, 0.85] (n=135) | high |
| Correct (judge, answerable) | 0.82 [0.76, 0.88] (n=135) | high |
| Correct or partial (judge, answerable) | 0.91 [0.86, 0.96] (n=135) | high |
| Correct when evidence was retrieved | 0.95 [0.91, 0.99] (n=106) | high |
| Faithful: every claim supported (judge, answered) | 1.00 [1.00, 1.00] (n=124) | high |
| Abstained on unanswerable questions | 1.00 [1.00, 1.00] (n=15) | high |
| Abstained on answerable, evidence retrieved | 0.01 [0.00, 0.03] (n=106) | low |
| Abstained on answerable, evidence not retrieved | 0.34 [0.17, 0.52] (n=29) | high |
| Citations valid and consistent with inline markers | 1.00 [1.00, 1.00] (n=124) | high |
| Answer cites a chunk holding the gold evidence | 1.00 [1.00, 1.00] (n=105) | high |

## By question type

| Type | n | Correct | Faithful (full) | Abstained |
|---|---|---|---|---|
| single_fact | 60 | 0.93 [0.87, 0.98] (n=60) | 1.00 [1.00, 1.00] (n=57) | 0.05 [0.00, 0.12] (n=60) |
| multi_hop | 30 | 0.90 [0.80, 1.00] (n=30) | 1.00 [1.00, 1.00] (n=30) | 0.00 [0.00, 0.00] (n=30) |
| comparative | 24 | 0.58 [0.38, 0.79] (n=24) | 1.00 [1.00, 1.00] (n=19) | 0.21 [0.04, 0.38] (n=24) |
| paraphrased | 21 | 0.67 [0.48, 0.86] (n=21) | 1.00 [1.00, 1.00] (n=18) | 0.14 [0.00, 0.29] (n=21) |
| unanswerable | 15 | 1.00 [1.00, 1.00] (n=15) | - | 1.00 [1.00, 1.00] (n=15) |
