# Experiment log

Every row is measured on India train, 12,000 training / 6,000 validation
entities unless noted, scored on the held-out half. Kept so failed directions
are not retried, and so the methodology write-up rests on measurements rather
than recollection.

`ceiling` = macro F_0.5 a PERFECT matcher would score on the candidate sets we
actually keep. `score` = adaptive-k macro F_0.5.

## Results

| # | change | ceiling | score | verdict |
|---|---|---|---|---|
| 1 | baseline: geometry features only | 0.9345 | 0.6058 | — |
| 2 | + string features | 0.9345 | **0.8615** | **keep** (+0.2557) |
| 3 | + C3, bug fixes, singleton head | 0.9391 | 0.8686 | keep (+0.0071) |
| 4 | DF_CAP 2k→20k, ADDR 10k→50k | 0.8323 | 0.7812 | **reject** (−0.087) |
| 5 | CAND_CAP 60→150 | 0.9552 | **0.8815** | **keep** (+0.0129) |
| 6 | TOP_K 90/140, CAND_CAP 300 | **0.9622** | 0.8573 | **reject** (−0.024) |

## What the numbers mean

**Experiments 4 and 6 are the important ones, and they say opposite-sounding
things that reconcile into one rule.**

Experiment 4: loosening the df cap at a fixed budget floods the candidate list
with distractors, which outrank true matches and push them out. Retrieved true
matches per entity fell 3.02 → 2.47.

Experiment 5: widening the budget at a tight df cap keeps what the channels
already found. 3.02 → 3.14.

> Keep retrieval tight and selective; spend the budget on plausible candidates.
> Wider retrieval and a bigger budget are not the same lever.

**Experiment 6 identifies the current bottleneck.** The ceiling rose to 0.9622
while the score fell to 0.8573. More true matches became *available* and the
matcher got *worse*, because positives fell to 1.72% and it could not reject
the extra distractors.

> The binding constraint is matcher discrimination, not retrieval. We can
> already reach more matches than the matcher can exploit. Retrieval tuning is
> exhausted until the matcher improves.

**Recall has plateaued at ~93.5%.** Retrieved true matches per entity by
candidate count: 59.4 → 3.02, 87.1 → 3.14, 185.7 → 3.19. Doubling the pool a
second time bought +0.05. The remaining ~6.5% of links are pairs our channels
cannot reach *at all* — no shared token, 4-gram or address key. Abbreviation
(`Corp`/`Corporation` share nothing) is the named cause, which is why the
corruption grammar belongs in the **normaliser and blocking keys**, not only in
the matcher.

## Corrections to earlier assumptions

**The ceiling-vs-recall table in RESEARCH.md is ~0.02 optimistic.** It assumes
each link is retrieved independently. Real losses are correlated — an entity
tends to get most of its matches or few — and correlated loss gives a lower
macro ceiling. Measured 0.9552 at 92.1% recall where the table predicts ~0.975.
Anyone planning a target against that table should subtract ~0.02. This raises
the recall needed for 0.986 from ~96% to nearer 99%.

**"Reachability" is not "recall".** `diag_dfcap.py` measured whether a shared
token exists at all (96.31% at cap 10,000). That says nothing about whether the
candidate survives the per-entity cut. Conflating the two sent this project
after RAM and cloud credits for a problem that was `CAND_CAP = 60`.

**Disjointness does not supply new training labels.** Ground truth is complete,
so every blocked candidate already carries a certified label. What the zero
re-use property adds is each negative's *owner*, i.e. why it is a negative
(chain sibling, same normalised name, unrelated distractor). That is a
reweighting signal, not extra data. (Correction supplied by PR #14.)

## Cheap signals that predicted expensive results

The **positives rate** printed in the first lines of training predicted both
rejected experiments before their ceilings were computed:

| candidates/entity | positives | true matches retrieved | recall |
|---|---|---|---|
| 59.4 | 5.08% | 3.02 | 88.6% |
| 87.1 | 3.61% | 3.14 | 92.1% |
| 185.7 | 1.72% | 3.19 | 93.5% |

`candidates × positives` is the retrieved-true-matches-per-entity figure, and it
tracks the ceiling directly. Check it before spending 20 minutes on a run.

## Rejected after measurement

- **fold / skeleton name channels** — +0.06% marginal recall; their apparent
  per-channel attribution was overlap with the address channel. Cost two
  inverted indices and per-record Python string generation.
- **per-digit address keys in C5** — +0.01% recall for 40 minutes of wall
  clock; C6 already indexes address tokens, digits included.
- **vectorised strfeatures (PR #9)** — claimed ~16x, measured 0.9x on 120,000
  retrieval-shaped pairs, in 176 lines against 104. Reverted.
- **redecide in resolve_conflicts** — flat to slightly worse; defaulted off by
  its author.

## Accepted after measurement

- **strfeatures precompute (PR #13)** — claimed 17x, measured 6.1x on build and
  5.1x end-to-end, output bit-identical. Moved string features from ~75% of
  runtime to 1.9%.
- **singleton head (PR #7)** — precision 0.626 against a 0.55 gate and 0.474
  break-even, +0.0012.

## Open

- Learned pre-ranker for the cut (PR #16) — the direct attack on experiment 6.
- Hard-negative reweighting (PR #14) — targets the chain population, where
  39.28% of S1 entities share a name with a different business.
- Corruption grammar, mined but not yet run or consumed.
- Everything here is India-only. US is 38% of test entities, France 15%, and
  France already behaves differently (11.53% predicted singletons vs US 5.58%).
