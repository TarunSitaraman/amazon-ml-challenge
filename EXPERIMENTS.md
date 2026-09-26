# Experiment log

Every row is measured on India train, 12,000 training / 6,000 validation
entities unless noted, scored on the held-out half. Kept so failed directions
are not retried, and so the methodology write-up rests on measurements rather
than recollection.

`ceiling` = macro F_0.5 a PERFECT matcher would score on the candidate sets we
actually keep. `score` = adaptive-k macro F_0.5.

## Leaderboard calibration

| submission | model | offline (India) | leaderboard | gap |
|---|---|---|---|---|
| 1 | geometry + string features, CAND_CAP 60 | 0.8615 | **0.844** | -0.0175 |

Offline tracks online to within ~0.02, so offline is a usable proxy.

The gap is consistent with France (15% of test, zero labels) scoring ~0.75
while US/India transfer roughly as measured: 0.85*0.86 + 0.15*0.75 = 0.8435.
France abstains on 11.53% of entities against US 5.58%, and each wrong
abstention scores 0. One number cannot confirm this; US validation can
separate a France-specific gap from general shrinkage.

## Results

| # | change | ceiling | score | verdict |
|---|---|---|---|---|
| 1 | baseline: geometry features only | 0.9345 | 0.6058 | — |
| 2 | + string features | 0.9345 | **0.8615** | **keep** (+0.2557) |
| 3 | + C3, bug fixes, singleton head | 0.9391 | 0.8686 | keep (+0.0071) |
| 4 | DF_CAP 2k→20k, ADDR 10k→50k | 0.8323 | 0.7812 | **reject** (−0.087) |
| 5 | CAND_CAP 60→150 | 0.9552 | **0.8815** | **keep** (+0.0129) |
| 6 | TOP_K 90/140, CAND_CAP 300 | **0.9622** | 0.8573 | **reject** (−0.024) |
| 7 | hard negatives 0.5/0.25/0.25 (15k/8k) | 0.9552 | 0.8810 vs 0.8792 | keep (+0.0018; chains +0.0069) |
| 8 | deep retrieval + lgb pre-ranker cut to 60 (15k/6k) | 0.9529 | 0.8769 vs 0.8808 | **reject** (−0.0039) |
| 9 | **India+US reference**, 15k/6k per country, corrected split | 0.9648 | **0.8960** | reference for submission 2 |
| 10 | hard negatives on India+US | — | 0.9107 vs 0.9134 (US-only split) | **reject** (−0.0027; chains −0.0029) |
| 11 | 50k/country, NEG_KEEP 0.2 (PR #22), 6.03M pairs | 0.9638 | 0.8902 vs 0.8960 | no gain (−0.0058, ~1.6 SE); data size vs subsampling confounded |
| 12 | **C4 address-token-pair channel** (PR #19), India+US 15k/6k | **0.9781** | **0.9038** vs 0.8960 | **keep** (+0.0078) |
| 13 | **learned Devanagari transliteration** (PR #20) + C4 | 0.9785 | **0.9083** vs 0.9038 | **keep** (+0.0045) |
| 14 | **corruption-grammar features** (PR #17, miner run) + C4 + translit | 0.9785 | **0.9166** vs 0.9083 | **keep** (+0.0083) |
| 15 | + singleton head (gate passed: precision 0.762, recall 0.905) | 0.9785 | **0.9185** | **keep** (+0.0019) |

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

## C4 address pairs (PR #19, merged)

Pairs of rare address tokens, preferring digit tokens (house number x locality),
mirroring C3 for names. Ceiling gain was consistent on every split: India
train/valid +0.0125/+0.0140, US train/valid +0.0133/+0.0145. It gained 1.9-3.0%
of links while pushing only 0.02-0.05% out of the cap -- larger than the 1.31%
'address too common' category, so it also recovers links that were reachable
but ranked out. US ceiling reached 0.9873. End to end +0.0078. Enable with C4=1.

## Learned transliteration (PR #20, merged)

The generator's Devanagari vocabulary is closed: 197 distinct held-out word
pairs, 0.0% unseen in training. A whole-word lexicon learned from aligned
pairs (114 words, 23 grapheme table entries) takes held-out exact token match
from 28.31% (rules) to 98.32%. Ceiling barely moves (India valid 0.9683 ->
0.9686, +36 links) because address channels already reached cross-script links;
the +0.0045 end-to-end comes from the MATCHER, whose name features now carry
signal for Devanagari records. Needs data/translit_model.json at predict time
(regenerate with src/mine_translit.py). Two Windows fixes were needed: the model
file was written with the cp1252 default encoding, and the report crashed
formatting a None rate.

## No leak in IDs or row order (diag_leak.py)

IDs are uniform random 9-digit integers; a linked record's ID tracks its S1
entity's at Spearman +0.0008 (S2) / -0.0016 (S3); an entity's records are never
ID- or row-neighbours (0.0000% vs ~0.0001% random); distractors are placed
uniformly. The leaderboard top (0.990) is not exploiting a trivial shortcut.

Two legitimate text signals the matcher underuses:
- Corruption marks. Real matches are corrupted copies; distractors look clean:
  name tokens 3.30 vs 4.04, empty address 4.5% vs 0.3%, all-caps (S2) 20.5% vs
  14.5%, all-lower 6.8% vs 2.7%.
- Cross-source twins. An S2 record with an exact-name S3 record is matched
  24.9% of the time vs 9.9% for distractors (2.5x).

## Where the matcher loses (diag_matcher.py, C4 + translit model)

Gap 0.0702 (realises 92.8% of ceiling 0.9785). False negatives are 60.3% of it:
'stopped early' alone is 38.8% (1,612 true matches ranked above every rejected
FP of their entity, yet not accepted). False positives 39.7%: distractors 21.8%,
siblings 16.9%, chains 1.0%.

'Stopped early' is NOT a decision-layer problem: tune_decision's best rule
(fixed 0.6) gains only +0.0007, and the recall correction moves mean k 2.91 ->
2.99. Those true matches have low absolute p; any rule that accepts them accepts
same-scored FPs in other entities. The matcher is under-confident about an
entity's 3rd/4th matches -- which a triangulation feature (similarity to the
entity's own confident candidates) targets directly.

## Corruption grammar (PR #11 miner + PR #17 features)

Mined from 300k sampled aligned pairs: 35,899 entries (abbreviations, region
codes, forbidden swaps, junk affixes). The ceiling is unchanged (0.9785) -- the
features act only in the matcher -- yet the score rises +0.0083. Consistent with
diag_misses (abbreviation causes only 0.06% of recall loss): the grammar is a
PRECISION device. Its forbidden table turns 'these two names share no token'
from neutral into evidence against a match, which a precision-weighted metric
rewards. With these features the singleton head passes its gate for the first
time (+0.0019). Needs data/corruption_grammar.json at predict time
(regenerate with src/mine_corruption.py --sample 300000).

## Pre-ranker (PR #16, closed unmerged)

A learned lgb pre-ranker reaches a much higher ceiling than chan.max at small
caps (cap 30: 0.9400 vs 0.8463; cap 60: 0.9509 vs 0.9391) but every ranker
converges to retrieval's maximum, 0.9552, by cap 120 -- it cannot find what
retrieval did not return. End to end, deep retrieval cut to 60 by lgb lost to
chan.max at 150 (0.8769 vs 0.8808).

So the gap between reachable (97.26%) and kept (~90%) sits INSIDE the channels'
per-channel top-k, not at the final cut, and the matcher copes fine with ~87
candidates: its collapse in experiment 6 only appeared near 186. The candidate
list is not the constraint. The matcher is.

## Open

- Learned pre-ranker for the cut (PR #16) — the direct attack on experiment 6.
- Hard-negative reweighting (PR #14) — targets the chain population, where
  39.28% of S1 entities share a name with a different business.
- Corruption grammar, mined but not yet run or consumed.
- Everything here is India-only. US is 38% of test entities, France 15%, and
  France already behaves differently (11.53% predicted singletons vs US 5.58%).
