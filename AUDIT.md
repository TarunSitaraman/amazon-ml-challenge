# Stage 0 — Data Audit

All numbers measured from the provided files. Nothing here is assumed.
Reproduce with `src/prepare_data.py`, then `audit_data.py`, `audit_signals.py`,
`audit_recall_ceiling.py` (run from the repo root).

## Scale

| | S1 | S2 | S3 |
|---|---|---|---|
| train | 2,206,821 | 5,034,616 | 5,285,603 |
| test | 1,732,544 | 4,887,273 | 5,082,316 |

Test S1 by country: US 663,106 · India 809,986 · **France 259,452 (15.0%)**.
France appears only in test — 15% of the macro-average carries zero training labels.

`business_name` is never empty. `business_address` is empty in ~2.6% of S2/S3
(test: S2 129,408, S3 136,098) and never in S1.

## Structural constraints (the two that change the architecture)

**1. Disjoint assignment.** 7,638,365 links over 7,638,365 *distinct* S2/S3 records.
Record re-use is exactly **0** — every S2/S3 record belongs to at most one S1 entity.
The task is constrained assignment, not independent pair classification. A candidate's
score is only meaningful relative to its best rival S1 suitor.

**2. Country never crosses.** 0 / 7,638,365 cross-country links. Country is a
verified-safe hard block, and France is therefore self-contained.

## Match cardinality

Mean 3.46, max 11. Singletons are **rare**, so "predict empty when unsure" is wrong here.

| matches | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8+ |
|---|---|---|---|---|---|---|---|---|---|
| share | 5.58% | 5.40% | 17.00% | 24.05% | 21.94% | 14.59% | 7.47% | 2.90% | ~1.06% |

Joint (n_S2, n_S3) top cells: (1,1) 12.22%, (1,2) 11.38%, (2,1) 10.11%, (2,2) 9.43%.
Matched in S2 only: 143,029. S3 only: 164,498.

Metric consequence — for an entity with 3 true matches: 2 correct scores **0.909**;
all 3 with one extra wrong scores **0.667**. A confident subset beats a greedy superset.

## Distractors

S2: 3,693,619 / 5,034,616 matched (73.36%) → **1,340,997 match nothing**.
S3: 3,944,746 / 5,285,603 matched (74.63%) → **1,340,857 match nothing**.
The near-identical distractor counts suggest a fixed synthetic distractor budget.

## Separability: true matches vs random same-country records

Sample of 100,000 S1 entities / 366,431 true links.

| | TRUE S2 | TRUE S3 | RANDOM |
|---|---|---|---|
| name normalised equal | 25.47% | 26.27% | 0.00% |
| name token Jaccard (mean) | 0.635 | 0.653 | 0.031 |
| name Jaccard = 0 | 16.09% | 12.44% | 83.92% |
| one name ⊆ other | 57.52% | 61.57% | 0.05% |
| address shares a number | 77.98% | 79.25% | 1.79% |

Random negatives are trivially separable. **The real difficulty is hard negatives**,
so model selection should be driven by hard-negative performance, not by these gaps.
Note 16.09% of true S2 links share *no* name token — name-only blocking cannot see them.

## Chains — the false-merge minefield

**39.28%** of S1 entities (866,894) share a normalised `business_name` with a *different*
S1 entity. Since S1 is deduplicated, these are provably distinct businesses.
Worst: `primary care group` ×253, `ear nose throat group` ×251, `pediatric group` ×222.

Name similarity alone is therefore catastrophic on ~39% of the data — **address is the
only discriminator for these**, and aggressive name normalisation actively creates merges.

## Script asymmetry — transliteration is one-directional

| source | Devanagari names | share of its India rows |
|---|---|---|
| S1 | **0** | 0.00% |
| S2 | 269,424 | 13.35% |
| S3 | 158,003 | 7.47% |

S1 is **always Latin**. Only Devanagari→Latin is ever needed, never the reverse — and
7.6M aligned true pairs are available to learn it from, with no external data.

## Identity-poor tokens (S1 name field)

`limited` 23.67% · `private` 19.60% · `llc` 16.12% · `inc` 10.80% · `ltd` 6.73% ·
`pvt` 5.50% · `india` 2.78% · `and` 2.60%. IDF weighting is mandatory; these tokens
carry no identity signal but dominate raw overlap.

## Blocking recall ceiling

Per-channel recall over 219,534 true links. **A channel's recall is not its usability** —
`addr_token` means "shares any address token", which is high-recall but far too
non-selective to use alone. Selectivity must be measured separately.

| channel | ALL | S2 | S3 | US | India | same script | script mismatch |
|---|---|---|---|---|---|---|---|
| name_token | 85.74% | 83.83% | 87.54% | 92.17% | 76.06% | 89.06% | 7.08% |
| name_4gram | 90.78% | 89.02% | 92.44% | 97.65% | 80.43% | 94.33% | 6.75% |
| name_any | 91.38% | 89.56% | 93.10% | 98.25% | 81.04% | 94.94% | 7.08% |
| addr_number | 78.56% | 77.77% | 79.31% | 76.26% | 82.02% | 78.38% | 82.85% |
| addr_token | 95.57% | 95.54% | 95.59% | 95.27% | 96.01% | 95.38% | **99.88%** |
| addr_rare_token | 84.31% | 85.04% | 83.62% | 83.63% | 85.33% | 84.29% | 84.75% |
| **UNION** | **100.00%** | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

Conclusions:
- **No true link is unreachable.** The information is present; blocking is purely an
  engineering problem of retrieving it cheaply.
- **Name and address are complementary, not redundant.** Name collapses to ~7% on
  script mismatch; address holds at 99.88%. Address is what rescues transliteration.
- **India is much harder on names** (name_4gram 80.43% vs US 97.65%) and leans on address.
