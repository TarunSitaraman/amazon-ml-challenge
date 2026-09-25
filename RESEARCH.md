# Research Synthesis — Stage 0

Four independent research passes (global structure, blocking at scale, signal/noise modelling,
adversarial critique) run against the measured audit in [AUDIT.md](AUDIT.md), then reconciled.
Claims below are marked **verified** (recomputed here), **derived** (arithmetic from measured
counts), or **unverified** (hypothesis needing measurement).

All rules cited come from the two official PDFs, which are the governing authority for this repo:
`amazon_ml_challenge_problem_statement.pdf` and
`guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf`.

---

## 1. The metric, in closed form

**verified** — algebra plus the PDF's worked example.

```
F_0.5 = 1.25·c / (0.25·n + k)
        c = correct predictions, k = predicted count, n = true count
        F_0.5 = 1.0 when n = 0 and k = 0
```

Checks against the PDF's only worked example (n=2, k=3, c=2): 2.5/3.5 = **0.714** ✓.

> **Correction on record.** An earlier internal brief stated "n=3, predict 1 correct → 0.625".
> That was an arithmetic error (it used n=4). The correct value is **1.25/1.75 = 0.714**.
> The PDF is consistent; nothing in it needs re-verification against the official scorer.

### The stopping rule falls straight out

Adding a candidate with calibrated probability `p` raises expected F₀.₅ iff

```
p > c / (k + 0.25·n)      ⟺      p > 0.8 · F_k
```

**verified** by direct differencing. Three consequences:

- **Ceiling at 0.80.** Since `F_k ≤ 1`, the bar never exceeds 0.80 — any candidate with
  calibrated `p > 0.80` is accepted unconditionally, no tuning.
- **Self-tightening.** The bar *rises* as the set improves. A strong entity demands more of its
  next candidate. This is "a confident subset beats a greedy superset", mechanised.
- **Greedy is optimal.** `p` is non-increasing down the sorted list while the bar is
  non-decreasing, so there is a single crossing — stop at the first rejection, one pass.

### k=0 → k=1 is singular and must be special-cased

The marginal rule is 0/0 at k=0. Deriving properly, with measured `E[n | n≥1] = 3.666`:

```
accept the first candidate iff   p₁ > 1.533 · P̂(n=0 | entity)
```

| P̂(n=0) | 0.056 (prior) | 0.25 | 0.50 | 0.65 |
|---|---|---|---|---|
| entry bar p₁ | **0.086** | 0.383 | 0.766 | 0.996 |

A generic prior sets the entry bar at 8.6% — correctly permissive, since predicting nothing
scores 0 for 94.4% of entities. A good singleton estimate swings this bar ~9×. It is the highest-
leverage single number in the decision layer.

---

## 2. Cardinality, not ranking, is the structural headroom

**verified** — recomputed against the measured match-count distribution.

| policy (perfect ranker) | macro F₀.₅ |
|---|---|
| always predict [] | 0.0558 |
| fixed top-2 | **0.7929** |
| fixed top-3 | **0.8102** ← best fixed k |
| fixed top-4 | 0.7604 |
| oracle k | 1.0000 |

> A *perfect ranker* with any fixed cutoff caps at **0.810**. The remaining 0.190 is pure
> cardinality decision-making and no amount of ranking improvement reaches it.

(One correction: a research pass reported 0.7829 for top-2; recomputation gives **0.7929**.
Use 0.7929 as the unit-test target, or a correct implementation will look buggy.)

73% of that headroom lives in entities with n ∈ {0,1,2} — 27.98% of entities. **The money is in
detecting low cardinality and stopping early, not in finding the 7th match.**

---

## 3. Singleton probability is correlated across sources

**verified** from measured counts.

```
P(n_S2=0) = (123,247 + 164,498) / 2,206,821 = 13.04%
P(n_S3=0) = (123,247 + 143,029) / 2,206,821 = 12.07%
independence would predict both=0 at 1.57% → ~34,720 entities
observed both=0 = 123,247                  → lift = 3.55×
```

`P(n_S3=0 | n_S2=0) = 42.8%` against a 12.07% base rate.

There is a latent per-entity "obscurity" governing whether an entity appears at all; *conditional
on appearing*, the per-source counts are independent (three odds ratios on the joint table all
equal 1.000 ± 0.003). Any pipeline computing `P(n=0) = Π(1−pᵢ)` from independent pairwise
probabilities **systematically underestimates singleton probability for exactly the entities that
are singletons**. A small entity-level head reading both sources' best scores fixes it.

---

## 4. The test set is denser than train — unresolved, and it matters

**verified** arithmetic, **unresolved** interpretation.

| | train | test | change |
|---|---|---|---|
| corpus records per S1 entity | 4.68 | 5.75 | **+23%** |
| — US | 4.674 | 5.756 | +23.2% |
| — India | 4.680 | 5.824 | +24.4% |
| — France | — | 5.531 | — |

Uniform across countries, so **not a France artefact**. Two readings, both consistent with the
row counts:

- **A — distractor budget fixed:** test mean cardinality rises to ~4.26, distractor rate stays ~26%.
- **B — cardinality fixed:** test distractor rate rises to ~40%, mean stays 3.46.

**These give opposite threshold advice.** A says accept more; B says accept considerably less.

> A research pass proposed resolving this from per-country record ratios. That does not work —
> the ratio is the same measurement restated, and both hypotheses predict it equally. It also
> does not discriminate via S2/S3 distractor symmetry, since both hypotheses preserve that.
> **The honest position: unresolved.** It needs a score-profile diagnostic (compare the
> distribution of top-K scores per entity between train and test under one fixed scorer) or two
> threshold-differing submissions. Until resolved, do **not** calibrate to the train count histogram.

Risk is asymmetric under F₀.₅: being too strict costs less than being too loose.

---

## 5. Blocking — architecture and operating point

### Recall is cheaper than it looks

Under a perfect matcher, macro-F₀.₅ ceiling vs per-link candidate recall:

| recall | 0.80 | 0.85 | 0.90 | 0.95 | 0.99 | 1.00 |
|---|---|---|---|---|---|---|
| ceiling | 0.923 | 0.947 | 0.967 | 0.984 | 0.997 | 1.000 |

The frontier is **flat above ~95%**. Chasing 95→99 buys ~1.3 ceiling points and costs roughly 3×
the candidates, which degrades matcher precision — weighted 2×. **Target 95–97%, not 99%.**

### The cost equation governs every design choice

```
ops = Σ_keys  df_query(k) · df_corpus(k)
```

This disqualifies, on cost rather than taste: **Soundex/Metaphone** (4-char keyspace ⇒ df in the
thousands; also encodes *English* phonology, meaningless on Devanagari and wrong for French),
**uncapped char-3gram TF-IDF** (~1e12 ops), **sorted-token prefix keys** (alphabetically biased
toward common tokens), and **MinHash/LSH** (delivers what an IDF-weighted inverted index already
gives, but probabilistically and without IDF).

### Recommended channels, unioned (never intersected)

| channel | what it uniquely reaches |
|---|---|
| C1 canonical-name exact key | cheap, and a normaliser sanity check |
| **C2 IDF-weighted rare name token, BOTH directions** | backbone; order-invariant |
| C3 rarest-token-pair | precision lane, rare by construction |
| C4 df-capped char 4-gram | within-token corruption (typos, agglutination) |
| **C5 postal × house-number** | ★ pairs whose names share *nothing* |
| C6 df-capped rare address token | locality overlap when postal is missing |
| C7 initialism/acronym | acronym↔expansion; nothing else reaches these |
| C8 script folding / romanisation | cross-script, feeds C2/C3/C4 as extra keys |
| C9 digit-multiset | postal-free fallback |
| C11 cap-raising escalation | targeted rescue for starved entities |

Two structural ideas worth more than the channel list:

1. **Index the small side too.** Every corpus record has ≤1 correct S1 answer, so streaming the
   9.97M corpus *as queries* against a 1.73M S1 index is 5.75× smaller, *cheaper* to query, and
   guarantees every record gets its shot — including the 265,506 empty-address records that the
   forward direction starves. The two directions retrieve different pairs. Run both.
2. **Fit df/IDF transductively on the TEST corpus.** Unsupervised, legal, free — and it is the
   single thing that makes France work. French legal forms (SARL, SAS, SCI, EURL), stopwords and
   street generics surface as high-df tokens and get suppressed automatically, **with nobody on
   the team knowing French or downloading a list.** Mine the suffix/stopword lists; never author them.

**Target operating point: 95–97% per-link recall at ≤60 candidates/entity (mean ~40).**
Peak RAM ≈ 4.4 GB for the India shard — memory is *not* the constraint once you avoid Python
dicts, fp32 dense vectors and uncapped n-gram matrices. Wall clock is.

### Dense embeddings: NO, conditionally

Eleven of the twelve documented noise types are **orthographic**, not semantic. Sparse channels
cover them deterministically with no GPU and no install risk. Dense buys exactly one thing:
semantic equivalence across translation (`व्यापार` → "vyapar" ≠ "Trading"), whose size is
**unmeasured**. Secondary argument: general encoders optimise *semantic* similarity, and
"State Bank of India, Andheri" vs "…, Bandra" are semantically near-identical and are
**different entities** — a dense channel preferentially retrieves same-chain-different-branch
records, which is actively harmful under a 2×-precision metric.

Gate: measure the cross-script residue after romanisation. <1.5% of links → no dense, final.
>4% → build it for the India shard only, starved entities only.

---

## 6. Signal — what distinguishes a true match from a hard distractor

Random negatives are trivially separable (name token Jaccard 0.635 vs 0.031; address shares a
number 78% vs 1.8%). **Model selection must therefore be driven by hard negatives, not by these gaps.**

**The 7.6M aligned pairs are a supervised corruption corpus** and are fully legal to mine.
Learnable, not hand-writable: abbreviation tables per country, state/region codes, character edit
costs, Devanagari↔Latin grapheme mapping, junk-affix vocabulary (`<<`, `--`), token-drop rates,
permutation-type histogram, per-source corruption signatures.

The precision insight: **the transformations the generator CANNOT perform matter more than the
ones it can.** An induced abbreviation table is primarily a *refusal* device — it rejects
`Corp ↔ Cooperative`, `Pvt ↔ Inc`, `41St Cross ↔ 42Nd Cross`. Similarity metrics are symmetric in
their ignorance; a grammar is asymmetric in its refusals.

### Hard negatives are exactly labelled, at unlimited scale

The zero-re-use property means for any record `r` matched to entity `e`, `(e', r)` is a
**guaranteed** negative for every `e' ≠ e` — zero label noise. Priority order:

1. **Blocker-conditioned** — top-K retrieved minus gold. Re-mine whenever the blocker changes.
2. **Sibling negatives** — the gold matches of `e`'s nearest *other* S1 entity. Textually
   near-identical, semantically distinct; strictly harder than any distractor.
3. **Same-normalised-name S1 pairs** — certified different by deduplication (39.28% of entities).
4. Distractors, after measuring whether they are adversarial or filler.

A model trained on random negatives learns name matching. A model trained on these learns
**numeral matching** — which is what actually separates chain outlets.

### Normalisation policy: keep parallel views, never collapse

Hard rejects, each of which manufactures false merges: **legal-suffix deletion** (S1 is
deduplicated, so `Smith Ltd` and `Smith Pvt Ltd` are different businesses), **stemming/Soundex**,
**generic-token deletion** (down-weight via block-conditional IDF instead — weighting is
reversible, deletion is not), and above all **digit removal from addresses**, which destroys the
single highest-precision signal in the dataset.

Mined abbreviations must be applied as a *probabilistic equivalence*, never a destructive rewrite.

---

## 7. Global structure: normalise, don't assign

Global assignment is a fantasy at this scale — Hungarian is O(V³) ≈ 5e18; min-cost flow on 52M
edges is not happening in Python in 51 hours. Connected-component decomposition **provably fails**:
bipartite percolation at top-30/entity gives an excess-degree product of 121.8 ≫ 1, so a giant
component swallows the graph. You cannot prune hard enough to decompose without destroying recall.

What *is* tractable and captures most of the value:

- **Sinkhorn capacity normalisation** — column capacity 1 (the measured disjointness), row budget
  from a cardinality estimate. Converts disjointness from an ignored constraint into a calibration
  signal on every edge: a record wanted by eleven entities is worth ~0.06 to each. Fully
  unsupervised, so it **transfers to France by construction rather than by hope**. ~2 min/shard.
  *Critical implementation risk:* 26–40% of columns are distractors that must receive zero mass.
  Without a correctly-sized null sink, Sinkhorn forces distractor mass onto entities and
  manufactures false positives at scale.
- **Marginal-F conflict exchange** — any record claimed by ≥2 entities goes to whichever gains the
  most expected F. Single pass, every swap F-monotone by construction. The lazy 90% of assignment.

---

## 8. Leaderboard context (as of 25 Sep, ~20:45 IST)

Top scores: **0.986955 / 0.986421 / 0.985884**, with ranks 4–8 clustered at 0.984–0.985.

This is decisive for strategy. The board is **saturated near the ceiling**, and the spread from
rank 1 to rank 8 is ~0.003. Implications:

- The problem is far more separable than a hard ER benchmark. Our own audit agrees — true vs
  random separation is enormous.
- The 0.810 fixed-k ceiling is **not** binding for the leaders. With confident (near 0/1)
  probabilities, a simple threshold recovers the right cardinality automatically. The cardinality
  headroom is real but is already being captured.
- **A competent baseline is not competitive.** The differentiator is the last ~1.5%: chains,
  transliteration, generic names, and singleton precision.
- With 5 submissions/day and this compression, blind threshold search is not viable. Offline
  validation must be trustworthy.

---

## 9. What is gated on what

| Decision | Gated on | Cost to resolve |
|---|---|---|
| Dense embeddings at all | cross-script residue after romanisation | 20 min |
| Sinkhorn (6 h) vs conflict exchange (2 h) | contested-pair rate ≥15% at ≥2× FP rate | 20 min |
| Any component-based method | largest connected component size | 30 min |
| Count-matched calibration | hypothesis A vs B (§4) | needs a scorer |
| Singleton head wiring | precision ≥0.55 at recall 0.5 (break-even 0.474) | 2 h |
| Negative-set composition | are distractors adversarial or filler? | 30 min |

Every one of these costs under an hour and saves six.
