# Business Entity Resolution: Solution Documentation

Every number below is measured on the provided files and is traceable to
[AUDIT.md](AUDIT.md), [RESEARCH.md](RESEARCH.md) or the commit message that
introduced the change (commit hashes given in brackets). Where a number comes
from a synthetic self-test rather than the competition data, it says so. Where
something has not been measured, it says that too.

Code: `code/business_entity_resolution/`. Reproduction steps are in its
`README.md`. No external data is used; every document frequency, IDF weight and
vocabulary is derived from the provided files at run time.

---

## 1. Methodology

### 1.1 Measure first, then design

Before any matching code was written, all 24M records were converted to
country-partitioned Parquet and audited [5a6d632]. Two structural facts from
that audit determined the architecture.

**Zero S2/S3 record re-use.** The training ground truth has 7,638,365 links over
7,638,365 *distinct* S2/S3 records. Every S2/S3 record belongs to at most one S1
entity. The task is therefore constrained assignment, not independent pair
classification: a candidate's score only means something relative to its best
rival S1 entity.

What was built on it:
- `features.py` gives the matcher entity-level context (rank, gap and ratio to
  the top candidate, candidate count) rather than scoring pairs in isolation.
- `disjoint.py` [278f04d] enforces one owner per record after the per-entity
  decision. `resolve_conflicts` gives a contested record to the entity with the
  largest expected F_0.5 gain, or to nobody if every gain is non-positive.
  `sinkhorn_normalise` normalises scores with a per-record capacity of 1 plus a
  null sink, because 1,340,997 S2 records and 1,340,857 S3 records match
  nothing and must be able to receive zero mass. Both are
  wired into `predict.py --disjoint off|resolve|sinkhorn`, default `off`.
- `analyze_submission.py` [ca9d9d9] counts records predicted for two entities.
  Under zero re-use each such record is guaranteed wrong for at least one of
  them, so the count is a direct error measurement that needs no labels.

**Zero cross-country links.** 0 of 7,638,365 links cross a country. Country is
therefore used as a hard block: each country is an independent job, peak memory
is one country shard, and France (15.0% of test S1, 259,452 entities, with no
training labels) is self-contained.

### 1.2 The metric, in closed form

F_0.5 = 1.25·c / (0.25·n + k), with c correct, k predicted, n true. This
reproduces the problem statement's worked example (n=2, k=3, c=2 gives 0.714).
`metric.py` asserts it and reproduces the perfect-ranker ceilings from the real
ground truth (top-2 0.7930, top-3 0.8102, top-4 0.7605, oracle 1.0000)
[69e9af9], so a broken metric cannot silently invalidate later experiments.

Differencing that formula gives the decision rule: adding a candidate with
calibrated probability p raises expected F_0.5 iff p > 0.8·F_k. The bar never
exceeds 0.80 and rises as the set improves, so one greedy pass over candidates
sorted by p is optimal. The k=0 to k=1 step is special-cased: accept the first
candidate iff p₁ > 1.533·P̂(n=0), where 1.533 comes from the measured
E[n | n≥1] = 3.666.

Why cardinality gets its own layer: a perfect ranker with the best fixed cut-off
(top-3) caps at 0.8102 macro F_0.5. On India validation, the adaptive rule beats
the best fixed-k policy by +0.1361 [5d8fcb2].

### 1.3 Pipeline

1. Blocking per country shard (section 2) produces up to 60 candidates per S1
   entity (`CAND_CAP = 60`).
2. A LightGBM binary classifier scores each pair (section 3).
3. Isotonic regression calibrates the scores, since the decision rule consumes
   probabilities, not ranks.
4. `choose_k` applies the stopping rule per entity, corrected for blocking
   recall (`--recall`; n in the metric is the true count, which includes
   matches blocking never retrieved [07fb186]).
5. Optionally, `disjoint.py` resolves records claimed by several entities.
6. Output is written with explicit `\n` line endings and checked by
   `validate.py` (section 4.4).

### 1.4 Validation protocol

`train_eval.py` samples S1 entities per country and splits them into train and
validation entities. Isotonic calibration is fit on the first half of the
validation entities and metrics are reported only on the second half, so the
reported numbers are out-of-sample for both model and calibrator [278f04d].
Per-candidate probabilities are dumped to `valstate.pkl`, so `tune_decision.py`
can sweep decision rules in about a second instead of a six-minute re-blocking
[07fb186]. Decision rules are tuned offline against this file, not on the
leaderboard.

---

## 2. Candidate generation (blocking)

### 2.1 Reachability is not the problem, cost is

Over 219,534 sampled true links, the union of cheap channels reaches 100.00%.
No true link is unreachable, so blocking is an engineering problem of retrieving
them cheaply. Per channel:

| channel | recall (all) | US | India | script mismatch |
|---|---|---|---|---|
| name token | 85.74% | 92.17% | 76.06% | 7.08% |
| name 4-gram | 90.78% | 97.65% | 80.43% | 6.75% |
| address number | 78.56% | 76.26% | 82.02% | 82.85% |
| address token | 95.57% | 95.27% | 96.01% | 99.88% |
| union | 100.00% | 100.00% | 100.00% | 100.00% |

Name and address are complementary: name channels fall to about 7% when the two
names are in different scripts, while address holds at 99.88%. India is
materially harder on names (name 4-gram 80.43% against US 97.65%).

The cost of any key-based channel is ops = Σ over keys of df_query·df_corpus,
which is why common keys are capped by document frequency (df).

### 2.2 Channels in the current pipeline

Five channels, unioned, never intersected. Each emits its own score, so the
matcher sees them as separate features.

| channel | key | why |
|---|---|---|
| C1 canonical name | exact normalised name, df cap 5 | cheap; a name shared by more entities is a chain |
| C2 name tokens | IDF-weighted inverted index, cosine, top 40 | backbone |
| C6 address tokens | IDF-weighted inverted index, cosine, top 60 | reaches records whose name is in another script |
| C5 postal × house number | composite key | the only channel independent of the name |
| C3 rarest token pair | packed pair of the 3 rarest tokens, pair df cap 50 | reach without raising the df cap (2.4) |

Document frequency is fitted on the corpus being searched, which at inference
time means transductively on the test corpus. That is unsupervised, uses only
provided files, and lets high-frequency French legal forms and street words be
down-weighted without anyone authoring a French list.

### 2.3 Measured progression (India train, per-link recall)

Each step is a single change, measured in isolation [69e9af9]:

| change | per-link recall |
|---|---|
| name channels only | 32.65% |
| + address token channel | 71.83% |
| + raising the df caps | 83.66% |
| + cosine scoring instead of raw IDF sum | 87.04% (macro F_0.5 ceiling 0.9381) |

The address channel was the single largest step, consistent with the audit's
finding that address is what survives script mismatch. In the same commit,
Devanagari text was found to be stripped to empty by the normaliser, blanking
16.9% of the corpus; transliteration with schwa deletion cut that to 7.3%.

87.04% is below the 95–97% operating point RESEARCH.md §5 derives (the
perfect-matcher ceiling is 0.923 at 80% recall, 0.984 at 95%, 0.997 at 99%).
Blocking recall is still a known gap.

### 2.4 The df-cap frontier, and why C3 instead of a higher cap

`diag_dfcap.py` showed that the df cap, not reachability, was the binding
constraint [69e9af9]:

| address df cap | links reachable |
|---|---|
| 500 | 78.92% |
| 10,000 (current `ADDR_DF_CAP`) | 96.31% |
| 50,000 | 99.39% [6295599] |
| uncapped | 100% |

Raising the address cap from 10,000 to 50,000 multiplies retrieval work about
5×, taking the India shard from 75 minutes to about 6 hours [6295599]. C3 gets
at the same records differently: a pair of mid-frequency tokens is rare by
construction. In its self-test, "sharma" (df 36) and "medical" (df 32) are each
too common to index affordably, but the pair has df 2. The key is an exact
packed id pair, so it cannot collide, and rarity is ranked by corpus df on both
sides so query and corpus keys line up.

Not yet measured: C3's marginal recall on the real India shard. Adding it
changed the feature count from 26 to 27, so `model.pkl` must be retrained
before it can be scored.

---

## 3. Model and features

### 3.1 Matcher

LightGBM, binary objective, learning rate 0.06, 63 leaves, min 50 per leaf,
feature fraction 0.9, bagging 0.8, 350 rounds. Training pairs are the blocker's
own candidates for sampled training entities, labelled from the ground truth,
so the negatives are exactly the hard negatives the matcher meets at inference.
Training can span several countries; each country's corpus is freed before the
next is loaded [5d8fcb2].

Why hard negatives: against random same-country records, true pairs are
trivially separable (name token Jaccard 0.635 against 0.031; address shares a
number 77.98% against 1.79%). Those gaps say nothing about the real difficulty,
which is distinguishing a true match from a near-identical record of a
different entity.

### 3.2 Features (27)

**Blocking geometry (14, `features.py`):** the five channel scores, number of
channels that fired, max and sum of channel scores, rank within the entity,
number of candidates, gap and ratio to the top candidate, source (S2/S3), and
`name_dup`.

**String similarity (13, `strfeatures.py`):** name token Jaccard, containment,
char 4-gram similarity, exact match, length ratio, IDF-weighted rare shared
tokens; address token Jaccard and containment; digit-token Jaccard, shared digit
count, both-have-digits; empty-address flags for each side.

**Measured effect:** before `strfeatures.py` the matcher had only blocking
geometry and could not tell whether two names resembled each other. Adding it
took India validation macro F_0.5 from **0.6058 to 0.8615**, against a blocking
ceiling of 0.9345 in that run [5d8fcb2]. The top four features by gain became
`ad_cont`, `ad_jac`, `nm_4gram`, `dg_jac`, three of them address or digit
features.

### 3.3 The 39.28% name-collision rate and what it forbids

866,894 S1 entities (39.28%) share a normalised name with a different S1 entity
(largest groups: "primary care group" ×253, "ear nose throat group" ×251). S1 is
deduplicated, so these are provably different businesses, mostly chain outlets.
For them the name carries no identity and the address is the only
discriminator. Consequences:

- **No legal-suffix stripping.** "Smith Ltd" and "Smith Pvt Ltd" can both exist
  in a deduplicated S1. Deleting the suffix merges them. Identity-poor tokens
  (`limited` in 23.67% of S1 names, `private` 19.60%, `llc` 16.12%) are
  down-weighted by IDF instead, which is reversible.
- **No address-digit removal.** Digits are what separate two branches of one
  chain. Digit features are kept separate from the general address score, and
  the `strfeatures.py` self-test asserts that two branches of one chain are
  separated by digits, not by name [5d8fcb2].
- **`name_dup` feature** [67bb34c]: the number of other S1 entities sharing this
  entity's name, computed on each split's own S1 without labels, so the matcher
  can learn where to discount name similarity.

Caveat: the 39.28% was measured with the earlier normaliser. The acronym fix in
[8308705] makes more names collapse to the same form, so 39.28% is now a slight
underestimate.

---

## 4. Other relevant information

### 4.1 Measured and removed

| tried | measured | outcome |
|---|---|---|
| Fold and consonant-skeleton name channels | +0.06% marginal recall; the apparent per-channel attribution was overlap with the address channel. They cost two extra indices that pushed the India shard into swap. | Removed [5d8fcb2]; kept out when C3 was cherry-picked from a branch that still had them [6295599] |
| Every address digit run as a blocking key | +0.01% recall for 40 minutes of wall clock | Cut; C5 keeps only the postal\|number composite [69e9af9] |
| `redecide` in `resolve_conflicts` (a losing entity re-runs its decision over remaining candidates) | Flat to slightly worse on the synthetic self-test, likely because a loser's next-best candidates are mostly distractors | Kept as an option, off by default (`disjoint.py`) |
| Blocking-recall correction in `choose_k` | Only flips a decision below about 0.6 recall | Kept for correctness; recorded as small at the current 0.87 recall [07fb186] |

`redecide` and the `--disjoint` modes have only been measured on synthetic data;
the real-data comparison has not been run yet, which is why `--disjoint`
defaults to `off`.

### 4.2 Unresolved: train/test density

| | train | test | change |
|---|---|---|---|
| corpus records per S1 entity | 4.68 | 5.75 | +23% |
| US | 4.674 | 5.756 | +23.2% |
| India | 4.680 | 5.824 | +24.4% |

It is uniform across countries, so it is not a France artefact. Two
explanations fit the row counts equally well:

- **A. Distractor budget fixed:** test mean cardinality rises to about 4.26 and
  the distractor rate stays about 26%. Correction: accept more.
- **B. Cardinality fixed:** mean stays 3.46 and the distractor rate rises to
  about 40%. Correction: accept considerably less.

They imply opposite threshold corrections, and the row counts cannot separate
them. `analyze_submission.py` [ca9d9d9] compares mean predicted k under the same
model and stopping rule on train and test, which can lean toward one reading;
that comparison needs a full test run and has not been reported. Until it is,
thresholds are not calibrated to the train cardinality histogram. Under F_0.5,
being too strict costs less than being too loose.

### 4.3 Singleton probability

5.58% of S1 entities have no match; for them predicting nothing scores 1.0 and
anything else scores 0. Zero-match is correlated across sources at 3.55× what
independence predicts (123,247 entities with no match in either source against
about 34,720 expected), so the product rule Π(1−p) underestimates P(n=0) for
exactly the entities that have n=0. `singleton.py` [45c413d] is an entity-level
head for P(n=0). On synthetic data, mean P(n=0) for true singletons is 0.5661
from the head against 0.2575 from the product rule, and end-to-end macro F_0.5
is 0.8179 against 0.8028 (flat prior 0.7905). It is not yet in the pipeline: it
needs out-of-fold pair scores for training entities, because in-sample scores
are overconfident on exactly the entities it learns from.

### 4.4 Unseen country and format safety

- **France has no labels.** Document frequencies are fitted transductively per
  country, and `name_dup` needs no labels. A normaliser bug was found by testing
  French-shaped records: "S.A.R.L." normalised to "s a r l" while "Sarl" became
  "sarl", dropping name Jaccard from 1.0 to 0.38 on the same legal form. Fixed
  and verified at 1.00 Jaccard on S.A.R.L./Sarl, S.A.S/SAS, L.L.C./LLC and
  P.C./PC [8308705].
- **Script asymmetry.** S1 contains 0 Devanagari names; S2 has 269,424 (13.35%
  of its India rows) and S3 158,003 (7.47%). Transliteration is only ever
  needed from Devanagari to Latin. The one hand-written table is a
  Devanagari→Latin codepoint map from the Unicode chart.
- **CRLF.** Writing output with `pathlib.write_text` on Windows would produce
  `\r\n`, making every trailing ID read as "S3-123\r". `validate.py` now checks
  raw bytes, since Python text-mode reads hide the `\r` [fe39139]. The
  organisers' validator arrived as a 0-byte file, so `validate.py` re-implements
  the documented rules locally [69e9af9].

### 4.5 Considered and not built (reasoned, not measured)

Soundex/Metaphone, uncapped char-3gram TF-IDF, sorted-token prefix keys and
MinHash/LSH were rejected on the cost equation in section 2.1, not on a
measurement (RESEARCH.md §5). Dense embeddings were gated on the cross-script
residue after romanisation; that residue has not been measured, and no dense
model is used.

### 4.6 Current standing and known gaps

- Last measured India validation macro F_0.5: 0.8615, blocking ceiling 0.9345
  [5d8fcb2]. That model predates C3; the model has to be retrained on the
  five-channel blocker before a new number exists.
- Blocking recall (87.04%) is below the 95–97% target.
- Not yet measured on real data: C3 marginal recall, `--disjoint`
  resolve/sinkhorn, `redecide`, the singleton head, and the density diagnostic.
- Runtime: about 2.3 to 3 hours for a full test run on CPU, which is why
  feature changes are batched into one run [67bb34c].
