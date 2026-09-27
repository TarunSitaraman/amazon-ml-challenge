# Business Entity Resolution — reproduction guide

End-to-end: raw TSV → blocking → matching → entity-level decision → submission files.

No external data is used anywhere. Every table the pipeline relies on (document
frequencies, IDF weights, vocabulary) is derived from the provided files at run
time. The only hand-written table is a Devanagari→Latin codepoint map in
`src/textnorm.py`, which is alphabet knowledge from the Unicode chart — the same
kind as knowing `St` abbreviates `Street` — not a gazetteer, registry or
downloaded dataset. The transliteration that improves on it
(`src/mine_translit.py`) is learned from the training pairs alone.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.12. CPU only — no GPU is used or needed.

## Run

```bash
# 1. Convert the raw TSVs to country-partitioned Parquet (~5 min).
#    Point --raw-dir at the directory holding the seven provided .tsv files.
python src/prepare_data.py --raw-dir <dir> --out-dir data/parquet

# 2. Verify the metric implementation against the perfect-ranker oracle
#    ceilings. If this does not print PASS, stop: every later number is invalid.
python src/metric.py

# 3. Learn the transliteration and the corruption grammar from the training
#    pairs (both read norm() output, so translit comes first). Copies of the
#    two files used for the final submission are shipped in data/.
python src/mine_translit.py                     # -> data/translit_model.json
python src/mine_corruption.py --sample 300000   # -> data/corruption_grammar.json

# 4. Train the pairwise matcher, calibrate it, train the singleton head and
#    score the decision layer offline (India+US, 15k train / 6k validation
#    entities each). Writes model.pkl. Final model: 0.9342 held-out
#    macro F0.5 (singleton head on), blocking ceiling 0.9785.
REL_FEATS=1 C4=1 CAND_CAP=150 python src/train_eval.py India,US 15000 6000

# 5. Generate the submission files into output/ (~2.5 h for the full test set
#    on a 16 GB laptop). Each finished country is saved to
#    output/partial_<country>.tsv, so after a crash, rerunning the same command
#    skips the finished countries. --fresh discards the partials.
C4=1 CAND_CAP=150 python src/predict.py --fresh --singleton on --disjoint resolve

# 6. Check the documented format rules locally, then run the organisers'
#    validator as the authoritative check.
python src/validate.py
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

All commands are run from this directory, with `src/` on `PYTHONPATH`.
`CAND_FEATS` (candidate-record features, `src/candfeatures.py`) is on by
default. `TRI=1` (second-stage triangulation, `src/triangulate.py`) is off:
it measured -0.0011.

## Tuning knobs

The document-frequency cap is the cost knob (`ops = Σ df_query · df_corpus`) and
also the dominant recall constraint, so it is environment-overridable:

```bash
DF_CAP=2000 ADDR_DF_CAP=10000 TOP_K=40 ADDR_TOP_K=60 python src/predict.py
```

`src/diag_dfcap.py` measures the recall/cap frontier directly.

## Per-country calibration

France has no training labels and abstains on about twice the US rate on the
test set. `--calibrate France` rescales France's P(n=0) and stopping bar so its
predicted cardinality (singleton rate, mean k among matched entities) matches
the pooled prediction of the other countries on the same run. Off by default;
the other countries' rows are never changed and their partials are reused. See
`src/calibrate.py` for why this is defensible, and run it directly for its
synthetic self-test.

```bash
python src/predict.py --calibrate France
python src/predict.py --calibrate France --calibrate-ref US   # US as the only reference
```

## Profiling

Add `--profile` to `predict.py` or `train_eval.py` to print, per country and
overall, the wall clock of each stage (normalise, each index build, each
blocking channel's query, cap_candidates, strfeatures.build, features.build,
model.predict, isotonic, choose_k, disjoint, TSV write, ...) with its share of
the total, throughput (queries/s for blocking, pairs/s for features and
scoring) and the process peak RSS. Off by default; see `src/profiling.py`.

```bash
python src/predict.py --disjoint resolve --profile
python src/train_eval.py India 15000 0 --profile
```

## Module map

| file | role |
|---|---|
| `prepare_data.py` | TSV → country-partitioned Parquet |
| `textnorm.py` | normalisation; Indic transliteration (Bengali..Malayalam are shifted onto Devanagari first) |
| `strfeatures.py` | string similarity and corruption-grammar features |
| `candfeatures.py` | properties of the candidate record itself (case, junk affixes, S2/S3 twin) |
| `singleton.py` | entity-level P(no match) head (`--singleton on`) |
| `disjoint.py` | one owner per S2/S3 record (`--disjoint resolve`) |
| `metric.py` | F_0.5 closed form, the stopping rule, and the oracle self-test |
| `blocking.py` | five-channel candidate generation, country-sharded |
| `features.py` | per-channel similarities plus entity-level context |
| `train_eval.py` | trains the matcher, calibrates, scores the decision layer |
| `predict.py` | full test run, emits both submission files |
| `calibrate.py` | per-country transductive calibration of the decision (`--calibrate`) |
| `validate.py` | local format check |
| `mine_translit.py` | learns Devanagari → Latin transliteration (a word lexicon, per-grapheme spelling, schwa and anusvara decisions) from the aligned training pairs into `data/translit_model.json`, which `textnorm.norm` then uses; without the file the rule transliteration is used unchanged. Run it before `mine_corruption.py` and `train_eval.py`, since both read `norm()` output; `predict.py` refuses a `model.pkl` trained under a different file (`ALLOW_TRANSLIT_MISMATCH=1` overrides) |
| `mine_corruption.py` | learns the generator's corruption grammar (abbreviations, forbidden pairs, junk affixes, drop and reorder rates) from the aligned training pairs; `--sample N` for a subset, `--self-test` for the synthetic check |
| `audit_*.py`, `diag_dfcap.py` | the Stage 0 measurements behind the design |
| `diag_misses.py`, `diag_matcher.py` | why links are lost: `diag_misses` for links blocking never retrieves, `diag_matcher` for the matcher's errors on retrieved candidates (reads `valstate.pkl`, splits the lost macro F0.5 by error category) |

## Design notes

The measurements that drove the architecture are in `../../AUDIT.md`, and the
reasoning in `../../RESEARCH.md`. The three that mattered most:

- **Every S2/S3 record matches at most one S1 entity** (7,638,365 links over
  7,638,365 distinct records). This is constrained assignment, not independent
  pair classification.
- **No true link crosses a country boundary** (0 of 7,638,365), so country is a
  safe hard block and each country is an independent job.
- **39.28% of S1 entities share a normalised name with a different S1 entity.**
  S1 is deduplicated, so those are provably distinct businesses — which is why
  the normaliser never strips legal suffixes and never deletes address digits.
