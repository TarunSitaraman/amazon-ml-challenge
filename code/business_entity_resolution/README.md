# Business Entity Resolution — reproduction guide

End-to-end: raw TSV → blocking → matching → entity-level decision → submission files.

No external data is used anywhere. Every table the pipeline relies on (document
frequencies, IDF weights, vocabulary) is derived from the provided files at run
time. The only hand-written table is a Devanagari→Latin codepoint map in
`src/textnorm.py`, which is alphabet knowledge from the Unicode chart — the same
kind as knowing `St` abbreviates `Street` — not a gazetteer, registry or
downloaded dataset.

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

# 3. Train the pairwise matcher and score the decision layer offline.
#    Writes model.pkl.
python src/train_eval.py India 12000 6000

# 4. Generate the submission files into output/ (~3 h for the full test set).
#    Each finished country is saved to output/partial_<country>.tsv, so after
#    a crash, rerunning the same command skips the finished countries.
#    --fresh discards the partials and starts over.
python src/predict.py

# 5. Check the documented format rules locally, then run the organisers'
#    validator as the authoritative check.
python src/validate.py
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

All commands are run from this directory, with `src/` on `PYTHONPATH`.

## Tuning knobs

The document-frequency cap is the cost knob (`ops = Σ df_query · df_corpus`) and
also the dominant recall constraint, so it is environment-overridable:

```bash
DF_CAP=2000 ADDR_DF_CAP=10000 TOP_K=40 ADDR_TOP_K=60 python src/predict.py
```

`src/diag_dfcap.py` measures the recall/cap frontier directly.

## Module map

| file | role |
|---|---|
| `prepare_data.py` | TSV → country-partitioned Parquet |
| `textnorm.py` | normalisation, Devanagari transliteration, folded and skeleton views |
| `metric.py` | F_0.5 closed form, the stopping rule, and the oracle self-test |
| `blocking.py` | five-channel candidate generation, country-sharded |
| `features.py` | per-channel similarities plus entity-level context |
| `train_eval.py` | trains the matcher, calibrates, scores the decision layer |
| `predict.py` | full test run, emits both submission files |
| `validate.py` | local format check |
| `audit_*.py`, `diag_dfcap.py` | the Stage 0 measurements behind the design |

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
