"""F_0.5 metric and the decision rule derived from it.

The challenge scores macro-averaged F_0.5 per Source 1 entity. With c correct,
k predicted and n true, the definition collapses to an exact closed form:

    F_0.5 = 1.25*c / (0.25*n + k)        (= 1.0 when n == 0 and k == 0)

Differencing that gives the optimal stopping rule: accepting one more candidate
with calibrated probability p raises expected F_0.5 iff p > 0.8 * F_k. The bar
rises as the set improves, so a confident subset beats a greedy superset, and
because p is non-increasing while the bar is non-decreasing there is a single
crossing -- stop at the first rejection.

Run this module directly to verify the implementation against the perfect-ranker
oracle ceilings. If those numbers do not reproduce, every downstream experiment
is invalid.
"""
import numpy as np

# E[n | n >= 1] measured on train ground truth (7,638,365 / 2,083,574).
# Used for the k=0 -> k=1 decision, where the marginal rule is singular.
E_N_GIVEN_MATCHED = 3.666
FIRST_ACCEPT_FACTOR = 1.25 * (1.0 / (0.25 * E_N_GIVEN_MATCHED + 1.0))  # ~0.652


def f05(c, n, k):
    """Per-entity F_0.5. Array-safe."""
    c, n, k = np.asarray(c, float), np.asarray(n, float), np.asarray(k, float)
    denom = 0.25 * n + k
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom > 0, 1.25 * c / np.where(denom > 0, denom, 1.0), 0.0)
    return np.where((n == 0) & (k == 0), 1.0, out)


def macro_f05(predicted: dict, truth: dict) -> float:
    """predicted/truth: {s1_id: set of matched ids}. Scored over truth's keys,
    so an entity we failed to emit correctly counts as an empty prediction."""
    scores = np.empty(len(truth))
    for i, (eid, true_set) in enumerate(truth.items()):
        pred = predicted.get(eid, ())
        scores[i] = f05(len(true_set & set(pred)), len(true_set), len(pred))
    return float(scores.mean())


def choose_k(probs: np.ndarray, p_zero: float) -> int:
    """Number of candidates to accept, maximising expected F_0.5.

    probs: calibrated match probabilities, sorted descending.
    p_zero: P(this entity has no matches at all).
    """
    if probs.size == 0:
        return 0
    # k=0 -> k=1 is singular (F_0 = p_zero, not 1.25c/(0.25n+0)). Accept the
    # first candidate iff its expected contribution beats abstaining outright.
    if probs[0] * FIRST_ACCEPT_FACTOR <= p_zero:
        return 0

    n_hat = max(probs.sum(), 1e-9)   # expected true cardinality
    c = 0.0
    for k in range(probs.size):
        if k > 0 and probs[k] <= 0.8 * (1.25 * c / (0.25 * n_hat + k)):
            return k
        c += probs[k]
    return probs.size


def _self_test():
    """Reproduce the perfect-ranker fixed-k ceilings from the real ground truth."""
    import pathlib
    import pyarrow.parquet as pq

    gt = pq.read_table(pathlib.Path("data/parquet/train_ground_truth.parquet"))
    n = np.array([0 if not v else v.count(",") + 1
                  for v in gt.column("matched_entity_ids").to_pylist()], dtype=np.int64)
    print(f"entities={n.size:,}  mean n={n.mean():.4f}  "
          f"singletons={(n == 0).mean():.4%}  E[n|n>=1]={n[n > 0].mean():.4f}")

    print("\nperfect-ranker fixed-k ceilings (predict exactly k, all correct while available):")
    expected = {2: 0.7929, 3: 0.8102, 4: 0.7604}
    ok = True
    for k in range(1, 7):
        got = f05(np.minimum(n, k), n, k).mean()
        tag = ""
        if k in expected:
            tag = f"   expected {expected[k]:.4f}"
            if abs(got - expected[k]) > 5e-4:
                tag += "  <-- MISMATCH"
                ok = False
        print(f"  top-{k}: {got:.4f}{tag}")

    always_empty = f05(np.zeros_like(n), n, np.zeros_like(n)).mean()
    print(f"  always predict []: {always_empty:.4f}   expected {(n == 0).mean():.4f}")
    oracle = f05(n, n, n).mean()
    print(f"  oracle k:          {oracle:.4f}   expected 1.0000")
    ok &= abs(oracle - 1.0) < 1e-9

    # Spot-check the PDF's worked example: n=2, k=3, c=2 -> 0.714
    ex = float(f05(2, 2, 3))
    print(f"\nPDF worked example (n=2,k=3,c=2): {ex:.4f}   expected 0.7143")
    ok &= abs(ex - 5 / 7) < 1e-4

    print("\n" + ("PASS" if ok else "FAIL -- do not proceed until this passes"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
