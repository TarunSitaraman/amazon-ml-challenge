"""String-similarity features for candidate pairs.

The blocking-geometry features tell the matcher how a pair was *retrieved*, not
whether the two records look alike. These supply the latter.

Weighting note, from the audit: 39.28% of S1 entities share a normalised name
with a DIFFERENT S1 entity (chains, franchises -- S1 is deduplicated, so they
are provably distinct businesses). For that population the name is useless and
the address digits are the only discriminator: true pairs share an address digit
token 78% of the time against 1.8% for random pairs. Hence the digit features
are kept separate and explicit rather than folded into a general address score.
"""
import numpy as np

from textnorm import ngrams, norm

NAMES = ["nm_jac", "nm_cont", "nm_4gram", "nm_exact", "nm_lenratio",
         "ad_jac", "ad_cont", "dg_jac", "dg_shared", "dg_both_have",
         "ad_empty_q", "ad_empty_c", "nm_rare_shared"]


def _prep(texts):
    """-> (token sets, digit-token sets, 4-gram sets) for already-normalised text."""
    toks, digs, grams = [], [], []
    for t in texts:
        s = set(t.split())
        toks.append(s)
        digs.append({x for x in s if x[:1].isdigit()})
        grams.append(ngrams(t, 4))
    return toks, digs, grams


def _jac(a, b):
    if not a and not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _cont(a, b):
    m = min(len(a), len(b))
    return len(a & b) / m if m else 0.0


def build(q_names, q_addrs, q_idx, c_names, c_addrs, idf_lut=None):
    """q_*: per-entity normalised text. q_idx: entity index per pair.
    c_*: per-PAIR normalised candidate text (already gathered)."""
    n = len(c_names)
    out = np.zeros((n, len(NAMES)), np.float32)
    qn_t, qn_d, qn_g = _prep(q_names)
    qa_t, qa_d, _ = _prep(q_addrs)

    for i in range(n):
        e = q_idx[i]
        cn, ca = c_names[i], c_addrs[i]
        ct = set(cn.split())
        at = set(ca.split())
        cd = {x for x in at if x[:1].isdigit()}
        qt, qg = qn_t[e], qn_g[e]
        qa, qd = qa_t[e], qn_d[e] | qa_d[e]

        out[i, 0] = _jac(qt, ct)
        out[i, 1] = _cont(qt, ct)
        out[i, 2] = _jac(qg, ngrams(cn, 4))
        out[i, 3] = float(q_names[e] == cn and bool(cn))
        lq, lc = len(q_names[e]), len(cn)
        out[i, 4] = min(lq, lc) / max(lq, lc) if max(lq, lc) else 0.0
        out[i, 5] = _jac(qa, at)
        out[i, 6] = _cont(qa, at)
        out[i, 7] = _jac(qd, cd)
        out[i, 8] = len(qd & cd)
        out[i, 9] = float(bool(qd) and bool(cd))
        out[i, 10] = float(not qa)
        out[i, 11] = float(not at)
        if idf_lut is not None:
            shared = qt & ct
            out[i, 12] = max((idf_lut.get(t, 0.0) for t in shared), default=0.0)
    return out


def _self_test():
    qn = [norm("Acme Ltd"), norm("Acme Ltd")]
    qa = [norm("12 Main St, Springfield"), norm("12 Main St, Springfield")]
    cn = [norm("Acme Ltd"), norm("Acme Ltd")]
    ca = [norm("12 Main St, Springfield"), norm("99 Main St, Springfield")]
    X = build(qn, qa, np.array([0, 1]), cn, ca)
    col = {k: i for i, k in enumerate(NAMES)}
    same, chain = X[0], X[1]
    assert same[col["nm_exact"]] == 1.0 and same[col["nm_jac"]] == 1.0
    assert chain[col["nm_exact"]] == 1.0 and chain[col["nm_jac"]] == 1.0, \
        "same chain must be indistinguishable by NAME"
    assert chain[col["dg_jac"]] < same[col["dg_jac"]], \
        "digit features must separate two branches of one chain"
    disjoint = build([norm("Acme Ltd")], [norm("12 Main St")], np.array([0]),
                     [norm("Zeta Foods")], [norm("77 Oak Rd")])[0]
    assert disjoint[col["nm_jac"]] == 0.0 and disjoint[col["dg_jac"]] == 0.0
    empty = build([norm("Acme")], [""], np.array([0]), [norm("Acme")], [""])[0]
    assert empty[col["ad_empty_q"]] == 1.0 and empty[col["ad_empty_c"]] == 1.0
    print("PASS -- chain separated by digits, not by name")


if __name__ == "__main__":
    _self_test()
