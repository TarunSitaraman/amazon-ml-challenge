"""candfeatures: hand-made records, the plain-Python oracle across chunk
boundaries, the cross-source twin rule, and the column layout build_corpus,
train_eval and predict share."""
import os
import pathlib
import subprocess
import sys

import numpy as np
import pyarrow as pa

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import candfeatures as cf  # noqa: E402

COL = {k: i for i, k in enumerate(cf.ALL_NAMES)}


def _run(rows):
    ids, names = zip(*rows)
    tab = pa.table({"entity_id": list(ids), "business_name": list(names)})
    return cf.record_matrix(tab), cf._reference(list(names), list(ids))


def test_hand_made():
    X, ref = _run([
        ("S2-1", "ACME  TRADERS"),        # caps, double space
        ("S3-1", "acme  traders"),        # lower; twin of S2-1 (case-insensitive)
        ("S2-2", "<< Sharma Pvt. Ltd."),  # junk prefix; trailing period is not junk
        ("S2-3", "Sharma Pvt. Ltd."),     # same source as S2-2: no twin
        ("S3-2", "Verma & Sons --"),      # junk suffix; '&' in the middle is not
        ("S3-3", "Verma & Sons"),
        ("S2-4", None),
        ("S3-4", ""),                     # empty names are never twins
        ("S2-5", "श्री गणेश"),              # Devanagari: no case, no junk
    ])
    assert (X == ref).all()
    c = lambda k: X[:, COL[k]].tolist()  # noqa: E731
    assert c("cand_name_tokens") == [2, 2, 4, 3, 4, 3, 0, 0, 2]
    assert c("cand_all_caps") == [1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert c("cand_all_lower") == [0, 1, 0, 0, 0, 0, 0, 0, 0]
    assert c("cand_double_space") == [1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert c("cand_junk_affix") == [0, 0, 1, 0, 1, 0, 0, 0, 0]
    assert c("cand_xsrc_twin") == [1, 1, 0, 0, 0, 0, 0, 0, 0]


def test_oracle_across_chunks(monkeypatch):
    monkeypatch.setattr(cf, "_CHUNK", 7_000)
    rng = np.random.default_rng(3)
    words = ["Acme", "GLOBAL", "traders", "Ltd.", "&", "Co.", "गणेश", "<<", "--",
             "##", "...", " ", "e.K."]
    names = [(" " if rng.random() < .8 else "  ").join(rng.choice(words, rng.integers(0, 6)))
             for _ in range(30_000)]
    names = [n.upper() if rng.random() < .2 else n for n in names]
    rows = [(f"S{rng.choice([2, 3])}-{i}", n) for i, n in enumerate(names)]
    X, ref = _run(rows)
    assert X.shape == (len(rows), len(cf.ALL_NAMES))
    assert (X == ref).all()


def test_build_is_a_gather():
    X, _ = _run([("S2-1", "A"), ("S3-1", "b c")])
    assert (cf.build(X, np.array([1, 1, 0])) == X[[1, 1, 0]]).all()


def test_off_switch_drops_columns():
    code = ("import pyarrow as pa, candfeatures as cf, features, strfeatures;"
            "t = pa.table({'entity_id': ['S2-1'], 'business_name': ['A']});"
            "assert cf.NAMES == [] and cf.record_matrix(t).shape == (1, 0)")
    subprocess.run([sys.executable, "-c", code], cwd=SRC, check=True,
                   env={**os.environ, "CAND_FEATS": "0"})


def test_columns_append_after_existing():
    import features
    import strfeatures
    import train_eval
    n = len(features.NAMES) + len(strfeatures.NAMES)
    assert train_eval.FEATURE_NAMES[:n] == features.NAMES + strfeatures.NAMES
    assert train_eval.FEATURE_NAMES[n:] == cf.NAMES
