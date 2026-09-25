"""Stage timer (profiling.py): overhead, accounting, and the --profile wiring.

Synthetic only: the pipeline tests reuse test_silent_bugs' tiny parquet tree
and stub model, so they need no data/ and no model.pkl.
"""
import pathlib
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import profiling  # noqa: E402
from profiling import PROF, Profiler, pop_flag  # noqa: E402


@pytest.fixture(autouse=True)
def _prof_off():
    PROF.disable()
    yield
    PROF.disable()


def _pct_sums(text):
    """-> {table title: sum of its % column} parsed from printed tables."""
    sums, title, end = {}, None, None
    for line in text.splitlines():
        if line.startswith("profile: "):
            title = line[len("profile: "):]
            sums[title] = 0.0
        elif title and line.startswith("  total (wall)"):
            title = None
        elif title and line.startswith("  stage "):
            end = line.index(" %") + 2             # the % column is right-aligned
        elif title:
            sums[title] += float(line[end - 6:end])
    return sums


def _stage_names(text, title):
    lines = text.splitlines()
    i = lines.index(f"profile: {title}") + 2
    out = []
    w = lines[i - 1].index("calls") - 4            # "  " + name + " " + " calls"
    while not lines[i].startswith("  total (wall)"):
        out.append(lines[i][2:2 + w].strip())
        i += 1
    return out


def test_disabled_stage_is_shared_noop():
    p = Profiler()
    assert p.stage("a") is p.stage("b", n=5, unit="pairs") is profiling._NULL
    with p.stage("a"):
        pass
    p.begin("x")
    assert p.end() is None and p.report() is None and not p._total


def test_nested_stages_are_exclusive_and_pct_sums_to_100(capsys):
    p = Profiler()
    p.enable()
    p.begin("scope")
    with p.stage("outer", n=500, unit="pairs"):
        time.sleep(0.02)
        with p.stage("inner", n=1000, unit="pairs"):
            time.sleep(0.2)
    time.sleep(0.01)                      # untimed
    rows = {r[0]: r for r in p.end()}
    outer, inner = p._total["outer"], p._total["inner"]
    assert inner.sec >= 0.2 and outer.sec >= 0.02
    # outer's own time only: the inner stage is not counted twice (bounds are
    # loose on purpose, sleep can overshoot on a busy host)
    assert outer.sec < 0.15 and outer.incl == pytest.approx(outer.sec + inner.incl)
    assert rows["(untimed)"][2] >= 0.01
    assert sum(r[3] for r in rows.values()) == pytest.approx(100.0, abs=1e-6)
    # throughput over the whole block, children included
    assert rows["inner"][4] == pytest.approx(1000 / inner.incl)
    assert rows["outer"][4] == pytest.approx(500 / outer.incl)
    # a printed table sums to ~100 too, after rounding
    assert _pct_sums(capsys.readouterr().out)["scope"] == pytest.approx(100, abs=0.5)
    # the overall table spans the scope and anything outside it
    assert sum(r[3] for r in p.report()) == pytest.approx(100.0, abs=1e-6)


def test_peak_rss_reported_where_available():
    p = Profiler()
    p.enable()
    with p.stage("alloc"):
        x = np.ones(64 * 2**20 // 8)      # 64 MB, touched
        del x
    (name, calls, sec, pct, rate, unit, peak, rise), _ = p.rows(p._total, 1.0)
    if profiling.peak_rss() is None:
        pytest.skip("no peak-RSS source on this platform")
    assert peak >= 64 * 2**20 and rise >= 0


def test_overhead_under_one_percent():
    """Synthetic stage-sized calls: the timer adds < 1%, on or off.

    Each call does ~10ms of numpy work, the low end of a real stage (one
    stage per pipeline step per 20k-entity batch; most run for seconds).
    Comparing whole loops is noisy at +-1.5% here even for the disabled path,
    which does nothing measurable, so each wrapped call is paired with a bare
    one, in alternating order, and the overhead is the median difference.

    Measured on the dev container: ~3us per enabled call, yet ~1.3% against
    1ms stages -- the getrusage syscall costs the following work more than its
    own time. At 10ms it is well under 1%, and real stages are far longer.
    """
    rng = np.random.default_rng(0)
    a = rng.random(1_200_000)
    pc = time.perf_counter
    off, on = Profiler(), Profiler()
    on.enable()
    on.begin("x")

    def paired(p, n_pairs=120):
        bare, diff = [], []
        for i in range(n_pairs + 10):
            t0 = pc()
            if i % 2:
                np.sort(a)
                t1 = pc()
                with p.stage(f"s{i % 12}", n=len(a), unit="pairs"):
                    np.sort(a)
                t2 = pc()
                db, dw = t1 - t0, t2 - t1
            else:
                with p.stage(f"s{i % 12}", n=len(a), unit="pairs"):
                    np.sort(a)
                t1 = pc()
                np.sort(a)
                t2 = pc()
                dw, db = t1 - t0, t2 - t1
            if i >= 10:                   # warm-up pairs
                bare.append(db)
                diff.append(dw - db)
        return np.median(diff) / np.median(bare), np.median(bare)

    over_off, stage_s = paired(off)
    over_on, _ = paired(on)
    print(f"stage {stage_s*1e3:.2f}ms  overhead off {over_off:+.3%}  on {over_on:+.3%}")
    assert over_off < 0.01
    assert over_on < 0.01

    # And the fixed per-call cost on its own, free of workload noise: enabled
    # enter + exit must be under 1% of even a 1ms stage.
    k = 20_000
    t = pc()
    for _ in range(k):
        with on.stage("empty", n=1, unit="pairs"):
            pass
    per_call = (pc() - t) / k
    print(f"enabled cost per stage call {per_call*1e6:.1f}us")
    assert per_call < 0.01 * 1e-3, per_call


def test_open_stage_survives_disable_and_long_names(capsys):
    p = Profiler()
    p.enable()
    with p.stage("open"):
        p.disable()                       # must not raise on exit
    p.enable()
    long = "a stage name well over thirty characters"
    with p.stage(long):
        pass
    with p.stage("short"):
        pass
    p.report()
    out = capsys.readouterr().out
    assert _stage_names(out, "overall") == [long, "short", "(untimed)"]
    assert _pct_sums(out)["overall"] == pytest.approx(100.0, abs=0.5)


def test_rss_none_on_entry_is_skipped(monkeypatch):
    vals = iter([None, 123])
    monkeypatch.setattr(profiling, "peak_rss", lambda: next(vals))
    p = Profiler()
    p.enable()
    with p.stage("x"):
        pass
    assert p._total["x"].peak is None


def test_pop_flag():
    argv = ["train_eval.py", "India", "--profile", "15000", "0"]
    assert pop_flag(argv) and argv == ["train_eval.py", "India", "15000", "0"]
    assert not pop_flag(argv)


# ---- the --profile wiring ---------------------------------------------------

def _run_predict(tmp_path, monkeypatch, argv, out_dir):
    import predict
    from test_silent_bugs import _Iso, _Model, _write
    if not (tmp_path / "parquet").exists():
        _write(tmp_path / "parquet", "test")
        (tmp_path / "model.pkl").write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(predict, "ROOT", str(tmp_path / "parquet"))
    monkeypatch.setattr(predict, "OUT", tmp_path / out_dir)
    monkeypatch.setattr(predict, "BATCH", 2)
    monkeypatch.setattr(predict.pickle, "load",
                        lambda f: {"model": _Model(), "iso": _Iso()})
    monkeypatch.setattr(sys, "argv", ["predict.py"] + argv)
    predict.main()
    return {n: (tmp_path / out_dir / n).read_bytes()
            for n in ("matching_results.tsv", "candidate_pairs.tsv")}


def test_predict_profile_tables_and_unchanged_output(tmp_path, monkeypatch, capsys):
    plain = _run_predict(tmp_path, monkeypatch, ["--disjoint", "sinkhorn"], "a")
    assert "profile:" not in capsys.readouterr().out
    prof = _run_predict(tmp_path, monkeypatch,
                        ["--disjoint", "sinkhorn", "--profile"], "b")
    out = capsys.readouterr().out
    assert prof == plain                  # timing never changes the submission

    sums = _pct_sums(out)
    assert set(sums) == {"India", "overall"}
    for title, total in sums.items():
        assert total == pytest.approx(100.0, abs=0.5), (title, total)

    country = _stage_names(out, "India")
    for s in ["load_shard", "normalise corpus", "build_index c2_name",
              "build_index c6_addr", "build_index c1_canon", "build_index c5_postal",
              "build_index c3_pair", "normalise S1", "query c2_name",
              "query c6_addr", "query c1_canon", "query c5_postal", "query c3_pair",
              "union channels", "cap_candidates", "strfeatures.build",
              "features.build", "model.predict", "isotonic", "choose_k",
              "disjoint sinkhorn", "disjoint resolve", "(untimed)"]:
        assert s in country, s
    overall = _stage_names(out, "overall")
    assert "TSV write" in country             # the country's partial
    assert "TSV concat partials" in overall and "load model" in overall
    assert "TSV concat partials" not in country
    assert "queries/s" in out and "pairs/s" in out


def test_train_eval_prepare_country_profiles(tmp_path, monkeypatch, capsys):
    import train_eval
    from test_silent_bugs import _write
    _write(tmp_path / "parquet", "train")
    monkeypatch.setattr(train_eval, "ROOT", str(tmp_path / "parquet"))
    ref = train_eval.prepare_country("India", 2, 2, {}, np.random.default_rng(0))
    PROF.enable()
    got = train_eval.prepare_country("India", 2, 2, {}, np.random.default_rng(0))
    out = capsys.readouterr().out
    for a, b in zip(ref, got):
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[2], b[2])
    assert _pct_sums(out)["India"] == pytest.approx(100.0, abs=0.5)
    names = _stage_names(out, "India")
    assert {"strfeatures.build", "features.build", "labels",
            "query c2_name", "build_index c2_name"} <= set(names)
