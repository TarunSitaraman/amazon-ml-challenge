"""Stage-level wall-clock timing for predict.py and train_eval.py (--profile).

Usage:
    from profiling import PROF
    with PROF.stage("strfeatures.build", n=len(q), unit="pairs"):
        ...
    PROF.begin("India")   # per-country scope
    PROF.end()            # prints the country's table
    PROF.report()         # prints the overall table, since PROF.enable()

Off by default. Disabled, stage() returns one shared no-op context manager, so
the cost is a method call and an attribute check per stage -- stages are whole
pipeline steps, never per-pair loops, so that is a few hundred calls per run.

Each stage records EXCLUSIVE time: a stage nested inside another is subtracted
from its parent, so every second is counted once and the rows of a table plus
its "(untimed)" row add up to the scope's wall clock, i.e. 100%. Throughput
uses INCLUSIVE time (n items over the whole block), and so does "+MB".

Peak RSS is the process high-water mark (getrusage on POSIX, ctypes on
Windows, blank elsewhere; no psutil). "peak MB" is the mark when the stage
ended, "+MB" how much the stage raised it. A stage that allocates less than an
earlier stage already did shows +0 even though it did allocate: this answers
"which stage set the peak", not "how much does each stage use".
"""
import os
import sys
import time

_perf = time.perf_counter


def _peak_rss_reader():
    """-> a function returning peak RSS in bytes (or None if unavailable)."""
    try:
        import resource
        # ru_maxrss is KiB on Linux, bytes on macOS
        scale = 1 if sys.platform == "darwin" else 1024

        def read():
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
        read()
        return read
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD),
                            ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            get_info = ctypes.WinDLL("psapi").GetProcessMemoryInfo
            get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            get_info.restype = wintypes.BOOL
            cur = ctypes.WinDLL("kernel32").GetCurrentProcess
            cur.restype = wintypes.HANDLE
            handle, pmc = cur(), PMC()
            pmc.cb = ctypes.sizeof(PMC)

            def read():
                if not get_info(handle, ctypes.byref(pmc), pmc.cb):
                    return None
                return int(pmc.PeakWorkingSetSize)
            if read() is not None:
                return read
        except Exception:
            pass
    return lambda: None


peak_rss = _peak_rss_reader()


class _NullStage:
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL = _NullStage()


class _Stage:
    __slots__ = ("prof", "name", "n", "unit", "t0", "child", "rss0")

    def __init__(self, prof, name, n, unit):
        self.prof, self.name, self.n, self.unit = prof, name, n, unit

    def __enter__(self):
        self.child = 0.0
        self.rss0 = peak_rss()
        self.prof._stack.append(self)
        self.t0 = _perf()
        return self

    def __exit__(self, *exc):
        el = _perf() - self.t0
        stack = self.prof._stack
        # enable()/disable() inside an open stage drops it: record nothing
        if not stack or stack[-1] is not self:
            return False
        stack.pop()
        if stack:
            stack[-1].child += el
        self.prof._add(self.name, el - self.child, el, self.n, self.unit,
                       self.rss0, peak_rss())
        return False


class _Stat:
    __slots__ = ("calls", "sec", "incl", "n", "unit", "peak", "rise")

    def __init__(self, unit):
        self.calls, self.sec, self.incl, self.n, self.unit = 0, 0.0, 0.0, 0, unit
        self.peak, self.rise = None, 0


def _record(table, name, sec, incl, n, unit, rss0, rss1):
    st = table.get(name)
    if st is None:
        st = table[name] = _Stat(unit)
    st.calls += 1
    st.sec += sec
    st.incl += incl
    if n is not None:
        st.n += int(n)
    if rss0 is not None and rss1 is not None:
        st.peak = rss1 if st.peak is None else max(st.peak, rss1)
        st.rise += rss1 - rss0


class Profiler:
    def __init__(self):
        self.enabled = False
        self._stack = []
        self._total, self._t0 = {}, None
        self._scope, self._scope_t0, self._scope_title = None, None, None

    def enable(self):
        self.enabled = True
        self._total, self._t0 = {}, _perf()
        self._stack.clear()

    def disable(self):
        self.enabled = False
        self._stack.clear()
        self._scope = None

    def stage(self, name, n=None, unit=None):
        """Time a block as `name`. n items of `unit` give a throughput column."""
        if not self.enabled:
            return _NULL
        return _Stage(self, name, n, unit)

    def _add(self, *rec):
        _record(self._total, *rec)
        if self._scope is not None:
            _record(self._scope, *rec)

    def begin(self, title):
        if self.enabled:
            self._scope, self._scope_t0, self._scope_title = {}, _perf(), title

    def end(self, out=None):
        """Print and close the current scope. -> its rows (see rows())."""
        if not self.enabled or self._scope is None:
            return None
        wall = _perf() - self._scope_t0
        rows = self.rows(self._scope, wall)
        print(format_table(f"profile: {self._scope_title}", rows, wall),
              file=out or sys.stdout, flush=True)
        self._scope = None
        return rows

    def report(self, title="overall", out=None):
        """Print every stage since enable(), summed over all scopes."""
        if not self.enabled:
            return None
        wall = _perf() - self._t0
        rows = self.rows(self._total, wall)
        print(format_table(f"profile: {title}", rows, wall),
              file=out or sys.stdout, flush=True)
        return rows

    @staticmethod
    def rows(table, wall):
        """-> [(name, calls, sec, pct, per_sec, unit, peak, rise)], the untimed
        remainder last, so pct sums to 100."""
        out = []
        for name, st in table.items():
            rate = st.n / st.incl if st.n and st.incl > 0 else None
            out.append((name, st.calls, st.sec, 100.0 * st.sec / wall if wall else 0.0,
                        rate, st.unit, st.peak, st.rise if st.peak is not None else None))
        rest = max(wall - sum(st.sec for st in table.values()), 0.0)
        out.append(("(untimed)", 0, rest, 100.0 * rest / wall if wall else 0.0,
                    None, None, None, None))
        return out


def _mb(b):
    return "" if b is None else f"{b / 2**20:,.0f}"


def format_table(title, rows, wall):
    w = max([30] + [len(r[0]) for r in rows])
    lines = [title,
             f"  {'stage':{w}s} {'calls':>6s} {'sec':>9s} {'%':>6s} "
             f"{'throughput':>26s} {'peak MB':>9s} {'+MB':>7s}"]
    for name, calls, sec, pct, rate, unit, peak, rise in rows:
        thr = f"{rate:,.0f} {unit}/s" if rate is not None else ""
        lines.append(f"  {name:{w}s} {calls or '':>6} {sec:9.2f} {pct:6.1f} "
                     f"{thr:>26s} {_mb(peak):>9s} {_mb(rise):>7s}")
    lines.append(f"  {'total (wall)':{w}s} {'':>6s} {wall:9.2f} {100.0:6.1f}")
    return "\n".join(lines)


def pop_flag(argv, flag="--profile"):
    """Remove every `flag` from argv in place. -> whether it was there."""
    hit = flag in argv
    argv[:] = [a for a in argv if a != flag]
    return hit


PROF = Profiler()
stage = PROF.stage
