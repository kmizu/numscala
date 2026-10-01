#!/usr/bin/env python3
"""Differential-test generator for num-scala.

Produces randomized cases, evaluates them with NumPy (the oracle) and writes one JSON
object per line to src/test/resources/difftest/<category>.jsonl.  The munit suite
src/test/scala/numscala/DiffSuite.scala replays every case against num-scala.

Run from the repository root:  python3 project/difftest/gen_difftest.py
Deterministic for a given NumPy version (seeded); NumPy 2.4.2 was used.

Encoding
--------
array  : {"d": dtype, "s": shape, "v": flat C-order values, "w": [view steps]}
         the values describe a C-contiguous *base* array; the view steps are then applied
         in order ("T": transpose(axes) or full reverse, "ix": basic index) so the input
         can be a non-contiguous / negatively strided view.
value  : bool -> true/false, ints -> JSON ints, floats -> repr strings ("nan", "-0.0"),
         complex -> [re, im] repr strings.
index  : list of items {"i": int} | {"sl": [start, stop, step]} (null = omitted)
         | {"na": 1} (newaxis) | {"el": 1} (ellipsis) | {"a": array} (int or bool array)
result : {"ok": array} | {"err": exception class name} | {"str": string}
"""
import json
import os
import sys

import numpy as np

assert np.__version__.startswith("2."), np.__version__

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "src", "test", "resources", "difftest")
# optional: gen_difftest.py [seed] [scale] -- other seeds / larger runs for local bug hunting
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20261001
SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
RNG = np.random.default_rng(SEED)

DTYPES = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint32", "float32", "float64", "complex128"]
NUMERIC = DTYPES[1:]
REAL = DTYPES[:-1]


def rint(lo, hi):
    return int(RNG.integers(lo, hi + 1))


def choice(xs):
    return xs[rint(0, len(xs) - 1)]


# --------------------------------------------------------------------------- values

def fenc(x):
    return repr(float(x))


def enc_val(dt, x):
    k = np.dtype(dt).kind
    if k == "b":
        return bool(x)
    if k in "iu":
        return int(x)
    if k == "f":
        return str(x) if dt == np.float32 else fenc(x)  # float32: shortest float32 repr
    return [fenc(x.real), fenc(x.imag)]


def enc_plain(a):
    """Encodes a contiguous array (no view steps)."""
    a = np.asarray(a)
    return {"d": a.dtype.name, "s": list(a.shape), "v": [enc_val(a.dtype, x) for x in a.ravel()]}


def rand_values(dt, n, small=False, nan=True, finite_range=None):
    d = np.dtype(dt)
    if d.kind == "b":
        return RNG.integers(0, 2, n).astype(bool)
    if d.kind in "iu":
        info = np.iinfo(d)
        mode = rint(0, 3)
        if small or mode < 2:
            lo, hi = (0, 9) if d.kind == "u" else (-9, 9)
        else:
            lo, hi = int(info.min), int(info.max)
        if finite_range:
            lo, hi = max(lo, finite_range[0]), min(hi, finite_range[1])
        v = RNG.integers(lo, hi, n, endpoint=True, dtype=np.int64 if d != np.uint32 else np.uint64)
        v = v.astype(d)
        if not finite_range and not small and n and rint(0, 3) == 0:
            # edge values: extremes, zero and -1 (division / overflow corner cases)
            edges = [info.min, info.max, 0, 1] + ([-1] if d.kind == "i" else [])
            for _ in range(rint(1, 2)):
                v[rint(0, n - 1)] = choice(edges)
        return v
    if d.kind == "f":
        mode = rint(0, 4)
        if small or mode <= 1:
            v = RNG.integers(-9, 10, n).astype(np.float64) / choice([1, 2, 4])
        elif mode == 2:
            v = RNG.standard_normal(n) * 10.0 ** rint(-6, 8)
        else:
            v = RNG.uniform(-100, 100, n)
        if finite_range:
            v = np.clip(np.trunc(v), finite_range[0], finite_range[1])
        elif nan and n > 0 and rint(0, 3) == 0:
            for _ in range(rint(1, 2)):
                v[rint(0, n - 1)] = choice([np.nan, np.inf, -np.inf, -0.0])
        return v.astype(d)
    re = rand_values("float64", n, small, nan=False, finite_range=finite_range)
    im = rand_values("float64", n, small, nan=False, finite_range=finite_range)
    if finite_range:
        im = np.zeros(n)
    return (re + 1j * im).astype(d)


def rand_shape(maxdim=3, maxlen=4, allow_zero=True):
    nd = rint(0, maxdim)
    return tuple(rint(0 if allow_zero and rint(0, 5) == 0 else 1, maxlen) for _ in range(nd))


def rand_view_steps(shape):
    """Random view steps applicable to a base array of `shape`; returns (steps, final shape)."""
    steps = []
    a = np.empty(shape)
    for _ in range(rint(0, 2)):
        r = rint(0, 2)
        if r == 0 and a.ndim >= 2:
            axes = [int(x) for x in RNG.permutation(a.ndim)]
            steps.append({"T": axes})
            a = a.transpose(axes)
        elif r == 1 and a.ndim >= 1:
            items = []
            idx = []
            for n in a.shape:
                st = choice([1, -1, 2, -2, 1])
                if n == 0:
                    items.append({"sl": [None, None, st]})
                    idx.append(slice(None, None, st))
                    continue
                if st > 0:
                    s0 = rint(0, max(0, n - 1)) if rint(0, 2) == 0 else None
                    items.append({"sl": [s0, None, st]})
                    idx.append(slice(s0, None, st))
                else:
                    items.append({"sl": [None, None, st]})
                    idx.append(slice(None, None, st))
            steps.append({"ix": items})
            a = a[tuple(idx)]
    return steps, a.shape


def apply_steps(a, steps):
    for s in steps:
        if "T" in s:
            a = a.transpose(s["T"])
        else:
            a = a[dec_index(s["ix"])]
    return a


def base_for(dt, view_shape_hint=None, **kw):
    """Random array (possibly a view). Returns (numpy array, encoding)."""
    shape = rand_shape(**{k: v for k, v in kw.items() if k in ("maxdim", "maxlen", "allow_zero")})
    vals = rand_values(dt, int(np.prod(shape)), **{k: v for k, v in kw.items() if k in ("small", "nan", "finite_range")})
    base = vals.reshape(shape)
    steps, _ = rand_view_steps(shape) if kw.get("views", True) else ([], shape)
    enc = enc_plain(base)
    enc["w"] = steps
    return apply_steps(base, steps), enc


def arr_of_shape(dt, shape, views=True, **kw):
    """Random array with exactly `shape`, built as a view of a (possibly transposed) base."""
    shape = tuple(shape)
    steps = []
    if views and len(shape) >= 2 and rint(0, 2) == 0:
        perm = [int(x) for x in RNG.permutation(len(shape))]
        inv = list(np.argsort(perm))
        base_shape = tuple(shape[i] for i in inv)
        # base.transpose(perm) has shape base_shape[perm] == shape
        base_shape = [0] * len(shape)
        for j, p in enumerate(perm):
            base_shape[p] = shape[j]
        base_shape = tuple(base_shape)
        steps.append({"T": perm})
    else:
        base_shape = shape
    if views and len(shape) >= 1 and rint(0, 2) == 0:
        ax = rint(0, len(shape) - 1)
        rev = {"ix": [{"sl": [None, None, -1 if i == ax else 1]} for i in range(len(shape))]}
        steps.append(rev)
    vals = rand_values(dt, int(np.prod(base_shape)), **kw)
    base = vals.reshape(base_shape)
    enc = enc_plain(base)
    enc["w"] = steps
    a = apply_steps(base, steps)
    assert a.shape == shape, (a.shape, shape, steps)
    return a, enc


def dec_index(items):
    out = []
    for it in items:
        if "i" in it:
            out.append(it["i"])
        elif "sl" in it:
            out.append(slice(*it["sl"]))
        elif "na" in it:
            out.append(None)
        elif "el" in it:
            out.append(Ellipsis)
        else:
            out.append(dec_array(it["a"]))
    return tuple(out)


def dec_array(e):
    def dv(x):
        if isinstance(x, list):
            return complex(float(x[0]), float(x[1]))
        if isinstance(x, str):
            return float(x)
        return x
    base = np.array([dv(x) for x in e["v"]], dtype=e["d"]).reshape(e["s"])
    return apply_steps(base, e.get("w", []))


def result(fn):
    with np.errstate(all="ignore"):
        try:
            r = fn()
        except Exception as ex:  # noqa: BLE001
            return {"err": type(ex).__name__}
    if isinstance(r, str):
        return {"str": r}
    if isinstance(r, (list, tuple)):
        return {"list": [enc_plain(np.asarray(x)) for x in r]}
    return {"ok": enc_plain(np.asarray(r))}


CASES = {}


def emit(cat, case):
    CASES.setdefault(cat, []).append(case)


# --------------------------------------------------------------------------- indexing

def rand_index(shape):
    """Random NumPy index tuple for an array of `shape` (basic + advanced + masks)."""
    items = []
    nd = len(shape)
    ax = 0
    used_ellipsis = False
    while ax < nd or (rint(0, 4) == 0 and len(items) < 4):
        r = rint(0, 9)
        if ax >= nd:
            items.append({"na": 1})
            continue
        n = shape[ax]
        if r <= 1 and n > 0:
            items.append({"i": rint(-n, n - 1)})
            ax += 1
        elif r <= 4:
            st = choice([1, 1, 2, -1, -2, 3])
            def b():
                return None if rint(0, 1) == 0 else rint(-n - 2, n + 2)
            items.append({"sl": [b(), b(), st]})
            ax += 1
        elif r == 5:
            items.append({"na": 1})
        elif r == 6 and not used_ellipsis:
            used_ellipsis = True
            items.append({"el": 1})
            # ellipsis consumes a random number of remaining axes: emulate by stopping early
            rest = rint(0, nd - ax)
            ax = nd - rest
        elif r == 7 and n > 0:
            k = rint(0, 4)
            ish = choice([(k,), (k,), (2, 2), ()]) if rint(0, 3) == 0 else (k,)
            idt = choice(["int64", "int32", "int8", "uint8", "int16"])
            if np.dtype(idt).kind == "u":
                vals = RNG.integers(0, n, int(np.prod(ish)))
            else:
                vals = RNG.integers(-n, n, int(np.prod(ish)))
            items.append({"a": {**enc_plain(vals.astype(idt).reshape(ish)), "w": []}})
            ax += 1
        elif r == 8:
            # boolean mask over 1 or 2 axes
            m = rint(1, min(2, nd - ax))
            msh = shape[ax:ax + m]
            mv = RNG.integers(0, 2, int(np.prod(msh))).astype(bool).reshape(msh)
            items.append({"a": {**enc_plain(mv), "w": []}})
            ax += m
        else:
            items.append({"sl": [None, None, 1]})
            ax += 1
    return items


def gen_indexing(n):
    for _ in range(n):
        dt = choice(DTYPES)
        a, enc = base_for(dt, maxdim=4, maxlen=4)
        idx = rand_index(a.shape)
        emit("getitem", {"a": enc, "idx": idx, "r": result(lambda: a[dec_index(idx)])})
    for _ in range(n // 2):
        dt = choice(DTYPES)
        shape = rand_shape(maxdim=3, maxlen=4)
        base = rand_values(dt, int(np.prod(shape))).reshape(shape)
        steps, vshape = rand_view_steps(shape)
        enc = enc_plain(base)
        enc["w"] = steps
        idx = rand_index(vshape)
        b2 = base.copy()
        v = apply_steps(b2, steps)
        try:
            with np.errstate(all="ignore"):
                sel_shape = v[dec_index(idx)].shape
        except Exception:  # noqa: BLE001
            sel_shape = ()
        # value: same dtype, shape = selection shape with some leading dims dropped / set to 1
        vs = list(sel_shape[rint(0, len(sel_shape)):]) if rint(0, 2) else list(sel_shape)
        vs = [1 if rint(0, 4) == 0 else x for x in vs]
        scalar = rint(0, 3) == 0
        if scalar:
            vs = []
        val = rand_values(dt, int(np.prod(vs))).reshape(vs)

        def run(b2=b2, v=v, idx=idx, val=val):
            v[dec_index(idx)] = val
            return b2
        emit("setitem", {"a": enc, "idx": idx, "val": {**enc_plain(val), "w": []}, "scalar": scalar,
                         "r": result(run)})


# --------------------------------------------------------------------------- shapes / views

def gen_views(n):
    for _ in range(n):
        dt = choice(DTYPES)
        a, enc = base_for(dt, maxdim=4, maxlen=4)
        r = rint(0, 2)
        if r == 0:
            sz = a.size
            dims = []
            rem = sz
            for _ in range(rint(0, 3)):
                ds = [d for d in range(1, max(rem, 1) + 1) if rem % d == 0] or [1]
                d = choice(ds)
                dims.append(d)
                rem //= d if d else 1
            dims.append(rem)
            if rint(0, 2) == 0 and dims:
                dims[rint(0, len(dims) - 1)] = -1
            if sz == 0 and -1 in dims:
                dims = [0 if x == -1 else x for x in dims]
            emit("views", {"op": "reshape", "a": enc, "shape": dims,
                           "r": result(lambda: a.reshape(dims)),
                           "view": bool(np.shares_memory(a.reshape(dims), a)) if a.size and _ok(lambda: a.reshape(dims)) else None})
        elif r == 1:
            axes = [int(x) for x in RNG.permutation(a.ndim)] if rint(0, 1) else None
            if axes is not None and a.ndim and rint(0, 3) == 0:
                axes = [x - a.ndim if rint(0, 1) else x for x in axes]
            emit("views", {"op": "transpose", "a": enc, "axes": axes,
                           "r": result(lambda: a.transpose(axes) if axes is not None else a.T), "view": None})
        else:
            emit("views", {"op": "ravel", "a": enc, "r": result(lambda: a.ravel()),
                           "view": bool(np.shares_memory(a.ravel(), a)) if a.size else None})


def _ok(fn):
    try:
        fn()
        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- binary ops

BIN_OPS = ["add", "subtract", "multiply", "true_divide", "floor_divide", "remainder", "fmod", "power",
           "equal", "not_equal", "less", "less_equal", "greater", "greater_equal"]


def bshapes():
    s = rand_shape(maxdim=3, maxlen=3)
    t = list(s)
    for i in range(len(t)):
        if rint(0, 2) == 0:
            t[i] = 1
    if rint(0, 2) == 0:
        t = t[rint(0, len(t)):]
    if rint(0, 1):
        return s, tuple(t)
    return tuple(t), s


def gen_binary(n):
    for _ in range(n):
        op = choice(BIN_OPS)
        da, db = choice(DTYPES), choice(DTYPES)
        sa, sb = bshapes()
        small = op in ("power",)
        # complex power with non-finite operands is a C99 Annex G corner case (not covered)
        nan = not (op == "power" and "complex128" in (da, db))
        a, ea = arr_of_shape(da, sa, small=small, nan=nan)
        b, eb = arr_of_shape(db, sb, small=small, nan=nan)
        if op == "power" and np.dtype(np.result_type(a, b)).kind in "iub":
            b = np.abs(b) if b.dtype.kind != "b" else b
            eb = enc_plain(b)
            eb["w"] = []
        f = getattr(np, op)
        emit("binary", {"op": op, "a": ea, "b": eb, "r": result(lambda: f(a, b))})


# --------------------------------------------------------------------------- reductions

RED_OPS = ["sum", "prod", "mean", "std", "var", "min", "max", "argmin", "argmax", "cumsum", "cumprod",
           "nansum", "nanmean", "nanmin", "nanmax"]


def gen_reduce(n):
    for _ in range(n):
        op = choice(RED_OPS)
        dt = choice(DTYPES)
        if op.startswith("nan") and np.dtype(dt).kind == "c" and op in ("nanmin", "nanmax"):
            dt = "float64"
        a, enc = base_for(dt, maxdim=3, maxlen=4, small=(op in ("prod", "cumprod")))
        if a.ndim == 0:
            axis = None
        else:
            axis = choice([None, rint(-a.ndim, a.ndim - 1), rint(-a.ndim, a.ndim - 1)])
            if a.ndim >= 2 and rint(0, 3) == 0 and op not in ("argmin", "argmax", "cumsum", "cumprod"):
                axis = sorted(int(x) for x in RNG.choice(a.ndim, rint(1, a.ndim), replace=False))
        keep = bool(rint(0, 1)) if op not in ("cumsum", "cumprod") else False
        ddof = rint(0, 1) if op in ("std", "var") else 0
        f = getattr(np, op)
        kw = {}
        if op not in ("cumsum", "cumprod"):
            kw["keepdims"] = keep
        if ddof:
            kw["ddof"] = ddof
        emit("reduce", {"op": op, "a": enc, "axis": axis, "keepdims": keep, "ddof": ddof,
                        "r": result(lambda: f(a, axis=tuple(axis) if isinstance(axis, list) else axis, **kw))})


# --------------------------------------------------------------------------- sorting / sets

def gen_sort(n):
    for _ in range(n):
        op = choice(["sort", "argsort", "unique", "searchsorted"])
        dt = choice(DTYPES)
        cnan = np.dtype(dt).kind != "c"
        if op == "searchsorted":
            a = np.sort(rand_values(dt, rint(0, 8), small=True, nan=cnan))
            vdt = dt if rint(0, 1) or dt == "complex128" else choice(REAL)
            v, ev = base_for(vdt, maxdim=2, maxlen=4, small=True, nan=cnan)
            side = choice(["left", "right"])
            emit("sort", {"op": op, "a": {**enc_plain(a), "w": []}, "v": ev, "side": side,
                          "r": result(lambda: np.searchsorted(a, v, side=side))})
            continue
        a, enc = base_for(dt, maxdim=3, maxlen=5, small=True, nan=cnan)
        if op == "unique":
            emit("sort", {"op": op, "a": enc, "r": result(lambda: np.unique(a))})
            continue
        axis = None if a.ndim == 0 or rint(0, 3) == 0 else rint(-a.ndim, a.ndim - 1)
        if op == "sort":
            emit("sort", {"op": op, "a": enc, "axis": axis, "r": result(lambda: np.sort(a, axis=axis))})
        else:
            emit("sort", {"op": op, "a": enc, "axis": axis,
                          "r": result(lambda: np.argsort(a, axis=axis, kind="stable"))})


# --------------------------------------------------------------------------- joining / editing

def gen_shape_ops(n):
    for _ in range(n):
        op = choice(["concatenate", "stack", "array_split", "split", "pad", "roll", "flip"])
        dt = choice(DTYPES)
        if op in ("concatenate", "stack"):
            shape = list(rand_shape(maxdim=3, maxlen=3))
            if op == "concatenate" and not shape:
                shape = [rint(0, 3)]
            k = rint(1, 3)
            nd = len(shape)
            axis = rint(-nd, nd - 1) if op == "concatenate" else rint(-nd - 1, nd)
            arrs, encs = [], []
            for _ in range(k):
                s = list(shape)
                if op == "concatenate":
                    s[axis] = rint(0, 3)
                a, e = arr_of_shape(dt, s)
                arrs.append(a)
                encs.append(e)
            f = getattr(np, op)
            emit("shape", {"op": op, "arrs": encs, "axis": axis, "r": result(lambda: f(arrs, axis=axis))})
            continue
        a, enc = base_for(dt, maxdim=3, maxlen=5, views=True)
        if op in ("array_split", "split"):
            if a.ndim == 0:
                continue
            axis = rint(-a.ndim, a.ndim - 1)
            if rint(0, 1):
                ios = rint(1, 4)
            else:
                ios = sorted(rint(0, 6) for _ in range(rint(0, 3)))
            f = getattr(np, op)
            emit("shape", {"op": op, "a": enc, "axis": axis, "ios": ios,
                           "r": result(lambda: f(a, ios, axis=axis))})
        elif op == "pad":
            mode = choice(["constant", "edge", "reflect", "symmetric", "wrap", "maximum", "minimum", "mean",
                           "median", "linear_ramp"])
            if mode in ("mean", "median", "linear_ramp") and np.dtype(dt).kind in "bc":
                mode = "edge"
            if mode in ("mean", "median", "linear_ramp"):
                # computed through float64 by both libraries; keep integers exactly representable
                a, enc = base_for(dt, maxdim=3, maxlen=5, small=True)
            pw = [[rint(0, 3), rint(0, 3)] for _ in range(a.ndim)]
            if a.ndim == 0:
                continue
            cv = 1 if dt == "bool" else rint(0, 3) if np.dtype(dt).kind == "u" else rint(-3, 3)
            kw = {"constant_values": cv} if mode == "constant" else {}
            emit("shape", {"op": op, "a": enc, "pw": pw, "mode": mode, "cv": cv,
                           "r": result(lambda: np.pad(a, pw, mode=mode, **kw))})
        elif op == "roll":
            if a.ndim == 0 or rint(0, 2) == 0:
                axis = None
                shift = rint(-7, 7)
            elif rint(0, 1):
                axis = rint(-a.ndim, a.ndim - 1)
                shift = rint(-7, 7)
            else:
                axis = [rint(0, a.ndim - 1) for _ in range(rint(1, 2))]
                shift = [rint(-7, 7) for _ in axis]
            emit("shape", {"op": op, "a": enc, "axis": axis, "shift": shift,
                           "r": result(lambda: np.roll(a, shift, axis=axis))})
        else:
            axis = None if a.ndim == 0 or rint(0, 2) == 0 else rint(-a.ndim, a.ndim - 1)
            emit("shape", {"op": op, "a": enc, "axis": axis, "r": result(lambda: np.flip(a, axis=axis))})


# --------------------------------------------------------------------------- products

def gen_products(n):
    for _ in range(n):
        op = choice(["matmul", "dot"])
        da, db = choice(DTYPES), choice(DTYPES)
        m, k, p = rint(0, 3), rint(0, 3), rint(1, 3)
        form = rint(0, 4)
        if form == 0:
            sa, sb = (k,), (k,)
        elif form == 1:
            sa, sb = (m, k), (k,)
        elif form == 2:
            sa, sb = (k,), (k, p)
        elif form == 3:
            sa, sb = (m, k), (k, p)
        else:
            bt = rint(1, 2)
            sa, sb = (bt, m, k), ((k, p) if op == "matmul" and rint(0, 1) else (bt, k, p) if op == "matmul" else (k, p))
        a, ea = arr_of_shape(da, sa, small=True, nan=False)
        b, eb = arr_of_shape(db, sb, small=True, nan=False)
        f = getattr(np, op)
        emit("products", {"op": op, "a": ea, "b": eb, "r": result(lambda: f(a, b))})


# --------------------------------------------------------------------------- printing

def print_values(dt, n):
    d = np.dtype(dt)
    if d.kind == "f" or d.kind == "c":
        mode = rint(0, 6)
        if mode == 0:
            v = RNG.integers(-20, 20, n).astype(np.float64)
        elif mode == 1:
            v = np.round(RNG.uniform(-10, 10, n), rint(0, 4))
        elif mode == 2:
            v = RNG.standard_normal(n) * 10.0 ** rint(-12, 18)
        elif mode == 3:
            v = RNG.uniform(0, 1, n) * 10.0 ** RNG.integers(-6, 9, n)
        elif mode == 4:
            v = RNG.standard_normal(n) * 1e-5
        elif mode == 5:
            v = RNG.uniform(-1e5, 1e5, n)
        else:
            v = RNG.standard_normal(n)
        if n and rint(0, 3) == 0:
            for _ in range(rint(1, 2)):
                v[rint(0, n - 1)] = choice([np.nan, np.inf, -np.inf, -0.0])
        if d.kind == "c":
            w = print_values("float64", n).astype(np.float64)
            return (v + 1j * w).astype(d)
        return v.astype(d)
    return rand_values(dt, n)


def gen_print(n):
    for _ in range(n):
        dt = choice(DTYPES)
        if rint(0, 100) == 0:
            shape = choice([(1001,), (11, 92), (3, 334), (2, 3, 167)])
        else:
            shape = rand_shape(maxdim=choice([2, 3, 4]), maxlen=choice([4, 5, 6]))
            while int(np.prod(shape)) > 150:
                shape = shape[:-1]
        base = print_values(dt, int(np.prod(shape))).reshape(shape)
        steps, _ = rand_view_steps(shape)
        enc = enc_plain(base)
        enc["w"] = steps
        a = apply_steps(base, steps)
        emit("print", {"a": enc, "str": str(a), "repr": repr(a)})


# --------------------------------------------------------------------------- astype

def gen_astype(n):
    for _ in range(n):
        src, dst = choice(DTYPES), choice(DTYPES)
        sk, dk = np.dtype(src).kind, np.dtype(dst).kind
        kw = {}
        if sk in "fc" and dk in "iu":
            info = np.iinfo(dst)
            kw["finite_range"] = (max(int(info.min), -2 ** 31), min(int(info.max), 2 ** 31 - 1))
        a, enc = base_for(src, maxdim=3, maxlen=4, **kw)
        emit("astype", {"a": enc, "to": dst, "r": result(lambda: a.astype(dst))})


def main():
    def n(k):
        return int(k * SCALE)
    gen_indexing(n(700))
    gen_views(n(400))
    gen_binary(n(1500))
    gen_reduce(n(1400))
    gen_sort(n(600))
    gen_shape_ops(n(700))
    gen_products(n(400))
    gen_print(n(500))
    gen_astype(n(500))
    os.makedirs(OUT, exist_ok=True)
    total = 0
    for cat, cases in CASES.items():
        with open(os.path.join(OUT, cat + ".jsonl"), "w") as fh:
            for c in cases:
                fh.write(json.dumps(c, separators=(",", ":")) + "\n")
        total += len(cases)
        print(f"{cat}: {len(cases)}", file=sys.stderr)
    print(f"total: {total}", file=sys.stderr)


if __name__ == "__main__":
    main()
