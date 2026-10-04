#!/usr/bin/env python3
"""Summarise a JMH JSON result (GemmBench/KernelBench) as Markdown tables for docs/perf.

usage: python3 project/perf/jmh_report.py docs/perf/jmh-NS-CPU-001.json
"""
import json, math, sys
from collections import defaultdict

rows = json.load(open(sys.argv[1]))

def key(r):
    p = r.get("params", {})
    if "workers" in p:
        return "ParallelGemmBench", p["shape"], int(p["workers"])
    return r["benchmark"].rsplit(".", 2)[-2], r["benchmark"].rsplit(".", 1)[-1], p.get("shape") or p.get("backend")

res = {}
for r in rows:
    pm = r["primaryMetric"]
    sec = r.get("secondaryMetrics", {})
    alloc = sec.get("gc.alloc.rate.norm", {}).get("score")
    gc_time = sec.get("gc.time", {}).get("score")
    raw = [x for fork in pm["rawData"] for x in fork]
    raw.sort()
    med = raw[len(raw) // 2] if len(raw) % 2 else (raw[len(raw) // 2 - 1] + raw[len(raw) // 2]) / 2
    p95 = raw[min(len(raw) - 1, math.ceil(0.95 * len(raw)) - 1)]
    res[key(r)] = dict(score=pm["score"], err=pm["scoreError"], med=med, p95=p95, alloc=alloc, gc=gc_time,
                       n=len(raw), forks=r["forks"], wi=r["warmupIterations"], mi=r["measurementIterations"],
                       wt=r["warmupTime"], mt=r["measurementTime"], jvm=r["jvm"], jdk=r["jdkVersion"],
                       vm=r["vmVersion"], threads=r["threads"])

def flops(shape):
    dims = shape.split("-")[0].split("x")
    m, n, k = map(int, dims)
    return 2.0 * m * n * k

def fmt(x, d=1):
    return "-" if x is None else f"{x:,.{d}f}"

any_r = next(iter(res.values()))
print(f"Settings: forks={any_r['forks']}, warmup={any_r['wi']} x {any_r['wt']}, measurement={any_r['mi']} x {any_r['mt']}, "
      f"threads={any_r['threads']}, JDK {any_r['jdk']} ({any_r['vm']}).\n")

shapes = [s for (c, b, s) in res if c == "GemmBench" and b == "scalarGemmInto"]
print("### GEMM (median us/op, p95 in parentheses; GFLOP/s from the median, FMA = 2 FLOPs)\n")
print("| shape `MxNxK-op` | legacy np.matmul | np.matmul (new, scalar) | scalar gemmInto | vector25 gemmInto | vector GFLOP/s | B/op legacy | B/op np.matmul | B/op scalar | B/op vector |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for s in shapes:
    g = {b: res.get(("GemmBench", b, s)) for b in ["legacyNpMatmul", "npMatmul", "scalarGemmInto", "vectorGemmInto"]}
    cell = lambda r: "-" if r is None else f"{fmt(r['med'])} ({fmt(r['p95'])})"
    gf = flops(s) / (g["vectorGemmInto"]["med"] * 1e3) if g["vectorGemmInto"] else None
    print(f"| {s} | {cell(g['legacyNpMatmul'])} | {cell(g['npMatmul'])} | {cell(g['scalarGemmInto'])} | {cell(g['vectorGemmInto'])} | "
          f"{fmt(gf)} | " + " | ".join(fmt(g[b]['alloc'], 0) if g[b] else "-" for b in g) + " |")

def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))

print("\n### Speedups (ratio of medians)\n")
sv = {s: res[("GemmBench", "scalarGemmInto", s)]["med"] / res[("GemmBench", "vectorGemmInto", s)]["med"] for s in shapes}
nl = {s: res[("GemmBench", "legacyNpMatmul", s)]["med"] / res[("GemmBench", "npMatmul", s)]["med"] for s in shapes}
print("| shape | vector25 vs scalar gemmInto | new np.matmul vs legacy |")
print("|---|---:|---:|")
for s in shapes:
    print(f"| {s} | {sv[s]:.2f}x | {nl[s]:.2f}x |")
print(f"| **geometric mean** | **{geomean(list(sv.values())):.2f}x** | **{geomean(list(nl.values())):.2f}x** |")

kb = sorted({b for (c, b, s) in res if c == "KernelBench"})
print("\n### Row ops, scan and reductions (median us/op; B/op)\n")
print("| kernel | scalar | vector25 | vector vs scalar | B/op scalar | B/op vector |")
print("|---|---:|---:|---:|---:|---:|")
kv = {}
for b in kb:
    s_, v_ = res.get(("KernelBench", b, "scalar")), res.get(("KernelBench", b, "vector25"))
    kv[b] = s_["med"] / v_["med"]
    print(f"| {b} | {fmt(s_['med'], 2)} | {fmt(v_['med'], 2)} | {kv[b]:.2f}x | {fmt(s_['alloc'], 0)} | {fmt(v_['alloc'], 0)} |")
allr = list(sv.values()) + list(kv.values())
print(f"\nGate input (NS-CPU-001 §9.3): geometric mean of vector25/scalar over all {len(allr)} workloads = "
      f"**{geomean(allr):.2f}x**; worst workload = **{min(allr):.2f}x** "
      f"({min(list(sv.items()) + list(kv.items()), key=lambda t: t[1])[0]}).")

par = sorted({(s, w) for (c, s, w) in res if c == "ParallelGemmBench"})
if par:
    shapes_p = sorted({s for s, _ in par})
    workers = sorted({w for _, w in par})
    print("\n### Parallel GEMM (ParallelF32.gemmInto, vector25; median us/op, GFLOP/s, speedup vs 1 worker)\n")
    print("| shape | " + " | ".join(f"{w} worker{'s' if w > 1 else ''}" for w in workers) + " |")
    print("|---|" + "---:|" * len(workers))
    for sh in shapes_p:
        base = res[("ParallelGemmBench", sh, workers[0])]["med"]
        cells = []
        for w in workers:
            r = res.get(("ParallelGemmBench", sh, w))
            cells.append("-" if r is None else f"{fmt(r['med'])} us, {flops(sh) / (r['med'] * 1e3):.0f} GF/s, {base / r['med']:.2f}x")
        print(f"| {sh} | " + " | ".join(cells) + " |")
