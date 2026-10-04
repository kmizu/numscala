# bitcheck

This tool checks that two numscala versions produce **bit-identical** results for the kernels a model's
reproducibility depends on: `gemmInto` in all four layouts on both backends, `gatherRowsInto` and
`affineScanInto`. It runs each case twice, cold and after JIT warm-up. Requires JDK 25.

```bash
# in the repository root: publish the working tree locally
sbt publishLocal vector25/publishLocal        # note the printed version, e.g. 0.4.0+3-abcdef12-SNAPSHOT
cd tools/bitcheck
sbt -batch run                                # released version (default 0.4.0, or -Dnsv=x.y.z)
sbt -batch -Dnsv=0.4.0+3-abcdef12-SNAPSHOT run
```

The two printed `hash=` lines must be equal whenever a release promises unchanged bits.
