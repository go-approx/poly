# Performance against NumPy

**Not yet measured.** The comparison (Go against NumPy on one core, interleaved
rounds, median of at least 7 rounds, `uptime` recorded) must run on an idle
host; at the time of writing the shared build host had a one-minute load average
between 12 and 200, which makes any ratio meaningless, so no figure is
published here rather than a misleading one.

The driver is in place: `perf_oracle_test.go` (build tag `oracle`) and the
benchmarks in `bench_test.go`:

```
GOWORK=off go test -tags oracle -run TestPerf -timeout 6h -v
```

It writes this file. Target: no slower than NumPy at 10^3 to 10^6 points for
`polyval`, `polyfit` (degree 3 and 10), `roots` (degree 10 and 100) and
`chebval`/`chebfit`/`chebvander`.
