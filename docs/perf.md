# Performance against NumPy 2.5.4

The comparison runs under [refbench](https://github.com/go-scikit/refbench)
v0.1.2 (`rbtest.Perf`): one core (GOMAXPROCS=1; NumPy's BLAS threads pinned to
1), interleaved rounds (Go, NumPy, Go, NumPy, ...), the 1-minute load read before
every round and the measurement refused at a load of 8 or more. Target: no
slower than NumPy (ratio <= 1); a row above 1 is an open defect.

- The cases are in [`bench/perf_test.go`](../bench/perf_test.go): one or more
  per public function, against the `numpy` / `numpy.polynomial` function it ports
  (numpy 2.5.4, pinned with its hashes in `bench/refs.txt`). Both sides get the
  same bytes (`go test -run TestDataParity -perf.parity .`) and every NumPy
  expression is run once by `-perf.smoke`.
- Functions with no NumPy counterpart are named in
  [perf-exempt.md](perf-exempt.md) with the reason.
- The table, with the allocation count of every case, is
  [perf-table.md](perf-table.md), generated from [perf.json](perf.json).

```
cd bench
go test -run TestPerf .                          CI mode: allocations, table, inventory
go test -run TestPerf -cpu 1 -perf.run .         measure (refuses at a 1-minute load of 8 or more)
go test -run TestPerf -perf.record .             allocation counts only (a deferred report)
```

CI runs the first line: the allocation count of every case must equal
`perf.json` (an increase is a regression, a decrease must be recorded), the table
must be what `perf.json` generates, and every public function must have a case or
an exemption. Allocation counts are exact under any load; times are not, so the
timings of the table are recorded by hand on an idle machine (status `deferred`
until then).

## Batch evaluation

`PolyvalAll` and `ValAll` for `Chebyshev` run go-simd/floats v0.12.0 `Polyval`
and `Chebval`, the kernels that reproduce `numpy.polynomial.polynomial.polyval`
and `numpy.polynomial.chebyshev.chebval` operation by operation (no fused
multiply-add; the SIMD lanes run over different points, so each point sees
NumPy's order). `PolyvalAll` takes its coefficients highest degree first, the
kernel lowest first: the coefficients are reversed into a pooled buffer, so a
call allocates nothing beyond `dst` (`PolyvalAll` and `ValAll` rows: 0 allocs).
The results are bit-identical to `Polyval` and `chebval` (and so to NumPy,
whose fixtures `accuracy_test.go` compares bit for bit), Inf and NaN inputs
included; `kernels_test.go` checks it for 1 to 70 coefficients.

## Earlier measurement

Taken before the refbench harness, with the in-repository oracle (Go go1.27.2
darwin/arm64, median of 9 rounds, load 6.7 to 6.0): Go / NumPy between 0.16 and
0.95 on every `polyval`, `polyfit`, `chebval`, `chebfit`, `chebvander` and `roots`
row measured (n = 1000 to 10^6); the slowest, `chebfit` deg 10 at n = 10^6, was
0.95.
