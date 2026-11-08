# Public functions without a performance case

Read by `refbench` in the CI mode (`cd bench && go test -run TestPerf .`): every
public function of the module has a case in `bench/perf_test.go` or is named here
with its reason. The method and the table are in [perf.md](perf.md).

- `PolyvalComp`: the compensated Horner scheme of Graillat, Langlois and Louvet (2009); NumPy has no compensated evaluation (`numpy.polyval` is plain Horner, measured as `Polyval`). The function costs about twice Horner's operations by design and buys twice the working precision.
- `PolyvalCompAll`: as `PolyvalComp`.
- `Basis.Domain`: a constant lookup; NumPy's counterpart is a class attribute, so there is no call to compare.
- `Basis.Window`: as `Basis.Domain`.
