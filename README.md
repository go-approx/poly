# poly

Polynomials in pure Go (`CGO_ENABLED=0`): evaluation, arithmetic, least-squares
fitting, roots, and the Chebyshev, Legendre, Hermite, HermiteE and Laguerre
bases, with the semantics of `numpy.polyfit`, `numpy.roots`, the `numpy.poly1d`
helpers and the `numpy.polynomial` package.

```go
import "github.com/go-approx/poly"

fit, _ := poly.Polyfit(x, y, 3, nil)        // highest degree first, like np.polyfit
r, _ := poly.Roots(fit.Coef)                // companion-matrix eigenvalues
v := poly.Polyval(fit.Coef, 0.5)            // Horner; PolyvalComp is compensated

s, _, _ := poly.FitSeries(poly.Chebyshev, x, y, 8, nil) // lowest degree first, like numpy.polynomial
y0, _ := s.Val(0.3)
```

**Coefficient order differs between the two families, as in NumPy**: the
`poly1d` family is highest degree first, the `numpy.polynomial` family lowest
degree first. See `doc.go`.

## What is provided

| here | NumPy | SciPy / MATLAB |
|---|---|---|
| `Polyval`, `PolyvalAll` | `np.polyval` | MATLAB `polyval` |
| `PolyvalComp`, `PolyvalCompAll` | (none) compensated Horner, Graillat-Langlois-Louvet 2009 | |
| `Polyder`, `Polyint` | `np.polyder`, `np.polyint` | `polyder`, `polyint` |
| `Conv`, `Deconv`, `Polymul`, `Polyadd`, `Polysub` | `np.convolve`, `np.polydiv`, `np.polymul`, `np.polyadd`, `np.polysub` | `conv`, `deconv` |
| `PolyFromRoots`, `PolyFromComplexRoots` | `np.poly` | `poly` |
| `Polyfit` (weights, rcond, covariance scaled or unscaled, rank) | `np.polyfit` | `polyfit` |
| `Roots` | `np.roots` | `roots` |
| `Basis` (`Chebyshev`, `Legendre`, `Hermite`, `HermiteE`, `Laguerre`) with `Val`, `Vander`, `Der`, `Int`, `RootsBasis`, `FitBasis` | `chebval`, `chebvander`, `chebder`, `chebint`, `chebroots`, `chebfit` and the `leg`, `herm`, `herme`, `lag` equivalents | |
| `Series` (`NewSeries`, `FitSeries`, `Val`, `Der`, `Int`, `Roots`), `MapParms`, `MapDomain`, `GetDomain` | `numpy.polynomial.Chebyshev` etc. with domain/window mapping | |

## Verification

Reference: NumPy 2.5.4 (SciPy 1.18.1 installed alongside), run as an oracle from
Go (`internal/oracle` starts `uv run --with numpy==2.5.4 python -c ...`, JSON
over a pipe; the Python is only the minimal call). Octave is not installed on the
host, so no Octave oracle was run. A tagged test (`-tags oracle`) generates the
fixtures in `testdata/` (deterministic PCG inputs, large cases regenerated from
their seed); the ordinary tests read them and need no Python.

Edge cases: degree 0, degree at or above the number of points, 1 and 2 points,
duplicate and unsorted x, NaN and Inf, zero weights, x spanning 1e-8..1e8,
ill-conditioned degree-20 fits, repeated and clustered roots, leading and
trailing zero coefficients, complex-conjugate roots, n up to 1e6. Where NumPy
raises, the Go function returns an error, and the fixture records it.

Measured against NumPy (`go test -v -run 'TestOracle|TestFixtureInventory'`):

| quantity | result |
|---|---|
| `Polyval`, `Polyder`, `Polyint`, `Polyadd`, `Polysub`, `Deconv`, `PolyFromRoots` | 3434 of 3434 values **bit-identical** |
| `Conv`, `Polymul` | 229 of 917 bit-identical; max error 0.90 eps relative to sum of abs(a_i b_j) (NumPy's convolve uses a different summation order) |
| `PolyvalComp` | worst error / theoretical bound = 0.00095; more accurate than Horner at 677 of 780 points |
| each of the 5 bases: `Val`, `Vander`, `Der`, `Int`, `Series.Val/Der/Int` | 10016 of 10016 values **bit-identical** |
| `Roots` | 8 of 24 cases identical in value and order; well-conditioned random degree <= 50: max positional error 4e-15; backward error at most 2.24 times NumPy's |
| basis roots (sorted as NumPy sorts) | max positional error 6e-15 .. 8e-13 over 48 cases per basis |
| `Polyfit` | 78 fits; coefficient deviation at most 1.67 conditioning units, SSR deviation at most 167 eps y^2 |
| basis fits | 80 fits per basis; coefficient deviation 1.2 .. 24.8 conditioning units |

Bit-identity holds on every architecture because every product that feeds a sum
is rounded explicitly (`float64(a*b)`), which forbids fused multiply-add.
A "conditioning unit" is eps times the condition-number-based bound of the
least-squares perturbation theory, so a deviation of a few units means the two
solvers (Householder QR here, SVD in NumPy) agree to the accuracy the problem
allows.

Gates: `gofmt -l .` empty, `go vet`, `go test -race`, 100.0% statement coverage
with `-coverpkg=./...`, and the CI described in `.github/workflows/ci.yml`
(Linux, macOS, Windows, arm64, qemu riscv64/loong64/ppc64le/s390x/arm, 386,
every GOOS/GOARCH compiled, govulncheck).

## Performance

See `docs/perf.md`. The comparison is Go against NumPy on one core, interleaved
rounds. It is only meaningful on an idle machine; the table states the load it
was taken at.

## Gaps

- float64 only; one-dimensional coefficient and data arrays; a single integer
  degree (no list of degrees as `np.polyfit` allows in 2.x).
- No complex-coefficient `Polyval` or fits (`PolyFromComplexRoots` returns complex
  coefficients).
- No `numpy.polynomial.Polynomial` (power-basis series) class beyond the
  `poly1d` functions.
- `Conv` is not bit-identical to `np.convolve` (summation order).
- No Octave oracle (Octave is not installed on the build host).

## Licence

BSD-3-Clause.
