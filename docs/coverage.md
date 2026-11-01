# Coverage against NumPy, SciPy and MATLAB (polynomials)

Inventory of what `go-approx/poly` (main, v0.1.0) implements against
`numpy.polynomial` (NumPy 2.5.3), the legacy `numpy` polynomial functions
(`poly`, `roots`, `polyfit`, `poly1d` family), the polynomial parts of
`scipy.interpolate`/`scipy.special`/`scipy.signal`, MATLAB polynomial functions
and the capabilities of Chebfun-style numerical computing with functions.
Interpolation and splines are in the sibling module `go-approx/interp`
(`docs/coverage.md` there). Status: **done** (same semantics and options that
matter), **partial** (what is missing is named), **missing**. "Dep" is what a
missing item needs beyond this package.

Sources. NumPy: `numpy/polynomial/__init__.py` at tag v2.5.3 (its list of
convenience classes and of the methods common to all of them, row by row); the
per-module function families (`polyval`, `chebmul`, `leggauss`, ...) are from
knowledge of the module documentation, not from the source, and need a re-check
against `numpy/polynomial/{polynomial,chebyshev,legendre,laguerre,hermite,hermite_e,polyutils}.py`
before a slice is started. MATLAB: the MathWorks function lists could not be
read by the tool used (client-rendered pages), so the MATLAB column is from
knowledge of the "Polynomials" family and must be re-checked. Chebfun: from its
documented capabilities (chebfun.org), names only; no Chebfun source was read.
Go side: README "Gaps", `go doc`, and the sources on main.

Go conventions to keep in mind: the `poly1d` family (`Polyval`, `Polyfit`, ...)
takes highest degree first, the `numpy.polynomial` family (`Basis`, `Series`)
lowest degree first, as NumPy does.

## Totals

| | done | partial | missing | rows |
|---|---|---|---|---|
| NumPy 2.5.3 / SciPy | 12 | 12 | 28 | 52 |
| MATLAB and Chebfun | 10 | 8 | 25 | 43 |

The first count uses the NumPy/SciPy column of the tables, the second the
MATLAB column (rows with a Chebfun name carry it in that column).

## Legacy `numpy` polynomial functions (`poly1d` family)

| # | NumPy / SciPy | MATLAB / Chebfun | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 1 | `polyval` | `polyval` | `Polyval`, `PolyvalAll`, `PolyvalComp` | partial | complex coefficients or `x` (README) |
| 2 | `polyder`, `polyint` | `polyder`, `polyint` | `Polyder`, `Polyint` | done | MATLAB `polyder(a,b)` and `[q,d]=polyder(b,a)` forms |
| 3 | `polyadd`, `polysub`, `polymul`, `polydiv` | `conv`, `deconv` | `Polyadd`, `Polysub`, `Polymul`, `Conv`, `Deconv` | done | `Conv` not bit-identical to `np.convolve` (summation order) |
| 4 | `poly` (from roots) | `poly` | `PolyFromRoots`, `PolyFromComplexRoots` | partial | matrix input (characteristic polynomial; needs a dense eigensolver) |
| 5 | `roots` | `roots` | `Roots` | done | |
| 6 | `polyfit` (weights, `rcond`, `full`, `cov`) | `polyfit` | `Polyfit` | partial | list of degrees (NumPy 2.x), MATLAB's `S` and `mu` outputs (centring and scaling) |
| 7 | `poly1d` class | | | missing | a `Poly` type for the power basis with operators, `deriv`, `integ`, `roots`, `order`, `coeffs` (see row 20) |
| 8 | `scipy.interpolate.lagrange` | | | missing | none (power-basis Lagrange; ill-conditioned above degree about 20, the barycentric form is in `go-approx/interp` slice 2) |
| 9 | | `polyvalm` | | missing | dense matrix product (go-ndarray) |
| 10 | | `polyeig` | | missing | dense generalised eigensolver |
| 11 | | `polyxpoly`, `polyaffine`, `poly2sym`, `sym2poly` | | missing | `polyaffine` is `MapDomain`-like (partial overlap); the symbolic ones are out of scope; `polyxpoly` is geometry |
| 12 | `scipy.signal.residue`, `residuez`, `invres` | `residue` | | missing | partial fractions; lives in `go-dsp/signal` |
| 13 | `numpy.convolve` | `conv` | `Conv` | done | |
| 14 | `numpy.polynomial.polynomial.polyvalfromroots` | | | missing | none |

## `numpy.polynomial`: convenience classes and common methods

Rows follow the lists in `numpy/polynomial/__init__.py`.

| # | NumPy | MATLAB / Chebfun | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 15 | `Chebyshev`, `Legendre`, `Laguerre`, `Hermite`, `HermiteE` classes | | `Basis`, `Series` | partial | class operators (below); `Chebyshev` first kind only (second kind `U_n` is not in NumPy either) |
| 16 | `Polynomial` class (power series) | | | missing | add a `Power` value to `Basis` (the `poly1d` family exists, but not as a `Series`) |
| 17 | `Poly.domain`, `Poly.window`, `basis_name`, `maxpower` | | `Basis.Domain`, `Window`, `String` | partial | per-series `domain` and `window` fields on `Series` (`FitSeries` takes a domain), `maxpower` |
| 18 | `Poly.basis(deg)` | | | missing | none |
| 19 | `Poly.identity()` | | | missing | none |
| 20 | `Poly.fit(x, y, deg)` | `polyfit`, `chebfun` fit | `FitSeries`, `FitBasis` | partial | list of degrees, `domain=None` default (data range) is done, `full=True` diagnostics, `symbol` |
| 21 | `Poly.fromroots(roots)` | | | missing | `Chebfromroots`-style per basis: multiply linear factors (needs row 25 `Mul`) |
| 22 | `p.copy()` | | | missing | not applicable (value type) |
| 23 | `p.cast(Poly)`, `p.convert(Poly)` | | | missing | basis conversion (`cheb2poly`, `poly2cheb`, `leg2poly`, ... generalised): via power basis, or direct recurrences; domain/window mapping via `MapDomain` |
| 24 | `p.deriv()`, `p.integ()` | `diff`, `cumsum` (Chebfun) | `Series.Der`, `Series.Int` | done | |
| 25 | `p + q`, `-`, `*`, `//`, `%`, `divmod`, `**`, call `p(x)` | `+`, `*`, `plus`, `mtimes`, `.^` (Chebfun) | `Series.Val` only | partial | series arithmetic per basis (`chebadd`, `chebsub`, `chebmul` by linearisation or companion, `chebdiv`, `chebpow`), `mulx`; scalar multiplication |
| 26 | `Poly.has_samecoef`, `has_samedomain`, `has_sametype`, `has_samewindow` | | | missing | with row 17 |
| 27 | `p.linspace()` | `linspace` | | missing | none |
| 28 | `p.mapparms()` | | `MapParms` | done | |
| 29 | `p.roots()` | `roots` | `Series.Roots`, `RootsBasis` | done | |
| 30 | `p.trim()`, `p.cutdeg()`, `p.truncate()` | `simplify`, `chebfun` truncate | | missing | none |
| 31 | `polyutils.getdomain`, `mapdomain`, `mapparms`, `trimcoef`, `trimseq`, `as_series`, `pu.RankWarning`, `PolyError`, `PolyDomainError` | | `GetDomain`, `MapDomain`, `MapParms`, `ErrRankDeficient` | partial | `trimcoef`, `trimseq`, `as_series` |

## `numpy.polynomial`: per-basis function families

For each basis `X` in `{poly, cheb, leg, lag, herm, herme}` NumPy provides the
functions below. The Go column names the generic function over `Basis`.

| # | NumPy | MATLAB / Chebfun | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 32 | `Xval` | `polyval`, `feval` | `Val`, `ValAll` | done | complex input |
| 33 | `Xval2d`, `Xval3d`, `Xgrid2d`, `Xgrid3d`, `Xvalfromroots` | `chebfun2` evaluation | | missing | none (tensor Clenshaw) |
| 34 | `Xvander` | | `Vander` | done | |
| 35 | `Xvander2d`, `Xvander3d` | | | missing | none (Kronecker of 1-D Vandermondes) |
| 36 | `Xder`, `Xint` | `diff`, `cumsum` | `Der`, `Int` | done | |
| 37 | `Xroots` | `roots` (colleague matrix) | `RootsBasis` | done | |
| 38 | `Xfit` | `polyfit` | `FitBasis` | partial | list of degrees, `full` |
| 39 | `Xadd`, `Xsub`, `Xmul`, `Xdiv`, `Xpow`, `Xmulx` | `plus`, `times` (Chebfun) | | missing | none for `add`/`sub`/`mulx`; `mul`, `div` via linearisation (`cheb2poly`-free) or convolution of coefficients in the power basis |
| 40 | `Xline` | | | missing | none |
| 41 | `Xcompanion` | `compan` | internal | missing | export the companion/colleague matrix (needs only the dense eigensolver `RootsBasis` already uses) |
| 42 | `Xfromroots` | `poly` | | missing | row 39 |
| 43 | `Xzero`, `Xone`, `Xx`, `Xdomain`, `Xtrim`, `X2poly`, `poly2X` | | `Basis.Domain` | partial | conversion functions (row 23), constants (trivial) |
| 44 | `leggauss`, `hermgauss`, `hermegauss`, `laggauss`, `chebgauss` | `legpts`, `hermpts`, `lagpts`, `jacpts`, `chebpts` (quadrature form) | | missing | Golub-Welsch (symmetric tridiagonal eigenproblem; dense eig) or Glaser-Liu-Rokhlin / Bogaert asymptotics for large `n`; `scipy.special.roots_legendre` etc. are the SciPy twins and belong in `go-numerics/special` |
| 45 | `Xweight` | | | missing | none |
| 46 | `chebinterpolate(func, deg)` | `chebfun(f, n)` | | missing | DCT (go-fft) or direct summation; the pure-Go path is `FitBasis` on Chebyshev points |
| 47 | `chebpts1`, `chebpts2` | `chebpts` (first and second kind) | | missing | none |

## SciPy polynomial-adjacent

| # | SciPy | MATLAB / Chebfun | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 48 | `scipy.special.eval_chebyt`, `eval_legendre`, `eval_hermite`, `eval_laguerre`, `eval_jacobi`, `eval_gegenbauer`, ... | `chebyshevT`, `legendreP`, `hermiteH`, `laguerreL`, `jacobiP`, `gegenbauerC` (Symbolic) | `Val` on five bases | partial | Jacobi, Gegenbauer, shifted and generalised-Laguerre families (go-numerics/special owns these) |
| 49 | `scipy.special.orthogonal.roots_*` | | | missing | row 44 |
| 50 | `scipy.interpolate.BPoly`, `PPoly` | `mkpp` | | missing | in `go-approx/interp` |
| 51 | `numpy.vander` | `vander` | | missing | none (power-basis Vandermonde; `Vander` over a `Power` basis, row 16) |
| 52 | `numpy.polynomial.polynomial.polygrid2d` | | | missing | row 33 |

## Chebfun-style capabilities (names only)

Chebfun represents a function by an adaptively chosen Chebyshev series. None of
it is provided; everything below builds on `Basis(Chebyshev)`.

| # | NumPy / SciPy | MATLAB / Chebfun | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 53 | `chebinterpolate` (fixed degree) | `chebfun(f)` adaptive construction (doubling `n` until coefficients decay to machine precision, plateau detection) | | missing | row 46; a `Func` type holding coefficients on a domain; DCT from go-fft |
| 54 | | `chebpts`, `chebcoeffs`, `chebfun` on intervals and piecewise (breakpoints) | | missing | row 47; piecewise composition |
| 55 | `Series.Roots` | `roots(f)` with subdivision for degree above 50 | `Series.Roots` | partial | recursive subdivision for high degree (colleague matrix is unstable above about 100) |
| 56 | `Series.Int` | `sum`, `cumsum` | `Series.Int` | done | definite integral helper `Integrate(a, b)` is missing (trivial) |
| 57 | | `max`, `min`, `norm`, `diff` | | missing | `Max`/`Min` via roots of the derivative; `Norm` via Clenshaw-Curtis weights |
| 58 | | `remez` (best polynomial approximation) | | missing | Remez exchange on Chebyshev series (the Parks-McClellan exchange exists in `go-dsp/signal` for filters; no shared kernel) |
| 59 | | `ratinterp`, `cf` (Carathéodory-Fejér), `padeapprox`, `aaa` | | missing | dense SVD; `AAA` is also in `go-approx/interp` slice 8 |
| 60 | | `chebop` (linear differential operators, collocation/ultraspherical) | | missing | banded and dense solvers; out of scope for a polynomial module |
| 61 | | `chebfun2`, `chebfun3` (low-rank bivariate approximation) | | missing | row 53 + adaptive cross approximation |
| 62 | | `trigfun` (Fourier series) | | missing | go-fft |
| 63 | | `legpts`, `jacpts`, `lagpts`, `hermpts`, `fejer1`, `fejer2`, `clencurt` | | missing | row 44; Clenshaw-Curtis/Fejér weights via FFT (go-fft) |
| 64 | | `poly`, `poly(f)` conversion of a chebfun to monomial coefficients | | missing | row 23 |

## Proposed slices

Ordered by value per cost; "dep" names what has to exist first.

1. **Power basis as a `Basis` and series arithmetic** (rows 7, 15 to 19, 21, 25, 26, 30, 39, 40, 42, 43, 51): `Power` in `Basis`, `Add`, `Sub`, `Mul`, `Div`, `Pow`, `MulX`, `Line`, `FromRoots`, `Trim`, `Truncate`, `CutDeg`, `Basis(deg)`, `Identity`, per-series domain and window. Dep: none.
2. **Basis conversion** (rows 23, 43, 64): `Convert`/`Cast`, `Cheb2Poly`, `Poly2Cheb`, `Leg2Poly`, ... Dep: slice 1.
3. **Multivariate evaluation** (rows 33, 35, 52): `Val2D`, `Val3D`, `Grid2D`, `Grid3D`, `Vander2D`, `Vander3D`. Dep: none.
4. **Complex coefficients and arguments, list of degrees, `full` fit output, MATLAB `S`/`mu`** (rows 1, 6, 20, 32, 38). Dep: a decision on the scalar constraint (generics over `float64 | complex128` as in `go-sparse/sparse`).
5. **Quadrature and points** (rows 44, 45, 47, 49, 63): `Leggauss`, `Hermgauss`, `Hermegauss`, `Laggauss`, `Chebgauss`, `Chebpts1/2`, weights. Dep: dense symmetric tridiagonal eigensolver (Golub-Welsch) or an own implicit QL; large `n` via asymptotic methods written from the papers.
6. **Companion matrix, `polyvalm`, `polyeig`, `poly(A)`** (rows 4, 9, 10, 41): Dep: dense eigensolver and matrix product from the dense linear algebra library.
7. **Chebfun core** (rows 46, 53 to 57): `Chebinterpolate`, adaptive construction, `Integrate`, `Max`/`Min`, `Norm`, subdivided `Roots`. Dep: slices 1 and 5, go-fft DCT.
8. **Best approximation and rational** (rows 58, 59): `Remez`, `AAA`, `CF`, `Ratinterp`. Dep: dense SVD; `AAA` shared with `go-approx/interp`.
9. **Out of scope here, filed elsewhere** (rows 8, 12, 48, 50, 60 to 62): Lagrange and barycentric forms to `go-approx/interp`; partial fractions to `go-dsp/signal`; Jacobi/Gegenbauer to `go-numerics/special`; `chebop`, `chebfun2`, `trigfun` to a separate function-approximation module if wanted.
