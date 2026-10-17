// Package poly provides polynomial evaluation, arithmetic, least-squares
// fitting and root finding with the semantics of numpy.polyfit, numpy.roots,
// numpy.poly1d helpers and the numpy.polynomial package (Chebyshev, Legendre,
// Hermite, HermiteE and Laguerre series), in pure Go (CGO_ENABLED=0).
//
// # Coefficient order: read this first
//
// Two conventions coexist, exactly as in NumPy, and they are NOT the same:
//
//   - HIGHEST degree first (np.poly1d, np.polyfit, np.roots, np.polyval ...):
//     Polyval, PolyvalComp, Polyder, Polyint, Conv, Deconv, Polyadd, Polysub,
//     Polymul, PolyFromRoots, Polyfit, Roots. The slice {1, -3, 2} is
//     x^2 - 3x + 2.
//   - LOWEST degree first (numpy.polynomial): Basis, Series, Val, Vander,
//     Der, Int, FitBasis, FitSeries, RootsBasis. The slice {2, -3, 1} is
//     2 + (-3) B_1(x) + B_2(x), B_k being the k-th basis polynomial.
//
// # Faithfulness to NumPy
//
// Evaluation, Vandermonde matrices, differentiation and integration follow
// NumPy's arithmetic operation by operation, each product rounded before its
// sum (explicit float64 conversions forbid fused multiply-add), so their
// results are bit-identical to NumPy's; the README lists the measured
// equality. Least-squares fits use Householder QR on the column-scaled
// design matrix, NumPy's SVD answer being recovered through
// github.com/go-ndarray/ndarray's Lstsq whenever the fit is rank-deficient or
// nearly so. Roots are eigenvalues of the companion matrix
// (Edelman and Murakami 1995, doi:10.1090/S0025-5718-1995-1262279-2), in
// LAPACK's order; the series companion ("comrade") matrices are NumPy's.
// PolyvalComp is the compensated Horner scheme of Graillat, Langlois and
// Louvet (2009), as accurate as Horner computed in twice the precision.
//
// Where NumPy raises an exception the functions return an error. Only
// float64 data, one-dimensional coefficient and data arrays, and a single
// integer degree are supported (see the README's "Gaps").
package poly
