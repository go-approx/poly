package poly

import (
	"errors"
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/linalg"
)

// ErrNonFinite is returned when a least-squares problem contains NaN or Inf
// (NumPy's lstsq raises LinAlgError there).
var ErrNonFinite = errors.New("poly: non-finite value in least-squares problem")

// ErrRankDeficient is returned when a covariance matrix is requested for a
// rank-deficient fit.
var ErrRankDeficient = errors.New("poly: rank-deficient fit has no covariance")

const eps = 0x1p-52

func dot(a, b []float64) float64 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float64
	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

func axpy(alpha float64, x, y []float64) {
	y = y[:len(x)]
	for i, v := range x {
		y[i] += alpha * v
	}
}

// householderQR overwrites the column-major m-by-n matrix a with its R factor
// (upper triangle; the strict lower part holds the reflector vectors) and
// applies Qᵀ to y. No pivoting; LAPACK dgeqr2 conventions.
func householderQR(a []float64, m, n int, y []float64) {
	for k := 0; k < min(m, n); k++ {
		if m-k == 1 {
			break
		}
		col := a[k*m+k : (k+1)*m]
		v := col[1:]
		xn := math.Sqrt(dot(v, v))
		if xn == 0 {
			continue // H = I
		}
		alpha := col[0]
		beta := -math.Copysign(math.Hypot(alpha, xn), alpha)
		tau := (beta - alpha) / beta
		s := 1 / (alpha - beta)
		for i := range v {
			v[i] *= s
		}
		col[0] = beta
		for j := k + 1; j < n; j++ {
			cj := a[j*m+k : (j+1)*m]
			w := tau * (cj[0] + dot(v, cj[1:]))
			cj[0] -= w
			axpy(-w, v, cj[1:])
		}
		w := tau * (y[k] + dot(v, y[k+1:m]))
		y[k] -= w
		axpy(-w, v, y[k+1:m])
	}
}

type lsq struct {
	x      []float64 // minimum-norm least-squares solution (length n)
	rank   int
	resid  []float64 // squared residual norm, only when rank == n and m > n
	r      []float64 // column-major m*n after QR (valid R in the leading n*n block when m >= n)
	viaSVD bool      // the rank decision was delegated to linalg.Lstsq
}

func finite(v []float64) bool {
	s := 0.0
	for _, e := range v {
		s += e * 0
	}
	return !math.IsNaN(s) // NaN or Inf anywhere gives NaN
}

// lstsqQR solves min ||a c - y|| for the column-major m-by-n matrix a (both a
// and y are overwritten). It uses Householder QR; when the diagonal of R
// comes within four decades of the rank cutoff rcond (or m < n) the rank
// decision and the minimum-norm solution are delegated to linalg.Lstsq on the
// triangular factor, which has the singular values of a and so gives NumPy's
// SVD-based answer.
func lstsqQR(a []float64, m, n int, y []float64, rcond float64) (lsq, error) {
	if !finite(a) {
		return lsq{}, ErrNonFinite
	}
	nanY := !finite(y)
	if nanY {
		// NumPy's SVD solve yields NaN coefficients (and a NaN residual) but
		// still decides the rank from the matrix alone.
		y = make([]float64, len(y))
	}
	l, err := lstsqFinite(a, m, n, y, rcond)
	if nanY && err == nil {
		for i := range l.x {
			l.x[i] = math.NaN()
		}
		if l.resid != nil {
			l.resid[0] = math.NaN()
		}
	}
	return l, err
}

func lstsqFinite(a []float64, m, n int, y []float64, rcond float64) (lsq, error) {
	householderQR(a, m, n, y)
	k := min(m, n)
	eff := rcond
	if eff <= 0 || eff >= 1 {
		eff = eps
	}
	if m >= n {
		dmin, dmax := math.Inf(1), 0.0
		for i := 0; i < n; i++ {
			d := math.Abs(a[i*m+i])
			dmin, dmax = math.Min(dmin, d), math.Max(dmax, d)
		}
		if dmax > 0 && dmin > 1e4*eff*dmax {
			x := make([]float64, n)
			copy(x, y[:n])
			for i := n - 1; i >= 0; i-- {
				x[i] /= a[i*m+i]
				axpy(-x[i], a[i*m:i*m+i], x[:i])
			}
			out := lsq{x: x, rank: n, r: a}
			if m > n {
				out.resid = []float64{dot(y[n:m], y[n:m])}
			}
			return out, nil
		}
	}
	rm := make([]float64, k*n)
	for j := 0; j < n; j++ {
		for i := 0; i <= min(j, k-1); i++ {
			rm[i*n+j] = a[j*m+i]
		}
	}
	am, err := ndarray.FromData(rm, k, n)
	if err != nil {
		return lsq{}, err
	}
	bm, err := ndarray.FromData(y[:k], k)
	if err != nil {
		return lsq{}, err
	}
	res, err := linalg.LstsqRcond(am, bm, rcond)
	if err != nil {
		return lsq{}, err
	}
	x := make([]float64, n)
	for i := range x {
		x[i] = res.X.At(i)
	}
	out := lsq{x: x, rank: res.Rank, r: a, viaSVD: true}
	if res.Rank == n && m > n {
		out.resid = []float64{dot(y[n:m], y[n:m])}
	}
	return out, nil
}

// rinvRinvT returns R⁻¹R⁻ᵀ (row-major n*n) for the upper-triangular R held in
// the leading n*n block of the column-major m-row matrix a.
func rinvRinvT(a []float64, m, n int) []float64 {
	ri := make([]float64, n*n) // R^-1, row-major, upper triangular
	for j := 0; j < n; j++ {
		ri[j*n+j] = 1 / a[j*m+j]
		for i := j - 1; i >= 0; i-- {
			s := 0.0
			for l := i + 1; l <= j; l++ {
				s += a[l*m+i] * ri[l*n+j]
			}
			ri[i*n+j] = -s / a[i*m+i]
		}
	}
	v := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			s := 0.0
			for l := j; l < n; l++ {
				s += ri[i*n+l] * ri[j*n+l]
			}
			v[i*n+j], v[j*n+i] = s, s
		}
	}
	return v
}
