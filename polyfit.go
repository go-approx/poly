package poly

import "math"

// CovMode selects the covariance estimate returned by Polyfit, as
// numpy.polyfit's cov argument.
type CovMode int

const (
	// CovNone returns no covariance matrix.
	CovNone CovMode = iota
	// CovScaled scales inv(AᵀA) by the residual variance SSR/(m-n),
	// numpy.polyfit(cov=True); it needs m > n.
	CovScaled
	// CovUnscaled returns inv(AᵀA) (numpy.polyfit(cov="unscaled")).
	CovUnscaled
)

// FitOptions are the optional arguments of Polyfit.
type FitOptions struct {
	// W are the weights: rows of the Vandermonde matrix AND the right-hand
	// side are multiplied by W[i], as NumPy does (so W[i] = 1/sigma[i], not
	// 1/sigma²). nil means unweighted.
	W []float64
	// Rcond is the relative cutoff on singular values; 0 means NumPy's
	// default len(x)*eps.
	Rcond float64
	// Cov selects the covariance matrix.
	Cov CovMode
}

// Fit is the result of a least-squares fit.
type Fit struct {
	// Coef are the coefficients: HIGHEST degree first for Polyfit, LOWEST
	// degree first for FitBasis.
	Coef []float64
	// Rank is the effective rank of the (scaled) design matrix.
	Rank int
	// Residuals is [sum of squared residuals] when the rank is full and there
	// are more points than coefficients, empty otherwise (as numpy's lstsq).
	Residuals []float64
	// Cov is the row-major (deg+1)x(deg+1) covariance matrix, in the order of
	// Coef, or nil when not requested.
	Cov []float64
}

// Polyfit is numpy.polyfit(x, y, deg, rcond, w=, cov=): the least-squares
// polynomial of degree deg, coefficients HIGHEST degree first. The
// Vandermonde matrix columns are scaled to unit 2-norm, then solved by
// Householder QR (see lstsqQR for the rank-deficient path).
func Polyfit(x, y []float64, deg int, o *FitOptions) (Fit, error) {
	if o == nil {
		o = &FitOptions{}
	}
	m, n := len(x), deg+1
	switch {
	case deg < 0 || m == 0 || len(y) != m || (o.W != nil && len(o.W) != m):
		return Fit{}, ErrArg
	}
	rcond := o.Rcond
	if rcond == 0 {
		rcond = float64(m) * eps
	}
	a := make([]float64, m*n)
	// numpy.vander(x, n): column j holds x^(n-1-j), by repeated multiplication.
	for i, xv := range x {
		p := 1.0
		for j := n - 1; j >= 0; j-- {
			a[j*m+i] = p
			p *= xv
		}
	}
	rhs := append([]float64(nil), y...)
	if o.W != nil {
		for j := 0; j < n; j++ {
			col := a[j*m : (j+1)*m]
			for i := range col {
				col[i] *= o.W[i]
			}
		}
		for i := range rhs {
			rhs[i] *= o.W[i]
		}
	}
	scale := make([]float64, n)
	for j := range scale {
		col := a[j*m : (j+1)*m]
		scale[j] = math.Sqrt(dot(col, col))
		for i := range col {
			col[i] /= scale[j]
		}
	}
	return finishFit(a, m, n, rhs, rcond, scale, o.Cov)
}

// finishFit runs the QR solve and builds the Fit, undoing the column scaling.
func finishFit(a []float64, m, n int, rhs []float64, rcond float64, scale []float64, cov CovMode) (Fit, error) {
	if cov == CovScaled && m <= n {
		return Fit{}, ErrArg
	}
	l, err := lstsqQR(a, m, n, rhs, rcond)
	if err != nil {
		return Fit{}, err
	}
	for j := range l.x {
		l.x[j] /= scale[j]
	}
	f := Fit{Coef: l.x, Rank: l.rank, Residuals: l.resid}
	if cov != CovNone {
		if l.rank != n {
			return Fit{}, ErrRankDeficient
		}
		v := rinvRinvT(l.r, m, n)
		fac := 1.0
		if cov == CovScaled {
			fac = l.resid[0] / float64(m-n)
		}
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				v[i*n+j] = v[i*n+j] / float64(scale[i]*scale[j]) * fac
			}
		}
		f.Cov = v
	}
	return f, nil
}
