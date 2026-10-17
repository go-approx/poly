package poly

import "math"

// Series is numpy.polynomial's ABCPolyBase: coefficients LOWEST degree first,
// in one of the Basis families, with a Domain mapped linearly onto the Window
// where the basis polynomials live.
type Series struct {
	Coef   []float64
	Domain [2]float64
	Window [2]float64
	Basis  Basis
}

// NewSeries returns a series with the family's default domain and window. The
// coefficients are copied; an empty slice is an error (NumPy raises).
func NewSeries(b Basis, coef []float64) (Series, error) {
	if len(coef) == 0 {
		return Series{}, ErrEmpty
	}
	if b < Chebyshev || b > Laguerre {
		return Series{}, ErrBasis
	}
	return Series{Coef: append([]float64(nil), coef...), Domain: b.Domain(), Window: b.Window(), Basis: b}, nil
}

// Val evaluates the series at x: x is first mapped from Domain to Window.
func (s Series) Val(x float64) (float64, error) {
	off, scl := MapParms(s.Domain, s.Window)
	return Val(s.Basis, s.Coef, off+float64(scl*x))
}

// ValAll evaluates the series at every element of x.
func (s Series) ValAll(dst, x []float64) ([]float64, error) {
	xn := MapDomain(nil, x, s.Domain, s.Window)
	return ValAll(s.Basis, dst, s.Coef, xn)
}

// Der returns the m-th derivative (ABCPolyBase.deriv: derivatives scaled by
// the domain-to-window slope).
func (s Series) Der(m int) (Series, error) {
	_, scl := MapParms(s.Domain, s.Window)
	c, err := Der(s.Basis, s.Coef, m, scl)
	s.Coef = c
	return s, err
}

// Int returns the m-fold integral (ABCPolyBase.integ): integrals are scaled
// by 1/slope; lbnd, when non-nil, is a point of the DOMAIN mapped to the
// window (nil means the window origin 0, as NumPy's lbnd=None).
func (s Series) Int(m int, k []float64, lbnd *float64) (Series, error) {
	off, scl := MapParms(s.Domain, s.Window)
	lb := 0.0
	if lbnd != nil {
		lb = off + float64(scl**lbnd)
	}
	c, err := Int(s.Basis, s.Coef, m, k, lb, 1./scl)
	s.Coef = c
	return s, err
}

// Roots returns the roots of the series in the domain, sorted by real part
// then imaginary part.
func (s Series) Roots() ([]complex128, error) {
	r, err := RootsBasis(s.Basis, s.Coef)
	if err != nil {
		return nil, err
	}
	off, scl := MapParms(s.Window, s.Domain)
	for i, z := range r {
		r[i] = complex(off+float64(scl*real(z)), float64(scl*imag(z)))
	}
	return r, nil
}

// SeriesFitOptions are the optional arguments of FitSeries.
type SeriesFitOptions struct {
	// Domain, when nil, defaults to [min(x), max(x)] (widened to
	// [min-1, max+1] when they are equal, as NumPy).
	Domain *[2]float64
	// Window, when nil, is the family's default window.
	Window *[2]float64
	// W are the weights (rows of the design matrix and y are multiplied).
	W []float64
	// Rcond is the singular-value cutoff; 0 means len(x)*eps.
	Rcond float64
}

// FitBasis is numpy.polynomial.<basis>.<basis>fit: a least-squares fit in the
// given family WITHOUT any domain mapping (x is used as is). The columns of
// the pseudo-Vandermonde matrix are scaled to unit norm (a zero column keeps
// scale 1), then solved by the same Householder QR as Polyfit. Coefficients
// are lowest degree first.
func FitBasis(b Basis, x, y []float64, deg int, w []float64, rcond float64) (Fit, error) {
	m, n := len(x), deg+1
	if deg < 0 || m == 0 || len(y) != m || (w != nil && len(w) != m) {
		return Fit{}, ErrArg
	}
	v, err := Vander(b, x, deg)
	if err != nil {
		return Fit{}, err
	}
	if rcond == 0 {
		rcond = float64(m) * eps
	}
	a := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			e := v[i*n+j]
			if w != nil {
				e *= w[i]
			}
			a[j*m+i] = e
		}
	}
	rhs := append([]float64(nil), y...)
	if w != nil {
		for i := range rhs {
			rhs[i] *= w[i]
		}
	}
	scale := make([]float64, n)
	for j := range scale {
		col := a[j*m : (j+1)*m]
		scale[j] = math.Sqrt(dot(col, col))
		if scale[j] == 0 {
			scale[j] = 1
		}
		for i := range col {
			col[i] /= scale[j]
		}
	}
	return finishFit(a, m, n, rhs, rcond, scale, CovNone)
}

// FitSeries is ABCPolyBase.fit (e.g. Chebyshev.fit): the domain is taken from
// x unless given, x is mapped onto the window, and the fit is FitBasis.
func FitSeries(b Basis, x, y []float64, deg int, o *SeriesFitOptions) (Series, Fit, error) {
	if o == nil {
		o = &SeriesFitOptions{}
	}
	var dom [2]float64
	if o.Domain != nil {
		dom = *o.Domain
	} else {
		var err error
		if dom, err = GetDomain(x); err != nil {
			return Series{}, Fit{}, err
		}
		if dom[0] == dom[1] {
			dom[0]--
			dom[1]++
		}
	}
	win := b.Window()
	if o.Window != nil {
		win = *o.Window
	}
	xn := MapDomain(nil, x, dom, win)
	f, err := FitBasis(b, xn, y, deg, o.W, o.Rcond)
	if err != nil {
		return Series{}, Fit{}, err
	}
	return Series{Coef: f.Coef, Domain: dom, Window: win, Basis: b}, f, nil
}

// RootsBasis is numpy's chebroots/legroots/hermroots/hermeroots/lagroots: the
// roots of the series c (lowest degree first, trailing zeros trimmed) as
// eigenvalues of the companion (comrade) matrix of the three-term
// recurrence, sorted lexicographically (real, then imaginary part). An empty
// c is an error; NaN/Inf coefficients are an error through the eigensolver.
func RootsBasis(b Basis, c []float64) ([]complex128, error) {
	if len(c) == 0 {
		return nil, ErrEmpty
	}
	if b < Chebyshev || b > Laguerre {
		return nil, ErrBasis
	}
	for len(c) > 1 && c[len(c)-1] == 0 {
		c = c[:len(c)-1]
	}
	if len(c) < 2 {
		return []complex128{}, nil
	}
	if len(c) == 2 {
		switch b {
		case Hermite:
			return []complex128{complex(-.5*c[0]/c[1], 0)}, nil
		case Laguerre:
			return []complex128{complex(1+c[0]/c[1], 0)}, nil
		}
		return []complex128{complex(-c[0]/c[1], 0)}, nil
	}
	n := len(c) - 1
	mat := companion(b, c)
	// numpy rotates the matrix by 180 degrees before eig.
	for i, j := 0, n*n-1; i < j; i, j = i+1, j-1 {
		mat[i], mat[j] = mat[j], mat[i]
	}
	r, err := eigvals(mat, n)
	if err != nil {
		return nil, err
	}
	sortRoots(r)
	return r, nil
}

// companion builds numpy's <basis>companion matrix (row-major n*n,
// n = len(c)-1 >= 2) with its exact arithmetic.
func companion(b Basis, c []float64) []float64 {
	n := len(c) - 1
	mat := make([]float64, n*n)
	top := func(i int) *float64 { return &mat[i*n+i+1] } // (i, i+1)
	bot := func(i int) *float64 { return &mat[(i+1)*n+i] }
	last := func(i int) *float64 { return &mat[i*n+n-1] }
	cl := c[n]
	switch b {
	case Chebyshev:
		scl := make([]float64, n)
		scl[0] = 1
		for i := 1; i < n; i++ {
			scl[i] = math.Sqrt(.5)
		}
		for i := 0; i < n-1; i++ {
			t := 0.5
			if i == 0 {
				t = math.Sqrt(.5)
			}
			*top(i), *bot(i) = t, t
		}
		for i := 0; i < n; i++ {
			*last(i) -= float64(float64(c[i]/cl)*(scl[i]/scl[n-1])) * .5
		}
	case Legendre:
		scl := make([]float64, n)
		for i := range scl {
			scl[i] = 1. / math.Sqrt(float64(2*i+1))
		}
		for i := 0; i < n-1; i++ {
			t := float64(float64(i+1)*scl[i]) * scl[i+1]
			*top(i), *bot(i) = t, t
		}
		f := float64(n) / float64(2*n-1)
		for i := 0; i < n; i++ {
			*last(i) -= float64(float64(c[i]/cl)*(scl[i]/scl[n-1])) * f
		}
	case Hermite, HermiteE:
		scl := make([]float64, n)
		scl[0] = 1
		for i := 1; i < n; i++ {
			d := float64(n - i) // arange(n-1,0,-1)[i-1]
			if b == Hermite {
				d *= 2
			}
			scl[i] = 1. / math.Sqrt(d)
		}
		for i := 1; i < n; i++ { // multiply.accumulate
			scl[i] = scl[i-1] * scl[i]
		}
		for i, j := 0, n-1; i < j; i, j = i+1, j-1 { // [::-1]
			scl[i], scl[j] = scl[j], scl[i]
		}
		for i := 0; i < n-1; i++ {
			var t float64
			if b == Hermite {
				t = math.Sqrt(.5 * float64(i+1))
			} else {
				t = math.Sqrt(float64(i + 1))
			}
			*top(i), *bot(i) = t, t
		}
		for i := 0; i < n; i++ {
			if b == Hermite {
				*last(i) -= float64(scl[i]*c[i]) / (2.0 * cl)
			} else {
				*last(i) -= float64(scl[i]*c[i]) / cl
			}
		}
	case Laguerre:
		for i := 0; i < n-1; i++ {
			*top(i), *bot(i) = -float64(i+1), -float64(i+1)
		}
		for i := 0; i < n; i++ {
			mat[i*n+i] = 2.*float64(i) + 1.
		}
		for i := 0; i < n; i++ {
			*last(i) += float64(c[i]/cl) * float64(n)
		}
	}
	return mat
}
