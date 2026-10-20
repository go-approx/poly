package poly

import (
	"errors"
	"math"
)

// Basis names an orthogonal-polynomial family of numpy.polynomial.
//
// Coefficient order: the Basis / Series API follows numpy.polynomial and
// stores coefficients LOWEST degree first (Coef[i] multiplies the i-th basis
// polynomial). The np.poly1d-style functions (Polyval, Polyfit, Roots, ...)
// store them HIGHEST degree first.
type Basis int

// The supported families.
const (
	Chebyshev Basis = iota // T_n, numpy.polynomial.chebyshev
	Legendre               // P_n, numpy.polynomial.legendre
	Hermite                // physicists' H_n, numpy.polynomial.hermite
	HermiteE               // probabilists' He_n, numpy.polynomial.hermite_e
	Laguerre               // L_n, numpy.polynomial.laguerre
)

// String returns the numpy class name.
func (b Basis) String() string {
	switch b {
	case Chebyshev:
		return "Chebyshev"
	case Legendre:
		return "Legendre"
	case Hermite:
		return "Hermite"
	case HermiteE:
		return "HermiteE"
	case Laguerre:
		return "Laguerre"
	}
	return "Basis(?)"
}

// Domain returns the default domain of the family (numpy class attribute).
func (b Basis) Domain() [2]float64 {
	if b == Laguerre {
		return [2]float64{0, 1}
	}
	return [2]float64{-1, 1}
}

// Window returns the default window of the family.
func (b Basis) Window() [2]float64 { return b.Domain() }

// ErrBasis is returned for an unknown Basis value.
var ErrBasis = errors.New("poly: unknown basis")

// Val evaluates the series sum c[i]*B_i(x) by Clenshaw's recurrence with
// numpy's exact operation order (chebval, legval, hermval, hermeval, lagval):
// every product is rounded before its sum, so results are bit-identical to
// NumPy's. An empty c is an error (NumPy raises IndexError).
func Val(b Basis, c []float64, x float64) (float64, error) {
	if len(c) == 0 {
		return 0, ErrEmpty
	}
	switch b {
	case Chebyshev:
		return chebval(c, x), nil
	case Legendre:
		return legval(c, x), nil
	case Hermite:
		return hermval(c, x), nil
	case HermiteE:
		return hermeval(c, x), nil
	case Laguerre:
		return lagval(c, x), nil
	}
	return 0, ErrBasis
}

// ValAll evaluates the series at every element of x (dst reused if large
// enough).
func ValAll(b Basis, dst, c, x []float64) ([]float64, error) {
	if len(c) == 0 {
		return nil, ErrEmpty
	}
	var f func([]float64, float64) float64
	var f4 func([]float64, float64, float64, float64, float64) (float64, float64, float64, float64)
	switch b {
	case Chebyshev:
		f, f4 = chebval, chebval4
	case Legendre:
		f, f4 = legval, legval4
	case Hermite:
		f, f4 = hermval, hermval4
	case HermiteE:
		f, f4 = hermeval, hermeval4
	case Laguerre:
		f, f4 = lagval, lagval4
	default:
		return nil, ErrBasis
	}
	if len(dst) < len(x) {
		dst = make([]float64, len(x))
	}
	dst = dst[:len(x)]
	i := 0
	if len(c) >= 3 { // the four-lane kernels start from the third coefficient
		for ; i+4 <= len(x); i += 4 {
			dst[i], dst[i+1], dst[i+2], dst[i+3] = f4(c, x[i], x[i+1], x[i+2], x[i+3])
		}
	}
	for ; i < len(x); i++ {
		dst[i] = f(c, x[i])
	}
	return dst, nil
}

func chebval(c []float64, x float64) float64 {
	var c0, c1 float64
	switch len(c) {
	case 1:
		c0 = c[0]
	case 2:
		c0, c1 = c[0], c[1]
	default:
		x2 := 2 * x
		c0, c1 = c[len(c)-2], c[len(c)-1]
		for i := 3; i <= len(c); i++ {
			tmp := c0
			c0 = c[len(c)-i] - c1
			c1 = tmp + float64(c1*x2)
		}
	}
	return c0 + float64(c1*x)
}

func legval(c []float64, x float64) float64 {
	var c0, c1 float64
	switch len(c) {
	case 1:
		c0 = c[0]
	case 2:
		c0, c1 = c[0], c[1]
	default:
		nd := len(c)
		c0, c1 = c[len(c)-2], c[len(c)-1]
		for i := 3; i <= len(c); i++ {
			tmp := c0
			nd--
			c0 = c[len(c)-i] - float64(c1*(float64(nd-1)/float64(nd)))
			c1 = tmp + float64(float64(c1*x)*(float64(2*nd-1)/float64(nd)))
		}
	}
	return c0 + float64(c1*x)
}

func hermval(c []float64, x float64) float64 {
	x2 := x * 2
	var c0, c1 float64
	switch len(c) {
	case 1:
		c0 = c[0]
	case 2:
		c0, c1 = c[0], c[1]
	default:
		nd := len(c)
		c0, c1 = c[len(c)-2], c[len(c)-1]
		for i := 3; i <= len(c); i++ {
			tmp := c0
			nd--
			c0 = c[len(c)-i] - float64(c1*float64(2*(nd-1)))
			c1 = tmp + float64(c1*x2)
		}
	}
	return c0 + float64(c1*x2)
}

func hermeval(c []float64, x float64) float64 {
	var c0, c1 float64
	switch len(c) {
	case 1:
		c0 = c[0]
	case 2:
		c0, c1 = c[0], c[1]
	default:
		nd := len(c)
		c0, c1 = c[len(c)-2], c[len(c)-1]
		for i := 3; i <= len(c); i++ {
			tmp := c0
			nd--
			c0 = c[len(c)-i] - float64(c1*float64(nd-1))
			c1 = tmp + float64(c1*x)
		}
	}
	return c0 + float64(c1*x)
}

func lagval(c []float64, x float64) float64 {
	var c0, c1 float64
	switch len(c) {
	case 1:
		c0 = c[0]
	case 2:
		c0, c1 = c[0], c[1]
	default:
		nd := len(c)
		c0, c1 = c[len(c)-2], c[len(c)-1]
		for i := 3; i <= len(c); i++ {
			tmp := c0
			nd--
			c0 = c[len(c)-i] - float64(c1*float64(nd-1))/float64(nd)
			c1 = tmp + float64(c1*(float64(2*nd-1)-x))/float64(nd)
		}
	}
	return c0 + float64(c1*(1-x))
}

// Vander returns the pseudo-Vandermonde matrix of the family, row-major
// len(x) rows by deg+1 columns: element (i,j) is B_j(x[i]), computed by
// numpy's forward recurrences (bit-identical to chebvander, legvander, ...).
func Vander(b Basis, x []float64, deg int) ([]float64, error) {
	if deg < 0 {
		return nil, ErrArg
	}
	if b < Chebyshev || b > Laguerre {
		return nil, ErrBasis
	}
	n := deg + 1
	v := make([]float64, len(x)*n)
	for i0 := 0; i0 < len(x); i0 += vanderBlock {
		i1 := min(len(x), i0+vanderBlock)
		vanderRows(b, v[i0*n:i1*n], x[i0:i1], n)
	}
	return v, nil
}

// vanderBlock rows are filled together, degree by degree: the recurrence
// chains one row's entries, so interleaving independent rows lets the CPU
// overlap them (the block stays in L1).
const vanderBlock = 64

// vanderRows fills the row-major rows v (len(x) rows of n) with B_0..B_{n-1}
// at each x, by the same operations, in the same order per entry, as
// numpy's forward recurrences.
func vanderRows(b Basis, v, x []float64, n int) {
	for i, xv := range x {
		v[i*n] = xv*0 + 1
	}
	if n == 1 {
		return
	}
	switch b {
	case Chebyshev:
		for i, xv := range x {
			v[i*n+1] = xv
		}
		// Four rows at a time keep their recurrences in registers: four
		// independent chains, contiguous stores, no reads back from the
		// matrix (1.6x faster than filling degree by degree). Same
		// operations, same order per entry.
		i := 0
		for ; i+4 <= len(x); i += 4 {
			r0, r1, r2, r3 := v[i*n:i*n+n], v[(i+1)*n:(i+1)*n+n], v[(i+2)*n:(i+2)*n+n], v[(i+3)*n:(i+3)*n+n]
			x0, x1, x2, x3 := x[i], x[i+1], x[i+2], x[i+3]
			a0, a1, a2, a3 := r0[0], r1[0], r2[0], r3[0]
			b0, b1, b2, b3 := x0, x1, x2, x3
			for k := 2; k < n; k++ {
				c0 := float64(b0*(2*x0)) - a0
				c1 := float64(b1*(2*x1)) - a1
				c2 := float64(b2*(2*x2)) - a2
				c3 := float64(b3*(2*x3)) - a3
				r0[k], r1[k], r2[k], r3[k] = c0, c1, c2, c3
				a0, a1, a2, a3 = b0, b1, b2, b3
				b0, b1, b2, b3 = c0, c1, c2, c3
			}
		}
		for k := 2; k < n; k++ {
			for j := i; j < len(x); j++ {
				r := v[j*n : j*n+n]
				r[k] = float64(r[k-1]*(2*x[j])) - r[k-2]
			}
		}
	case Legendre:
		for i, xv := range x {
			v[i*n+1] = xv
		}
		for k := 2; k < n; k++ {
			a, c, d := float64(2*k-1), float64(k-1), float64(k)
			for i, xv := range x {
				r := v[i*n : i*n+n]
				r[k] = (float64(float64(r[k-1]*xv)*a) - float64(r[k-2]*c)) / d
			}
		}
	case Hermite:
		for i, xv := range x {
			v[i*n+1] = xv * 2
		}
		for k := 2; k < n; k++ {
			c := float64(2 * (k - 1))
			for i, xv := range x {
				r := v[i*n : i*n+n]
				r[k] = float64(r[k-1]*(xv*2)) - float64(r[k-2]*c)
			}
		}
	case HermiteE:
		for i, xv := range x {
			v[i*n+1] = xv
		}
		for k := 2; k < n; k++ {
			c := float64(k - 1)
			for i, xv := range x {
				r := v[i*n : i*n+n]
				r[k] = float64(r[k-1]*xv) - float64(r[k-2]*c)
			}
		}
	case Laguerre:
		for i, xv := range x {
			v[i*n+1] = 1 - xv
		}
		for k := 2; k < n; k++ {
			a, c, d := float64(2*k-1), float64(k-1), float64(k)
			for i, xv := range x {
				r := v[i*n : i*n+n]
				r[k] = (float64(r[k-1]*(a-xv)) - float64(r[k-2]*c)) / d
			}
		}
	}
}

// Der is numpy's chebder/legder/hermder/hermeder/lagder: the m-th derivative
// of the series c (lowest degree first), each derivative scaled by scl
// (scl = d(window)/d(domain) for a variable change).
func Der(b Basis, c []float64, m int, scl float64) ([]float64, error) {
	if m < 0 {
		return nil, ErrArg
	}
	if b < Chebyshev || b > Laguerre {
		return nil, ErrBasis
	}
	c = append([]float64(nil), c...)
	n := len(c)
	if m == 0 || n == 0 {
		return c, nil // numpy returns an empty series unchanged
	}
	if m >= n {
		return []float64{c[0] * 0}, nil
	}
	for ; m > 0; m-- {
		n--
		for i := range c {
			c[i] *= scl
		}
		der := make([]float64, n)
		switch b {
		case Chebyshev:
			for j := n; j > 2; j-- {
				der[j-1] = float64(2*j) * c[j]
				c[j-2] += float64(float64(j)*c[j]) / float64(j-2)
			}
			if n > 1 {
				der[1] = 4 * c[2]
			}
			der[0] = c[1]
		case Legendre:
			for j := n; j > 2; j-- {
				der[j-1] = float64(2*j-1) * c[j]
				c[j-2] += c[j]
			}
			if n > 1 {
				der[1] = 3 * c[2]
			}
			der[0] = c[1]
		case Hermite:
			for j := n; j > 0; j-- {
				der[j-1] = float64(2*j) * c[j]
			}
		case HermiteE:
			for j := n; j > 0; j-- {
				der[j-1] = float64(j) * c[j]
			}
		case Laguerre:
			for j := n; j > 1; j-- {
				der[j-1] = -c[j]
				c[j-1] += c[j]
			}
			der[0] = -c[1]
		}
		c = der
	}
	return c, nil
}

// Int is numpy's chebint/legint/hermint/hermeint/lagint: the m-fold integral
// of the series c (lowest degree first) with integration constants k (missing
// ones are 0), lower bound lbnd, each integral scaled by scl.
func Int(b Basis, c []float64, m int, k []float64, lbnd, scl float64) ([]float64, error) {
	if m < 0 || len(k) > m {
		return nil, ErrArg
	}
	if len(c) == 0 {
		return nil, ErrEmpty
	}
	if b < Chebyshev || b > Laguerre {
		return nil, ErrBasis
	}
	c = append([]float64(nil), c...)
	if m == 0 {
		return c, nil
	}
	kk := make([]float64, m)
	copy(kk, k)
	for i := 0; i < m; i++ {
		n := len(c)
		for j := range c {
			c[j] *= scl
		}
		if n == 1 && c[0] == 0 {
			c[0] += kk[i]
			continue
		}
		tmp := make([]float64, n+1)
		switch b {
		case Chebyshev:
			tmp[0] = c[0] * 0
			tmp[1] = c[0]
			if n > 1 {
				tmp[2] = c[1] / 4
			}
			for j := 2; j < n; j++ {
				tmp[j+1] = c[j] / float64(2*(j+1))
				tmp[j-1] -= c[j] / float64(2*(j-1))
			}
		case Legendre:
			tmp[0] = c[0] * 0
			tmp[1] = c[0]
			if n > 1 {
				tmp[2] = c[1] / 3
			}
			for j := 2; j < n; j++ {
				t := c[j] / float64(2*j+1)
				tmp[j+1] = t
				tmp[j-1] -= t
			}
		case Hermite:
			tmp[0] = c[0] * 0
			tmp[1] = c[0] / 2
			for j := 1; j < n; j++ {
				tmp[j+1] = c[j] / float64(2*(j+1))
			}
		case HermiteE:
			tmp[0] = c[0] * 0
			tmp[1] = c[0]
			for j := 1; j < n; j++ {
				tmp[j+1] = c[j] / float64(j+1)
			}
		case Laguerre:
			tmp[0] = c[0]
			tmp[1] = -c[0]
			for j := 1; j < n; j++ {
				tmp[j] += c[j]
				tmp[j+1] = -c[j]
			}
		}
		v, _ := Val(b, tmp, lbnd) // tmp is never empty
		tmp[0] += kk[i] - v
		c = tmp
	}
	return c, nil
}

// MapParms returns (off, scl) such that x -> off + scl*x maps the interval
// old onto new (numpy.polynomial.polyutils.mapparms).
func MapParms(old, new [2]float64) (off, scl float64) {
	oldlen := old[1] - old[0]
	newlen := new[1] - new[0]
	off = (float64(old[1]*new[0]) - float64(old[0]*new[1])) / oldlen
	scl = newlen / oldlen
	return off, scl
}

// MapDomain maps x from the interval old to new (polyutils.mapdomain), into
// dst (allocated when too short) and returns it.
func MapDomain(dst, x []float64, old, new [2]float64) []float64 {
	off, scl := MapParms(old, new)
	if len(dst) < len(x) {
		dst = make([]float64, len(x))
	}
	dst = dst[:len(x)]
	for i, v := range x {
		dst[i] = off + float64(scl*v)
	}
	return dst
}

// GetDomain returns [min(x), max(x)] (polyutils.getdomain); a NaN in x gives
// NaN, as numpy's min/max. Empty x is an error.
func GetDomain(x []float64) ([2]float64, error) {
	if len(x) == 0 {
		return [2]float64{}, ErrEmpty
	}
	lo, hi := x[0], x[0]
	for _, v := range x {
		if math.IsNaN(v) {
			return [2]float64{v, v}, nil
		}
		lo, hi = math.Min(lo, v), math.Max(hi, v)
	}
	return [2]float64{lo, hi}, nil
}
