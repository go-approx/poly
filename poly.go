package poly

import (
	"errors"
	"math"
	"sort"
)

// ErrEmpty is returned where NumPy raises because an input array is empty.
var ErrEmpty = errors.New("poly: empty input")

// ErrArg is returned for an invalid argument (negative order, short k, ...),
// where NumPy raises ValueError.
var ErrArg = errors.New("poly: invalid argument")

// Polyval evaluates the polynomial p (coefficients HIGHEST degree first) at x
// by Horner's rule, with NumPy's exact arithmetic: y = 0; y = y*x + c for
// every coefficient, each product rounded before the sum (no fused
// multiply-add), so the result is bit-identical to numpy.polyval. As in NumPy
// the start value 0*x makes the result NaN for an infinite x.
func Polyval(p []float64, x float64) float64 {
	y := 0.0
	for _, c := range p {
		y = float64(y*x) + c
	}
	return y
}

// PolyvalAll evaluates p at every element of x, writing into dst (allocated
// when nil or too short) and returning it.
func PolyvalAll(dst, p, x []float64) []float64 {
	if len(dst) < len(x) {
		dst = make([]float64, len(x))
	}
	dst = dst[:len(x)]
	for i, xv := range x {
		y := 0.0
		for _, c := range p {
			y = float64(y*xv) + c
		}
		dst[i] = y
	}
	return dst
}

// twoSum is Knuth's error-free transformation: s+e == a+b exactly.
func twoSum(a, b float64) (s, e float64) {
	s = a + b
	bb := s - a
	e = (a - (s - bb)) + (b - bb)
	return s, e
}

// twoProd is the FMA-based error-free product: p+e == a*b exactly.
func twoProd(a, b float64) (p, e float64) {
	p = float64(a * b) // the conversion forbids fusing p into a later sum
	e = math.FMA(a, b, -p)
	return p, e
}

// PolyvalComp evaluates p (highest degree first) at x with the compensated
// Horner scheme of Graillat, Langlois and Louvet (2009): the result is as
// accurate as Horner's rule computed in twice the working precision, then
// rounded, whatever the arch's FMA fusing (TwoSum and an explicit math.FMA
// TwoProduct are used).
func PolyvalComp(p []float64, x float64) float64 {
	if len(p) == 0 {
		return 0
	}
	s := p[0]
	c := 0.0
	for _, a := range p[1:] {
		pi, pp := twoProd(s, x)
		var sg float64
		s, sg = twoSum(pi, a)
		c = float64(c*x) + (pp + sg)
	}
	return s + c
}

// PolyvalCompAll applies PolyvalComp to every element of x.
func PolyvalCompAll(dst, p, x []float64) []float64 {
	if len(dst) < len(x) {
		dst = make([]float64, len(x))
	}
	dst = dst[:len(x)]
	for i, xv := range x {
		dst[i] = PolyvalComp(p, xv)
	}
	return dst
}

// Polyder is numpy.polyder: the m-th derivative of p (highest degree first).
// The derivative of a constant is the empty slice, as in NumPy.
func Polyder(p []float64, m int) ([]float64, error) {
	if m < 0 {
		return nil, ErrArg
	}
	cur := append([]float64(nil), p...)
	for ; m > 0; m-- {
		n := len(cur) - 1
		if n < 0 {
			n = 0
		}
		y := make([]float64, n)
		for i := range y {
			y[i] = cur[i] * float64(n-i)
		}
		cur = y
	}
	return cur, nil
}

// Polyint is numpy.polyint: the m-fold antiderivative of p (highest degree
// first) with integration constants k (nil: zeros; one value: reused for
// every integration; otherwise at least m values, k[0] being the constant of
// the first integration).
func Polyint(p []float64, m int, k []float64) ([]float64, error) {
	if m < 0 {
		return nil, ErrArg
	}
	if k == nil {
		k = make([]float64, m)
	}
	if len(k) == 1 && m > 1 {
		v := k[0]
		k = make([]float64, m)
		for i := range k {
			k[i] = v
		}
	}
	if len(k) < m {
		return nil, ErrArg
	}
	cur := append([]float64(nil), p...)
	for lvl := 0; lvl < m; lvl++ {
		n := len(cur)
		y := make([]float64, n+1)
		for i, c := range cur {
			y[i] = c / float64(n-i)
		}
		y[n] = k[lvl]
		cur = y
	}
	return cur, nil
}

// Conv is numpy.convolve(a, v) in "full" mode, a direct O(len(a)*len(v))
// sum. Either input empty is an error, as in NumPy.
func Conv(a, v []float64) ([]float64, error) {
	if len(a) == 0 || len(v) == 0 {
		return nil, ErrEmpty
	}
	if len(a) < len(v) {
		a, v = v, a
	}
	out := make([]float64, len(a)+len(v)-1)
	for j, vj := range v {
		o := out[j : j+len(a)]
		for i, ai := range a {
			o[i] += ai * vj
		}
	}
	return out, nil
}

// trimLeading strips leading exact zeros, keeping a single 0 when nothing is
// left, as numpy.poly1d does.
func trimLeading(p []float64) []float64 {
	i := 0
	for i < len(p) && p[i] == 0 {
		i++
	}
	if i == len(p) {
		return []float64{0}
	}
	return p[i:]
}

// Polymul is numpy.polymul: the product of two polynomials (highest degree
// first); like numpy.poly1d it strips leading zeros of both operands first.
func Polymul(a, b []float64) []float64 {
	r, _ := Conv(trimLeading(a), trimLeading(b)) // never empty after trimLeading
	return r
}

func addSub(a, b []float64, sign float64) []float64 {
	n := max(len(a), len(b))
	out := make([]float64, n)
	for i := range out {
		var x, y float64
		if j := i - (n - len(a)); j >= 0 {
			x = a[j]
		}
		if j := i - (n - len(b)); j >= 0 {
			y = b[j]
		}
		out[i] = x + sign*y
	}
	return out
}

// Polyadd is numpy.polyadd (highest degree first, no trimming).
func Polyadd(a, b []float64) []float64 { return addSub(a, b, 1) }

// Polysub is numpy.polysub.
func Polysub(a, b []float64) []float64 { return addSub(a, b, -1) }

// Deconv is numpy.polydiv (scipy.signal.deconvolve's long division): the
// quotient and remainder of u by v, highest degree first. Leading remainder
// entries with |r| <= 1e-8 are dropped, which is what NumPy's
// allclose(r[0], 0, rtol=1e-14) (default atol 1e-8) does. A zero leading
// coefficient of v gives Inf/NaN as in NumPy. Empty inputs are an error.
func Deconv(u, v []float64) (q, r []float64, err error) {
	if len(u) == 0 || len(v) == 0 {
		return nil, nil, ErrEmpty
	}
	m, n := len(u)-1, len(v)-1
	scale := 1 / v[0]
	q = make([]float64, max(m-n+1, 1))
	r = append([]float64(nil), u...)
	for k := 0; k < m-n+1; k++ {
		d := scale * r[k]
		q[k] = d
		for j, vj := range v {
			r[k+j] -= float64(d * vj)
		}
	}
	for len(r) > 1 && math.Abs(r[0]) <= 1e-8 {
		r = r[1:]
	}
	return q, r, nil
}

// PolyFromRoots is numpy.poly for real roots: the monic polynomial with the
// given roots, highest degree first. No roots gives [1].
func PolyFromRoots(roots []float64) []float64 {
	a := make([]float64, 1, len(roots)+1)
	a[0] = 1
	for _, z := range roots {
		a = append(a, 0)
		for k := len(a) - 1; k >= 1; k-- {
			a[k] += float64(a[k-1] * -z)
		}
	}
	return a
}

func lessC(a, b complex128) bool {
	if real(a) != real(b) {
		return real(a) < real(b)
	}
	return imag(a) < imag(b)
}

// PolyFromComplexRoots is numpy.poly for complex roots. The coefficients are
// complex; when the roots are closed under conjugation (as numpy checks, by
// sorting) the imaginary parts are dropped (set to 0) and realOnly is true.
func PolyFromComplexRoots(roots []complex128) (coef []complex128, realOnly bool) {
	a := make([]complex128, 1, len(roots)+1)
	a[0] = 1
	for _, z := range roots {
		a = append(a, 0)
		nz := -z
		for k := len(a) - 1; k >= 1; k-- {
			w := a[k-1]
			pr := float64(real(w)*real(nz)) - float64(imag(w)*imag(nz))
			pi := float64(real(w)*imag(nz)) + float64(imag(w)*real(nz))
			a[k] += complex(pr, pi)
		}
	}
	s := append([]complex128(nil), roots...)
	c := make([]complex128, len(s))
	for i, z := range s {
		c[i] = complex(real(z), -imag(z))
	}
	sort.Slice(s, func(i, j int) bool { return lessC(s[i], s[j]) })
	sort.Slice(c, func(i, j int) bool { return lessC(c[i], c[j]) })
	realOnly = true
	for i := range s {
		if s[i] != c[i] {
			realOnly = false
			break
		}
	}
	if realOnly {
		for i := range a {
			a[i] = complex(real(a[i]), 0)
		}
	}
	return a, realOnly
}
