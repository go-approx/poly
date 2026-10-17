package poly

import (
	"sort"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/linalg"
)

// eigvals returns the eigenvalues of the row-major n*n matrix a, in the order
// of LAPACK's geev (as numpy.linalg.eigvals). NaN/Inf is an error.
func eigvals(a []float64, n int) ([]complex128, error) {
	am, err := ndarray.FromData(a, n, n)
	if err != nil {
		return nil, err
	}
	w, err := linalg.Eigvals(am)
	if err != nil {
		return nil, err
	}
	out := make([]complex128, n)
	for i := range out {
		out[i] = w.AtComplex(i)
	}
	return out, nil
}

// Roots is numpy.roots: the roots of the polynomial p (coefficients HIGHEST
// degree first). Leading zeros are stripped, each trailing zero adds a zero
// root (appended last), and the others are the eigenvalues of the companion
// matrix (Edelman & Murakami 1995 show the companion eigenvalue method is
// backward stable for balanced companions), in LAPACK order. NaN or Inf in p
// is an error (numpy raises LinAlgError through eig). The result is always
// complex128; NumPy returns a real array when every imaginary part is zero.
func Roots(p []float64) ([]complex128, error) {
	first, last := -1, -1
	for i, c := range p {
		if c != 0 {
			if first < 0 {
				first = i
			}
			last = i
		}
	}
	if first < 0 {
		return []complex128{}, nil
	}
	trailing := len(p) - last - 1
	q := p[first : last+1]
	var r []complex128
	if n := len(q) - 1; n > 0 {
		a := make([]float64, n*n)
		for j := 0; j < n; j++ {
			a[j] = -q[j+1] / q[0]
		}
		for i := 1; i < n; i++ {
			a[i*n+i-1] = 1
		}
		var err error
		if r, err = eigvals(a, n); err != nil {
			return nil, err
		}
	}
	return append(r, make([]complex128, trailing)...), nil
}

// sortRoots orders complex values lexicographically (real, then imaginary),
// as numpy.sort does for complex arrays.
func sortRoots(r []complex128) {
	sort.Slice(r, func(i, j int) bool { return lessC(r[i], r[j]) })
}
