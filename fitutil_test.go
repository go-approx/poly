package poly

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/linalg"
)

// designOf returns the scaled, weighted column-major design matrix of a fit
// case, its column scales, and the condition number of the matrix (Inf when
// rank deficient or underdetermined).
func designOf(c Case) (a, scale []float64, kappa float64) {
	x, deg := c.In("x"), int(c.N("deg"))
	var w []float64
	if c.has("w") {
		w = c.In("w")
	}
	m, n := len(x), deg+1
	if c.Op == "polyfit" || c.Op == "polyfitcov" {
		a, scale = polyDesign(x, w, deg)
	} else {
		v, _ := Vander(Basis(int(c.N("b"))), x, deg)
		a, scale = basisDesign(v, w, m, n)
	}
	if m < n {
		return a, scale, math.Inf(1)
	}
	q := append([]float64(nil), a...)
	householderQR(q, m, n, make([]float64, m))
	r := make([]float64, n*n)
	for j := 0; j < n; j++ {
		for i := 0; i <= j; i++ {
			r[i*n+j] = q[j*m+i]
		}
	}
	am, _ := ndarray.FromData(r, n, n)
	bm, _ := ndarray.FromData(make([]float64, n), n)
	res, err := linalg.Lstsq(am, bm)
	if err != nil {
		return a, scale, math.Inf(1)
	}
	return a, scale, res.S.At(0) / res.S.At(n-1)
}

func maxAbs(v []float64) float64 {
	m := 0.0
	for _, e := range v {
		m = math.Max(m, math.Abs(e))
	}
	return m
}

func sumSq(v []float64) float64 {
	s := 0.0
	for _, e := range v {
		s += e * e
	}
	return s
}
