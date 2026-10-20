package poly

import (
	"math"
	"math/big"

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

// exactSSR returns the least-squares residual sum of squares of the
// column-major design a against y, computed through the normal equations
// at 256 bits (enough for kappa^2 up to 1e20), rounded to float64.
func exactSSR(a, y []float64) float64 {
	m := len(y)
	n := len(a) / m
	const prec = 256
	nf := func() *big.Float { return new(big.Float).SetPrec(prec) }
	g := make([][]*big.Float, n)
	b := make([]*big.Float, n)
	for i := range g {
		g[i] = make([]*big.Float, n)
		for j := range g[i] {
			g[i][j] = nf()
		}
		b[i] = nf()
	}
	t, ai, aj, yr := nf(), nf(), nf(), nf()
	for r := 0; r < m; r++ {
		yr.SetFloat64(y[r])
		for i := 0; i < n; i++ {
			ai.SetFloat64(a[i*m+r])
			b[i].Add(b[i], t.Mul(ai, yr))
			for j := i; j < n; j++ {
				g[i][j].Add(g[i][j], t.Mul(ai, aj.SetFloat64(a[j*m+r])))
			}
		}
	}
	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			g[i][j] = nf().Set(g[j][i]) // a copy: elimination updates entries in place
		}
	}
	for k := 0; k < n; k++ {
		for i := k + 1; i < n; i++ {
			f := nf().Quo(g[i][k], g[k][k])
			for j := k; j < n; j++ {
				g[i][j].Sub(g[i][j], t.Mul(f, g[k][j]))
			}
			b[i].Sub(b[i], t.Mul(f, b[k]))
		}
	}
	x := make([]*big.Float, n)
	for i := n - 1; i >= 0; i-- {
		s := nf().Set(b[i])
		for j := i + 1; j < n; j++ {
			s.Sub(s, t.Mul(g[i][j], x[j]))
		}
		x[i] = s.Quo(s, g[i][i])
	}
	ssr, res := nf(), nf()
	for r := 0; r < m; r++ {
		res.SetFloat64(y[r])
		for j := 0; j < n; j++ {
			res.Sub(res, t.Mul(ai.SetFloat64(a[j*m+r]), x[j]))
		}
		ssr.Add(ssr, t.Mul(res, res))
	}
	f, _ := ssr.Float64()
	return f
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
