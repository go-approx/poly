//go:build oracle

package poly

import (
	"encoding/json"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/go-approx/poly/internal/oracle"
)

var (
	npMod = []string{"chebyshev", "legendre", "hermite", "hermite_e", "laguerre"}
	npPfx = []string{"cheb", "leg", "herm", "herme", "lag"}
	npCls = []string{"Chebyshev", "Legendre", "Hermite", "HermiteE", "Laguerre"}
)

type extract func(oracle.Value) map[string]oracle.Floats

func re(name string) extract {
	return func(v oracle.Value) map[string]oracle.Floats { return map[string]oracle.Floats{name: v.Re} }
}

func cx(v oracle.Value) map[string]oracle.Floats {
	return map[string]oracle.Floats{"re": v.Re, "im": v.Im}
}

// callOf maps a fixture case to the NumPy call that is its oracle.
func callOf(c Case) (oracle.Call, extract) {
	arr := func(n string) any { return oracle.Arr(c.In(n)) }
	kw := map[string]any{}
	if c.has("w") {
		kw["w"] = arr("w")
	}
	if c.hasNum("rcond") {
		kw["rcond"] = c.N("rcond")
	}
	ik := func(n string) int { return int(c.N(n)) }
	b := int(c.N("b"))
	fn := func(s string) string { return "np.polynomial." + npMod[b] + "." + npPfx[b] + s }
	series := func() oracle.Call {
		skw := map[string]any{}
		if c.has("dom") {
			skw["domain"] = arr("dom")
		}
		if c.has("win") {
			skw["window"] = arr("win")
		}
		return oracle.Call{Fn: "np.polynomial." + npCls[b], Args: []any{arr("c")}, Kw: skw}
	}
	fitOut := func(v oracle.Value) map[string]oracle.Floats {
		t := v.Tuple
		return map[string]oracle.Floats{"c": t[0].Re, "resid": t[1].Tuple[0].Re, "rank": t[1].Tuple[1].Re}
	}
	switch c.Op {
	case "polyval":
		return oracle.Call{Fn: "np.polyval", Args: []any{arr("p"), arr("x")}}, re("y")
	case "polyvalcomp":
		return oracle.Call{Fn: "ex", Args: []any{arr("p"), arr("x")}}, re("y")
	case "polyder":
		return oracle.Call{Fn: "np.polyder", Args: []any{arr("p"), ik("m")}}, re("y")
	case "polyint":
		if c.has("k") {
			kw["k"] = arr("k")
		}
		return oracle.Call{Fn: "np.polyint", Args: []any{arr("p"), ik("m")}, Kw: kw}, re("y")
	case "conv", "polyadd", "polysub", "polymul":
		fn := map[string]string{"conv": "np.convolve", "polyadd": "np.polyadd", "polysub": "np.polysub", "polymul": "np.polymul"}[c.Op]
		return oracle.Call{Fn: fn, Args: []any{arr("a"), arr("b")}}, re("y")
	case "polydiv":
		return oracle.Call{Fn: "np.polydiv", Args: []any{arr("a"), arr("b")}}, func(v oracle.Value) map[string]oracle.Floats {
			return map[string]oracle.Floats{"q": v.Tuple[0].Re, "r": v.Tuple[1].Re}
		}
	case "poly":
		return oracle.Call{Fn: "np.poly", Args: []any{arr("r")}}, re("y")
	case "polyc":
		z := make([]complex128, len(c.Arr["re"]))
		for i := range z {
			z[i] = complex(c.Arr["re"][i], c.Arr["im"][i])
		}
		return oracle.Call{Fn: "np.poly", Args: []any{oracle.Cplx(z)}}, func(v oracle.Value) map[string]oracle.Floats {
			im := v.Im
			if !v.Complex {
				im = make([]float64, len(v.Re))
			}
			return map[string]oracle.Floats{"re": v.Re, "im": im}
		}
	case "polyfit":
		kw["full"] = true
		return oracle.Call{Fn: "np.polyfit", Args: []any{arr("x"), arr("y"), ik("deg")}, Kw: kw}, func(v oracle.Value) map[string]oracle.Floats {
			t := v.Tuple
			return map[string]oracle.Floats{"c": t[0].Re, "resid": t[1].Re, "rank": t[2].Re}
		}
	case "polyfitcov":
		kw["cov"] = true
		if ik("cov") == 2 {
			kw["cov"] = "unscaled"
		}
		return oracle.Call{Fn: "np.polyfit", Args: []any{arr("x"), arr("y"), ik("deg")}, Kw: kw}, func(v oracle.Value) map[string]oracle.Floats {
			return map[string]oracle.Floats{"c": v.Tuple[0].Re, "V": v.Tuple[1].Re}
		}
	case "roots":
		return oracle.Call{Fn: "np.roots", Args: []any{arr("p")}}, func(v oracle.Value) map[string]oracle.Floats {
			im := v.Im
			if !v.Complex {
				im = make([]float64, len(v.Re))
			}
			return map[string]oracle.Floats{"re": v.Re, "im": im}
		}
	case "val":
		return oracle.Call{Fn: fn("val"), Args: []any{arr("x"), arr("c")}}, re("y")
	case "vander":
		return oracle.Call{Fn: fn("vander"), Args: []any{arr("x"), ik("deg")}}, re("y")
	case "fit":
		kw["full"] = true
		return oracle.Call{Fn: fn("fit"), Args: []any{arr("x"), arr("y"), ik("deg")}, Kw: kw}, fitOut
	case "der":
		return oracle.Call{Fn: fn("der"), Args: []any{arr("c"), ik("m"), c.N("scl")}}, re("y")
	case "int":
		a := []any{arr("c"), ik("m")}
		if c.has("k") {
			kw["k"] = arr("k")
		}
		kw["lbnd"] = c.N("lbnd")
		kw["scl"] = c.N("scl")
		return oracle.Call{Fn: fn("int"), Args: a, Kw: kw}, re("y")
	case "broots":
		return oracle.Call{Fn: fn("roots"), Args: []any{arr("c")}}, func(v oracle.Value) map[string]oracle.Floats {
			im := v.Im
			if !v.Complex {
				im = make([]float64, len(v.Re))
			}
			return map[string]oracle.Floats{"re": v.Re, "im": im}
		}
	case "sval":
		return oracle.Call{Fn: ".__call__", Args: []any{series(), arr("x")}}, re("y")
	case "sder":
		return oracle.Call{Fn: ".deriv", Args: []any{series(), ik("m")}}, func(v oracle.Value) map[string]oracle.Floats {
			return map[string]oracle.Floats{"y": v.Tuple[0].Re}
		}
	case "sint":
		ikw := map[string]any{}
		if c.has("k") {
			ikw["k"] = arr("k")
		}
		if c.hasNum("lbnd") {
			ikw["lbnd"] = c.N("lbnd")
		}
		return oracle.Call{Fn: ".integ", Args: []any{series(), ik("m")}, Kw: ikw}, func(v oracle.Value) map[string]oracle.Floats {
			return map[string]oracle.Floats{"y": v.Tuple[0].Re}
		}
	case "sroots":
		return oracle.Call{Fn: ".roots", Args: []any{series()}}, func(v oracle.Value) map[string]oracle.Floats {
			im := v.Im
			if !v.Complex {
				im = make([]float64, len(v.Re))
			}
			return map[string]oracle.Floats{"re": v.Re, "im": im}
		}
	case "sfit":
		if c.has("dom") {
			kw["domain"] = arr("dom")
		}
		if c.has("win") {
			kw["window"] = arr("win")
		}
		return oracle.Call{Fn: "np.polynomial." + npCls[b] + ".fit", Args: []any{arr("x"), arr("y"), ik("deg")}, Kw: kw}, func(v oracle.Value) map[string]oracle.Floats {
			return map[string]oracle.Floats{"c": v.Tuple[0].Re, "dom": v.Tuple[1].Re}
		}
	}
	panic("unknown op " + c.Op)
}

var specials = []float64{0, math.Copysign(0, -1), 1, -1, 2, -2, 1e-9, 1e3, 1e154, math.NaN(), math.Inf(1), math.Inf(-1)}

func ac(name, op string, arr map[string]oracle.Floats, num map[string]float64) Case {
	c := Case{Name: name, Op: op, Arr: arr, Num: map[string]oracle.Float{}}
	for k, v := range num {
		c.Num[k] = oracle.Float(v)
	}
	return c
}

type m = map[string]oracle.Floats
type nn = map[string]float64

func poly1dCases() []Case {
	r := newRNG(1)
	var cs []Case
	for _, deg := range []int{0, 1, 2, 3, 5, 8, 12, 20, 30, 50} {
		x := append(unif(r, 100, -2, 2), specials...)
		cs = append(cs, ac("polyval", "polyval", m{"p": unif(r, deg+1, -3, 3), "x": x}, nil))
	}
	cs = append(cs,
		ac("polyval empty p", "polyval", m{"p": {}, "x": {1, 2}}, nil),
		ac("polyval NaN coef", "polyval", m{"p": {1, math.NaN(), 2}, "x": {1, 2}}, nil),
		ac("polyval inf coef", "polyval", m{"p": {math.Inf(1), 1, 2}, "x": {0, 1, -1}}, nil),
		ac("polyval const inf x", "polyval", m{"p": {3}, "x": {math.Inf(1), 0, 1}}, nil),
	)
	// (x-1)^12 near its root: Horner is wildly inaccurate but bit-identical to numpy.
	bin := func(n int) []float64 {
		c := make([]float64, n+1)
		c[0] = 1
		for k := 1; k <= n; k++ {
			c[k] = -c[k-1] * float64(n-k+1) / float64(k)
		}
		return c
	}
	cs = append(cs, ac("polyval (x-1)^12 near root", "polyval", m{"p": bin(12), "x": unif(r, 100, 0.99, 1.01)}, nil))
	big := ac("polyval n=1e5 deg5", "polyval", m{"p": unif(r, 6, -1, 1)}, nil)
	big.Gens, big.Stride = map[string]Gen{"x": {Seed: 11, N: 100000, Lo: -1.5, Hi: 1.5}}, 997
	big2 := ac("polyval n=1e6 deg3", "polyval", m{"p": unif(r, 4, -1, 1)}, nil)
	big2.Gens, big2.Stride = map[string]Gen{"x": {Seed: 12, N: 1000000, Lo: -1.5, Hi: 1.5}}, 9973
	cs = append(cs, big, big2)

	// compensated Horner against exact rational arithmetic.
	for _, n := range []int{3, 5, 8, 10, 12, 15, 20, 25, 30, 40} {
		cs = append(cs, ac("polyvalcomp (x-1)^n", "polyvalcomp", m{"p": bin(n), "x": unif(r, 60, 0.9, 1.1)}, nil))
	}
	for _, deg := range []int{2, 7, 20} {
		cs = append(cs, ac("polyvalcomp random", "polyvalcomp", m{"p": unif(r, deg+1, -3, 3), "x": unif(r, 60, -2, 2)}, nil))
	}
	// derivatives / integrals
	for _, n := range []int{0, 1, 2, 3, 6, 11} {
		for _, mm := range []int{0, 1, 2, 3, 7, 20} {
			p := unif(r, n, -5, 5)
			cs = append(cs, ac("polyder", "polyder", m{"p": p}, nn{"m": float64(mm)}))
			cs = append(cs, ac("polyint", "polyint", m{"p": p}, nn{"m": float64(mm)}))
			if mm >= 1 {
				cs = append(cs, ac("polyint k", "polyint", m{"p": p, "k": unif(r, mm, -2, 2)}, nn{"m": float64(mm)}))
				cs = append(cs, ac("polyint k1", "polyint", m{"p": p, "k": {1.5}}, nn{"m": float64(mm)}))
			}
		}
	}
	cs = append(cs,
		ac("polyder neg", "polyder", m{"p": {1, 2}}, nn{"m": -1}),
		ac("polyint neg", "polyint", m{"p": {1, 2}}, nn{"m": -1}),
		ac("polyint short k", "polyint", m{"p": {1, 2}, "k": {1, 2}}, nn{"m": 3}),
		ac("polyder NaN", "polyder", m{"p": {1, math.NaN(), 2}}, nn{"m": 1}),
	)
	// arithmetic
	for _, sz := range [][2]int{{1, 1}, {1, 5}, {5, 1}, {3, 3}, {4, 9}, {12, 7}, {30, 31}, {200, 150}} {
		a, b := unif(r, sz[0], -3, 3), unif(r, sz[1], -3, 3)
		for _, op := range []string{"conv", "polyadd", "polysub", "polymul"} {
			cs = append(cs, ac(op, op, m{"a": a, "b": b}, nil))
		}
		cs = append(cs, ac("polydiv", "polydiv", m{"a": a, "b": b}, nil))
	}
	cs = append(cs,
		ac("conv empty", "conv", m{"a": {}, "b": {1}}, nil),
		ac("polymul leading zeros", "polymul", m{"a": {0, 0, 1, 2}, "b": {0, 3}}, nil),
		ac("polymul all zero", "polymul", m{"a": {0, 0}, "b": {1, 2}}, nil),
		ac("polyadd empty", "polyadd", m{"a": {}, "b": {1, 2}}, nil),
		ac("polysub empty", "polysub", m{"a": {1, 2, 3}, "b": {}}, nil),
		ac("polydiv exact", "polydiv", m{"a": {1, 0, -1}, "b": {1, -1}}, nil),
		ac("polydiv shorter", "polydiv", m{"a": {1, 2}, "b": {1, 2, 3}}, nil),
		ac("polydiv zero lead", "polydiv", m{"a": {1, 2, 3}, "b": {0, 1}}, nil),
		ac("polydiv empty", "polydiv", m{"a": {}, "b": {1}}, nil),
		ac("polydiv strip remainder", "polydiv", m{"a": {1, 1, 1e-9}, "b": {1, 1}}, nil),
		ac("conv NaN", "conv", m{"a": {1, math.NaN()}, "b": {1, 2}}, nil),
	)
	// poly from roots
	for _, n := range []int{0, 1, 2, 3, 5, 10, 20} {
		cs = append(cs, ac("poly real", "poly", m{"r": unif(r, n, -3, 3)}, nil))
	}
	wil := make([]float64, 20)
	for i := range wil {
		wil[i] = float64(i + 1)
	}
	cs = append(cs, ac("poly wilkinson", "poly", m{"r": wil}, nil))
	cs = append(cs, ac("poly repeated", "poly", m{"r": {1, 1, 1, 1, 2, 2}}, nil))
	// conjugate-closed complex roots (real result) and not closed (complex result)
	cs = append(cs,
		ac("poly conj", "polyc", m{"re": {1, 1, -2, 0.5, 0.5}, "im": {2, -2, 0, 1, -1}}, nil),
		ac("poly conj order", "polyc", m{"re": {0.5, 1, 0.5, -2, 1}, "im": {1, 2, -1, 0, -2}}, nil),
		ac("poly complex", "polyc", m{"re": {1, 2, 3}, "im": {1, 0, -1}}, nil),
		ac("poly complex empty", "polyc", m{"re": {}, "im": {}}, nil),
	)
	return cs
}

func fitCase(name string, x, y []float64, deg int, w []float64, rcond float64, class string) Case {
	c := ac(name, "polyfit", m{"x": x, "y": y}, nn{"deg": float64(deg)})
	c.Class = class
	if w != nil {
		c.Arr["w"] = w
	}
	if rcond != 0 {
		c.Num["rcond"] = oracle.Float(rcond)
	}
	return c
}

func smooth(r *rand.Rand, x []float64, noise float64) []float64 {
	y := make([]float64, len(x))
	for i, v := range x {
		y[i] = math.Sin(2*v) + 0.3*v*v + noise*(r.Float64()-0.5)
	}
	return y
}

func shuffle(r *rand.Rand, x ...[]float64) {
	n := len(x[0])
	r.Shuffle(n, func(i, j int) {
		for _, s := range x {
			s[i], s[j] = s[j], s[i]
		}
	})
}

func fitCases(basis int, op string) []Case {
	r := newRNG(uint64(100 + basis))
	var cs []Case
	mk := func(name string, x, y []float64, deg int, w []float64, rcond float64, class string) {
		c := fitCase(name, x, y, deg, w, rcond, class)
		c.Op = op
		c.Num["b"] = oracle.Float(basis)
		cs = append(cs, c)
	}
	for _, n := range []int{5, 10, 30, 100, 400} {
		x := unif(r, n, -1, 1)
		y := smooth(r, x, 0.1)
		for _, deg := range []int{0, 1, 2, 3, 6, 10} {
			if deg < n {
				mk("random", x, y, deg, nil, 0, "")
			}
		}
		w := unif(r, n, 0.1, 2)
		mk("weighted", x, y, 3, w, 0, "")
		w0 := append([]float64(nil), w...)
		w0[0], w0[n-1] = 0, 0
		mk("zero weights", x, y, 3, w0, 0, "")
		mk("rcond", x, y, 3, nil, 1e-3, "")
		xs, ys := append([]float64(nil), x...), append([]float64(nil), y...)
		shuffle(r, xs, ys)
		mk("unsorted", xs, ys, 4, nil, 0, "")
	}
	// tiny and degenerate problems
	mk("1 point deg0", []float64{0.7}, []float64{2}, 0, nil, 0, "")
	mk("1 point deg1", []float64{0.7}, []float64{2}, 1, nil, 0, "")
	mk("1 point deg2", []float64{0.7}, []float64{2}, 2, nil, 0, "")
	for _, deg := range []int{0, 1, 2, 3} {
		mk("2 points", []float64{-0.4, 0.9}, []float64{1, -3}, deg, nil, 0, "")
	}
	mk("3 points deg5", []float64{-0.4, 0.2, 0.9}, []float64{1, -3, 2}, 5, nil, 0, "")
	mk("duplicate x", []float64{0.1, 0.1, 0.5, 0.5, 0.9, 0.9, -0.3, -0.3}, []float64{1, 1.1, 2, 2.2, 3, 2.9, 0, 0.1}, 3, nil, 0, "")
	mk("all x equal", []float64{0.5, 0.5, 0.5, 0.5, 0.5}, []float64{1, 2, 3, 4, 5}, 2, nil, 0, "")
	mk("x zero deg1", []float64{0, 0}, []float64{1, 3}, 1, nil, 0, "")
	mk("x zero deg0", []float64{0, 0}, []float64{1, 3}, 0, nil, 0, "")
	mk("NaN x", []float64{0, math.NaN(), 1, 2}, []float64{1, 2, 3, 4}, 1, nil, 0, "")
	mk("NaN y", []float64{0, 1, 2, 3}, []float64{1, math.NaN(), 3, 4}, 1, nil, 0, "")
	mk("Inf x", []float64{0, math.Inf(1), 1, 2}, []float64{1, 2, 3, 4}, 1, nil, 0, "")
	mk("Inf y", []float64{0, 1, 2, 3}, []float64{1, math.Inf(-1), 3, 4}, 1, nil, 0, "")
	mk("NaN w", []float64{0, 1, 2, 3}, []float64{1, 2, 3, 4}, 1, []float64{1, math.NaN(), 1, 1}, 0, "")
	mk("all zero weights", []float64{0, 1, 2, 3}, []float64{1, 2, 3, 5}, 1, []float64{0, 0, 0, 0}, 0, "")
	mk("negative deg", []float64{0, 1, 2}, []float64{1, 2, 3}, -1, nil, 0, "")
	mk("empty x", nil, nil, 1, nil, 0, "")
	mk("length mismatch", []float64{0, 1, 2}, []float64{1, 2}, 1, nil, 0, "")
	// huge ranges
	lx := make([]float64, 60)
	for i := range lx {
		lx[i] = math.Pow(10, -8+16*r.Float64())
	}
	mk("x 1e-8..1e8", lx, smooth(r, unif(r, 60, 0, 1), 0), 2, nil, 0, "ill")
	mk("x 1e-8 scale", unif(r, 50, 0, 2e-8), unif(r, 50, -1, 1), 3, nil, 0, "")
	mk("x 1e8 scale", unif(r, 50, -3e8, 3e8), unif(r, 50, -1, 1), 3, nil, 0, "")
	mk("x offset 1e8", unif(r, 50, 1e8, 1e8+10), smooth(r, unif(r, 50, 0, 1), 0), 2, nil, 0, "ill")
	// ill-conditioned: high degree on equispaced nodes
	for _, nd := range [][2]int{{30, 20}, {40, 25}, {60, 30}, {25, 24}, {40, 12}} {
		x := make([]float64, nd[0])
		for i := range x {
			x[i] = float64(i) / float64(nd[0]-1)
		}
		mk("equispaced", x, smooth(r, x, 0), nd[1], nil, 0, "ill")
		for i := range x {
			x[i] = 2*x[i] - 1
		}
		mk("equispaced [-1,1]", x, smooth(r, x, 0), nd[1], nil, 0, "ill")
	}
	// large n: inputs regenerated from a fixed-seed PCG stream
	for _, cfg := range []struct{ n, deg int }{{100000, 3}, {100000, 10}, {1000000, 3}, {1000000, 10}} {
		c := fitCase("large", nil, nil, cfg.deg, nil, 0, "")
		c.Op, c.Arr = op, nil
		c.Num["b"] = oracle.Float(basis)
		c.Gens = map[string]Gen{"x": {Seed: 21, N: cfg.n, Lo: -1, Hi: 1}, "y": {Seed: 22, N: cfg.n, Lo: -1, Hi: 1}}
		cs = append(cs, c)
	}
	return cs
}

func covCases() []Case {
	r := newRNG(7)
	var cs []Case
	for _, n := range []int{8, 50, 400} {
		x := unif(r, n, -1, 1)
		y := smooth(r, x, 0.2)
		for _, deg := range []int{1, 2, 4} {
			for mode := 1; mode <= 2; mode++ {
				c := ac("cov", "polyfitcov", m{"x": x, "y": y}, nn{"deg": float64(deg), "cov": float64(mode)})
				cs = append(cs, c)
				c2 := ac("cov weighted", "polyfitcov", m{"x": x, "y": y, "w": unif(r, n, 0.5, 2)}, nn{"deg": float64(deg), "cov": float64(mode)})
				cs = append(cs, c2)
			}
		}
	}
	x := []float64{0, 1, 2}
	cs = append(cs,
		ac("cov scaled n==order", "polyfitcov", m{"x": x, "y": {1, 2, 4}}, nn{"deg": 2, "cov": 1}),
		ac("cov scaled n<order", "polyfitcov", m{"x": x, "y": {1, 2, 4}}, nn{"deg": 3, "cov": 1}),
		ac("cov unscaled n==order", "polyfitcov", m{"x": x, "y": {1, 2, 4}}, nn{"deg": 2, "cov": 2}),
	)
	return cs
}

func rootsCases() []Case {
	r := newRNG(3)
	var cs []Case
	for _, deg := range []int{1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 50, 100} {
		cs = append(cs, ac("random", "roots", m{"p": unif(r, deg+1, -2, 2)}, nil))
	}
	cs = append(cs,
		ac("empty", "roots", m{"p": {}}, nil),
		ac("all zero", "roots", m{"p": {0, 0, 0}}, nil),
		ac("constant", "roots", m{"p": {5}}, nil),
		ac("constant zero", "roots", m{"p": {0}}, nil),
		ac("leading zeros", "roots", m{"p": {0, 0, 1, -3, 2}}, nil),
		ac("trailing zeros", "roots", m{"p": {1, -3, 2, 0, 0}}, nil),
		ac("both", "roots", m{"p": {0, 2, -3, 1, 0}}, nil),
		ac("only trailing", "roots", m{"p": {4, 0, 0, 0}}, nil),
		ac("linear", "roots", m{"p": {2, -6}}, nil),
		ac("NaN", "roots", m{"p": {1, math.NaN(), 2}}, nil),
		ac("Inf", "roots", m{"p": {1, math.Inf(1), 2}}, nil),
		ac("NaN leading", "roots", m{"p": {math.NaN(), 1, 2}}, nil),
		ac("overflow ratio", "roots", m{"p": {1e-300, 1e300, 1}}, nil),
		ac("conjugate pairs", "roots", m{"p": {1, 0, 2, 0, 1}}, nil),
		ac("x^n-1", "roots", m{"p": {1, 0, 0, 0, 0, 0, 0, 0, -1}}, nil),
		ac("repeated (x-1)^5", "roots", m{"p": {1, -5, 10, -10, 5, -1}}, nil),
		ac("repeated (x-1)^8", "roots", m{"p": {1, -8, 28, -56, 70, -56, 28, -8, 1}}, nil),
		ac("clustered", "roots", m{"p": PolyFromRoots([]float64{1, 1.0001, 1.0002, 2, 2.0001, 3})}, nil),
	)
	wil := make([]float64, 20)
	for i := range wil {
		wil[i] = float64(i + 1)
	}
	cs = append(cs, ac("wilkinson 20", "roots", m{"p": PolyFromRoots(wil)}, nil))
	// conjugate pairs built from known roots
	cs = append(cs, ac("conj-built", "roots", m{"p": PolyFromRoots([]float64{-1, 2, 0.5, 4})}, nil))
	return cs
}

func basisCases(b int) []Case {
	r := newRNG(uint64(500 + b))
	var cs []Case
	num := func(k ...float64) nn {
		n := nn{"b": float64(b)}
		for i := 0; i+1 < len(k); i += 2 {
			n[map[float64]string{0: "deg", 1: "m", 2: "scl", 3: "lbnd"}[k[i]]] = k[i+1]
		}
		return n
	}
	for _, deg := range []int{0, 1, 2, 3, 4, 5, 8, 13, 21, 40} {
		x := append(unif(r, 100, -3, 3), specials...)
		cs = append(cs, ac("val", "val", m{"c": unif(r, deg+1, -2, 2), "x": x}, num()))
	}
	cs = append(cs, ac("val empty c", "val", m{"c": {}, "x": {1}}, num()),
		ac("val NaN c", "val", m{"c": {1, math.NaN(), 3, 4}, "x": {0.5, 1}}, num()))
	big := ac("val n=1e5 deg20", "val", m{"c": unif(r, 21, -1, 1)}, num())
	big.Gens, big.Stride = map[string]Gen{"x": {Seed: 31, N: 100000, Lo: -1, Hi: 1}}, 997
	cs = append(cs, big)
	for _, deg := range []int{0, 1, 2, 5, 10, 30} {
		x := append(unif(r, 50, -3, 3), specials...)
		cs = append(cs, ac("vander", "vander", m{"x": x}, num(0, float64(deg))))
	}
	bv := ac("vander n=1e5 deg10", "vander", nil, num(0, 10))
	bv.Gens, bv.Stride = map[string]Gen{"x": {Seed: 32, N: 100000, Lo: -1, Hi: 1}}, 1009
	cs = append(cs, bv)
	cs = append(cs, ac("vander neg", "vander", m{"x": {1}}, num(0, -1)))
	for _, n := range []int{1, 2, 3, 4, 7, 13, 30} {
		for _, mm := range []int{0, 1, 2, 3, 5, 40} {
			for _, scl := range []float64{1, 0.5, -2, 3.7} {
				if (mm+int(scl*7))%3 == 0 && n > 3 { // thin the grid
					continue
				}
				cs = append(cs, ac("der", "der", m{"c": unif(r, n, -3, 3)}, num(1, float64(mm), 2, scl)))
			}
		}
	}
	cs = append(cs, ac("der NaN", "der", m{"c": {1, math.NaN(), 2, 3}}, num(1, 1, 2, 1)),
		ac("der Inf over", "der", m{"c": {1, 2}}, num(1, 5, 2, 1)),
		ac("der Inf cnt>=n", "der", m{"c": {math.Inf(1), 2}}, num(1, 5, 2, 1)),
		ac("der neg", "der", m{"c": {1, 2}}, num(1, -1, 2, 1)),
		ac("der empty", "der", m{"c": {}}, num(1, 1, 2, 1)))
	for _, n := range []int{1, 2, 3, 5, 10} {
		for _, mm := range []int{0, 1, 2, 3} {
			for ki, k := range [][]float64{nil, {1.5}, {0.3, -2}, {1, 2, 3}, {7, 8, 9, 10}} {
				if ki > mm+1 {
					continue
				}
				for _, lb := range []float64{0, -1, 0.5} {
					for _, scl := range []float64{1, 0.5, 2.5} {
						if (ki+int(lb*2)+int(scl*2)+n)%3 != 0 {
							continue
						}
						c := ac("int", "int", m{"c": unif(r, n, -3, 3)}, num(1, float64(mm), 2, scl, 3, lb))
						if k != nil {
							c.Arr["k"] = k
						}
						cs = append(cs, c)
					}
				}
			}
		}
	}
	cs = append(cs,
		ac("int zero const", "int", m{"c": {0}, "k": {2.5}}, num(1, 1, 2, 1, 3, 0)),
		ac("int zero const m3", "int", m{"c": {0}, "k": {2.5, 1}}, num(1, 3, 2, 2, 3, 1)),
		ac("int NaN", "int", m{"c": {1, math.NaN(), 3}}, num(1, 1, 2, 1, 3, 0)),
		ac("int NaN const", "int", m{"c": {math.NaN()}}, num(1, 1, 2, 1, 3, 0)),
		ac("int too many k", "int", m{"c": {1, 2}, "k": {1, 2, 3}}, num(1, 1, 2, 1, 3, 0)),
		ac("int neg", "int", m{"c": {1, 2}}, num(1, -1, 2, 1, 3, 0)),
	)
	for _, deg := range []int{0, 1, 2, 3, 4, 5, 8, 13, 20, 40} {
		cs = append(cs, ac("roots", "broots", m{"c": unif(r, deg+1, -2, 2)}, num()))
	}
	cs = append(cs,
		ac("roots empty", "broots", m{"c": {}}, num()),
		ac("roots zero", "broots", m{"c": {0, 0}}, num()),
		ac("roots trailing zeros", "broots", m{"c": {1, -2, 0.5, 0, 0}}, num()),
		ac("roots leading zeros", "broots", m{"c": {0, 0, 1, 2, 0.5}}, num()),
		ac("roots NaN", "broots", m{"c": {1, math.NaN(), 2}}, num()),
		ac("roots linear", "broots", m{"c": {3, 2}}, num()),
	)
	cs = append(cs, fitCases(b, "fit")...)
	// Series: domain/window mapping
	doms := [][]float64{{0, 10}, {-3, 5}, {1e-8, 1e8}, {2, 3}}
	wins := [][]float64{nil, {0, 1}, {-2, 2}}
	for di, d := range doms {
		for wi, w := range wins {
			for _, n := range []int{1, 2, 5, 12} {
				cf := unif(r, n, -2, 2)
				mk := func(op string, extra map[string]oracle.Floats, nums nn) {
					a := m{"c": cf, "dom": d}
					if w != nil {
						a["win"] = w
					}
					for k, v := range extra {
						a[k] = v
					}
					nums["b"] = float64(b)
					cs = append(cs, ac("series "+op, op, a, nums))
				}
				mk("sval", m{"x": append(unif(r, 30, d[0], d[1]), d[0], d[1], math.NaN())}, nn{})
				mk("sder", nil, nn{"m": float64((di + wi) % 3)})
				mk("sint", nil, nn{"m": float64(1 + (di+wi)%3)})
				mk("sint", m{"k": {0.5, -1}}, nn{"m": 2, "lbnd": d[0]})
				mk("sint", m{"k": {2}}, nn{"m": 1, "lbnd": 0})
				if n > 1 {
					mk("sroots", nil, nn{})
				}
			}
		}
	}
	// Series.fit: default domain from x, explicit domain/window, equal x
	for _, n := range []int{20, 200} {
		x := unif(r, n, 3, 17)
		y := smooth(r, x, 0.2)
		for _, deg := range []int{1, 3, 6} {
			c := ac("sfit", "sfit", m{"x": x, "y": y}, nn{"b": float64(b), "deg": float64(deg)})
			cs = append(cs, c)
			c2 := ac("sfit dom/win", "sfit", m{"x": x, "y": y, "dom": {0, 20}, "win": {-1, 1}}, nn{"b": float64(b), "deg": float64(deg)})
			cs = append(cs, c2)
			c3 := ac("sfit weights", "sfit", m{"x": x, "y": y, "w": unif(r, n, 0.2, 1)}, nn{"b": float64(b), "deg": float64(deg)})
			cs = append(cs, c3)
		}
	}
	cs = append(cs, ac("sfit equal x", "sfit", m{"x": {2, 2, 2}, "y": {1, 2, 3}}, nn{"b": float64(b), "deg": 1}),
		ac("sfit empty", "sfit", m{"x": {}, "y": {}}, nn{"b": float64(b), "deg": 1}))
	return cs
}

func (c *Case) fill(res oracle.Result, ex extract) {
	if res.Err != "" {
		c.Err, _, _ = strings.Cut(res.Err, ":")
		return
	}
	c.Out = ex(res.Val)
	if c.Stride > 0 {
		for k, v := range c.Out {
			if k != "y" {
				continue
			}
			var s oracle.Floats
			for i := 0; i < len(v); i += c.Stride {
				s = append(s, v[i])
			}
			c.Out[k] = s
		}
	}
}

func writeFixture(t *testing.T, file string, cs []Case) {
	b, err := json.Marshal(cs)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join("testdata", file), append(b, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("%s: %d cases, %d bytes", file, len(cs), len(b))
}

// TestGenerate regenerates testdata/*.json from the NumPy oracle:
//
//	GOWORK=off go test -tags oracle -run TestGenerate .
func TestGenerate(t *testing.T) {
	u, err := oracle.StartUV()
	if err != nil {
		t.Fatal(err)
	}
	defer u.Close()
	cl := oracle.Client{T: u}
	if err := cl.HealthCheck(); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll("testdata", 0o755); err != nil {
		t.Fatal(err)
	}
	files := map[string][]Case{
		"poly1d.json":  poly1dCases(),
		"polyfit.json": append(fitCases(0, "polyfit"), covCases()...),
		"roots.json":   rootsCases(),
	}
	for b := range npMod {
		files["basis_"+strings.ToLower(npCls[b])+".json"] = basisCases(b)
	}
	for file, cs := range files {
		cs := cs
		for lo := 0; lo < len(cs); lo += 40 {
			hi := min(lo+40, len(cs))
			items := make([]any, 0, hi-lo)
			exs := make([]extract, 0, hi-lo)
			for _, c := range cs[lo:hi] {
				call, ex := callOf(c)
				items = append(items, call)
				exs = append(exs, ex)
			}
			res, err := cl.Batch(items...)
			if err != nil {
				t.Fatalf("%s: %v", file, err)
			}
			for i := range res {
				cs[lo+i].fill(res[i], exs[i])
			}
		}
		writeFixture(t, file, cs)
	}
}
