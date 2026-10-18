package poly

import "math"

// runCase evaluates the library on a fixture case and returns outputs named
// like the fixture's.
func runCase(c Case) (map[string][]float64, error) {
	b := Basis(int(c.N("b")))
	out := map[string][]float64{}
	var w []float64
	if c.has("w") {
		w = c.In("w")
	}
	ik := func(n string) int { return int(c.N(n)) }
	var err error
	setFit := func(f Fit, err error) error {
		if err != nil {
			return err
		}
		out["c"] = f.Coef
		out["rank"] = []float64{float64(f.Rank)}
		out["resid"] = f.Residuals
		if out["resid"] == nil {
			out["resid"] = []float64{}
		}
		return nil
	}
	cmplx := func(r []complex128) {
		re, im := make([]float64, len(r)), make([]float64, len(r))
		for i, z := range r {
			re[i], im[i] = real(z), imag(z)
		}
		out["re"], out["im"] = re, im
	}
	series := func() Series {
		s := Series{Coef: c.In("c"), Domain: b.Domain(), Window: b.Window(), Basis: b}
		if c.has("dom") {
			copy(s.Domain[:], c.In("dom"))
		}
		if c.has("win") {
			copy(s.Window[:], c.In("win"))
		}
		return s
	}
	switch c.Op {
	case "polyval":
		out["y"] = PolyvalAll(nil, c.In("p"), c.In("x"))
	case "polyvalcomp":
		out["y"] = PolyvalCompAll(nil, c.In("p"), c.In("x"))
	case "polyder":
		out["y"], err = Polyder(c.In("p"), ik("m"))
	case "polyint":
		var k []float64
		if c.has("k") {
			k = c.In("k")
		}
		out["y"], err = Polyint(c.In("p"), ik("m"), k)
	case "conv":
		out["y"], err = Conv(c.In("a"), c.In("b"))
	case "polyadd":
		out["y"] = Polyadd(c.In("a"), c.In("b"))
	case "polysub":
		out["y"] = Polysub(c.In("a"), c.In("b"))
	case "polymul":
		out["y"] = Polymul(c.In("a"), c.In("b"))
	case "polydiv":
		out["q"], out["r"], err = Deconv(c.In("a"), c.In("b"))
	case "poly":
		out["y"] = PolyFromRoots(c.In("r"))
	case "polyc":
		re, im := c.In("re"), c.In("im")
		z := make([]complex128, len(re))
		for i := range z {
			z[i] = complex(re[i], im[i])
		}
		a, _ := PolyFromComplexRoots(z)
		cmplx(a)
	case "polyfit":
		err = setFit(Polyfit(c.In("x"), c.In("y"), ik("deg"), &FitOptions{W: w, Rcond: c.N("rcond")}))
	case "polyfitcov":
		f, e := Polyfit(c.In("x"), c.In("y"), ik("deg"), &FitOptions{W: w, Cov: CovMode(ik("cov"))})
		err = e
		if e == nil {
			out["c"], out["V"] = f.Coef, f.Cov
		}
	case "roots":
		var r []complex128
		r, err = Roots(c.In("p"))
		cmplx(r)
	case "val":
		out["y"], err = ValAll(b, nil, c.In("c"), c.In("x"))
	case "vander":
		out["y"], err = Vander(b, c.In("x"), ik("deg"))
	case "fit":
		err = setFit(FitBasis(b, c.In("x"), c.In("y"), ik("deg"), w, c.N("rcond")))
	case "der":
		out["y"], err = Der(b, c.In("c"), ik("m"), c.N("scl"))
	case "int":
		var k []float64
		if c.has("k") {
			k = c.In("k")
		}
		out["y"], err = Int(b, c.In("c"), ik("m"), k, c.N("lbnd"), c.N("scl"))
	case "broots":
		var r []complex128
		r, err = RootsBasis(b, c.In("c"))
		cmplx(r)
	case "sval":
		out["y"], err = series().ValAll(nil, c.In("x"))
	case "sder":
		var s Series
		s, err = series().Der(ik("m"))
		out["y"] = s.Coef
	case "sint":
		var lb *float64
		if c.hasNum("lbnd") {
			v := c.N("lbnd")
			lb = &v
		}
		var k []float64
		if c.has("k") {
			k = c.In("k")
		}
		var s Series
		s, err = series().Int(ik("m"), k, lb)
		out["y"] = s.Coef
	case "sroots":
		var r []complex128
		r, err = series().Roots()
		cmplx(r)
	case "sfit":
		o := &SeriesFitOptions{W: w}
		if c.has("dom") {
			d := [2]float64{}
			copy(d[:], c.In("dom"))
			o.Domain = &d
		}
		if c.has("win") {
			d := [2]float64{}
			copy(d[:], c.In("win"))
			o.Window = &d
		}
		var s Series
		s, _, err = FitSeries(b, c.In("x"), c.In("y"), ik("deg"), o)
		out["c"] = s.Coef
		out["dom"] = s.Domain[:]
	default:
		panic("unknown op " + c.Op)
	}
	return out, err
}

// sample applies the case's output stride to a full-length output.
func (c Case) sample(name string, v []float64) []float64 {
	if c.Stride == 0 || name != "y" {
		return v
	}
	var s []float64
	for i := 0; i < len(v); i += c.Stride {
		s = append(s, v[i])
	}
	return s
}

// normErr is max|g-w| / max|w| (absolute when w is all zero).
func normErr(g, w []float64) float64 {
	var num, den float64
	for i := range w {
		d := math.Abs(g[i] - w[i])
		if math.IsNaN(g[i]) && math.IsNaN(w[i]) {
			continue
		}
		if math.IsNaN(d) {
			return math.Inf(1)
		}
		num = math.Max(num, d)
		den = math.Max(den, math.Abs(w[i]))
	}
	if den == 0 {
		return num
	}
	return num / den
}
