package poly

import (
	"math"
	"math/cmplx"
	"testing"
)

var basisFiles = []string{"basis_chebyshev.json", "basis_legendre.json", "basis_hermite.json", "basis_hermitee.json", "basis_laguerre.json"}

// forEach runs every case of a fixture file through the library, checks that
// the library fails exactly where NumPy raised, and hands the others to f.
func forEach(t *testing.T, file string, f func(c Case, got map[string][]float64)) {
	t.Helper()
	for _, c := range loadCases(t, file) {
		if testing.Short() && c.huge() {
			continue // 10^6-point fixtures are too slow under qemu
		}
		got, err := runCase(c)
		switch {
		case c.Err != "" && err == nil:
			t.Errorf("%s %q: numpy raised %s, the library returned no error", file, c.Name, c.Err)
		case c.Err == "" && err != nil:
			t.Errorf("%s %q: numpy succeeded, the library failed: %v", file, c.Name, err)
		case c.Err == "":
			f(c, got)
		}
	}
}

// exactFiles asserts that every output named in outs is bit-identical (0 ULP,
// NaN==NaN, -0==0) to numpy's, and returns the number of values compared.
func exactOps(t *testing.T, file string, prefix string, ops ...string) (n, exact int) {
	t.Helper()
	want := map[string]bool{}
	for _, o := range ops {
		want[o] = true
	}
	forEach(t, file, func(c Case, got map[string][]float64) {
		if !want[c.Op] {
			return
		}
		for name, w := range c.Out {
			g := c.sample(name, got[name])
			if len(g) != len(w) {
				t.Errorf("%s%s %q.%s: %d values, numpy %d", prefix, c.Op, c.Name, name, len(g), len(w))
				continue
			}
			for i := range w {
				n++
				if ulpDist(g[i], w[i]) == 0 {
					exact++
				} else {
					t.Errorf("%s%s %q.%s[%d]: got %v numpy %v (%d ULP)", prefix, c.Op, c.Name, name, i, g[i], w[i], ulpDist(g[i], w[i]))
				}
			}
		}
	})
	return n, exact
}

func TestOraclePolyBitIdentical(t *testing.T) {
	n, exact := exactOps(t, "poly1d.json", "", "polyval", "polyder", "polyint", "polyadd", "polysub", "polydiv", "poly", "polyc")
	t.Logf("polyval/polyder/polyint/polyadd/polysub/polydiv/poly: %d of %d values bit-identical to numpy", exact, n)
	if n == 0 {
		t.Fatal("no cases")
	}
}

// TestOracleConv: numpy's convolve sums with a different order (SIMD
// partial sums, possibly FMA), so the comparison is relative to the sum of
// absolute products, the natural scale of a convolution's rounding error.
func TestOracleConv(t *testing.T) {
	var worst float64
	var n, exact int
	forEach(t, "poly1d.json", func(c Case, got map[string][]float64) {
		if c.Op != "conv" && c.Op != "polymul" {
			return
		}
		a, b := c.In("a"), c.In("b")
		if c.Op == "polymul" {
			a, b = trimLeading(a), trimLeading(b)
		}
		w, g := c.Out["y"], got["y"]
		if len(g) != len(w) {
			t.Fatalf("%s: length %d, numpy %d", c.Name, len(g), len(w))
		}
		for k := range w {
			s := 0.0
			for i := range a {
				if j := k - i; j >= 0 && j < len(b) {
					s += math.Abs(a[i] * b[j])
				}
			}
			n++
			if g[k] == w[k] || math.IsNaN(g[k]) && math.IsNaN(w[k]) {
				exact++
				continue
			}
			worst = math.Max(worst, math.Abs(g[k]-w[k])/s)
		}
	})
	t.Logf("conv/polymul: %d of %d bit-identical; max |go-numpy| / sum|a_i b_j| = %.3g eps", exact, n, worst/eps)
	if worst > 4*eps {
		t.Errorf("conv deviates by %.3g eps of the absolute-product sum", worst/eps)
	}
}

// TestOraclePolyvalComp checks the compensated Horner against the exactly
// rounded value (Python fractions) with the bound of Graillat, Langlois and
// Louvet: |res-p(x)|/|p(x)| <= u + gamma_{2n}^2 cond(p,x).
func TestOraclePolyvalComp(t *testing.T) {
	better, total, hornerWorse := 0, 0, 0
	var worstRatio float64
	forEach(t, "poly1d.json", func(c Case, got map[string][]float64) {
		if c.Op != "polyvalcomp" {
			return
		}
		p, x := c.In("p"), c.In("x")
		n := len(p) - 1
		u := eps / 2
		gamma := float64(2*n) * u / (1 - float64(2*n)*u)
		for i, xv := range x {
			exact := c.Out["y"][i]
			if exact == 0 {
				continue
			}
			ax, s := math.Abs(xv), 0.0
			for _, a := range p {
				s = s*ax + math.Abs(a)
			}
			cond := 2 * s / math.Abs(exact)
			bound := u + gamma*gamma*cond
			e := math.Abs(got["y"][i]-exact) / math.Abs(exact)
			r := e / bound
			worstRatio = math.Max(worstRatio, r)
			if r > 1 {
				t.Errorf("%s x=%v: rel err %.3g exceeds bound %.3g (cond %.3g)", c.Name, xv, e, bound, cond)
			}
			total++
			h := math.Abs(Polyval(p, xv)-exact) / math.Abs(exact)
			if e < h {
				better++
			}
			if h > bound {
				hornerWorse++
			}
		}
	})
	t.Logf("compensated Horner: worst error/bound = %.3g; more accurate than Horner at %d of %d points; plain Horner violates the compensated bound at %d points", worstRatio, better, total, hornerWorse)
	if hornerWorse == 0 {
		t.Error("fixture does not exercise ill-conditioned evaluation")
	}
}

func TestOracleBasisBitIdentical(t *testing.T) {
	for _, f := range basisFiles {
		n, exact := exactOps(t, f, f+" ", "val", "vander", "der", "int", "sval", "sder", "sint")
		t.Logf("%s: val/vander/der/int/Series.Val/Der/Int: %d of %d values bit-identical", f, exact, n)
		if n == 0 {
			t.Fatal("no cases")
		}
	}
}

func lastRoots(re, im []float64) []complex128 {
	r := make([]complex128, len(re))
	for i := range r {
		r[i] = complex(re[i], im[i])
	}
	return r
}

// matchErr returns the largest |g-w|/max(1,|w|) over the optimal-ish (greedy
// nearest) pairing of the two root sets, and over the positional pairing.
func matchErr(got, want []complex128) (matched, positional float64) {
	used := make([]bool, len(got))
	for i, w := range want {
		positional = math.Max(positional, cmplx.Abs(got[i]-w)/math.Max(1, cmplx.Abs(w)))
		best, bi := math.Inf(1), -1
		for j, g := range got {
			if d := cmplx.Abs(g - w); !used[j] && d < best {
				best, bi = d, j
			}
		}
		used[bi] = true
		matched = math.Max(matched, best/math.Max(1, cmplx.Abs(w)))
	}
	return matched, positional
}

// backwardErr is max |p(z)| / sum|p_k||z|^k over the roots (p highest first).
func backwardErr(p []float64, roots []complex128) float64 {
	worst := 0.0
	for _, z := range roots {
		var v complex128
		s := 0.0
		for _, a := range p {
			v = v*z + complex(a, 0)
			s = s*cmplx.Abs(z) + math.Abs(a)
		}
		worst = math.Max(worst, cmplx.Abs(v)/s)
	}
	return worst
}

func TestOracleRoots(t *testing.T) {
	var samePos, total int
	var worstWell, worstBackRatio float64
	forEach(t, "roots.json", func(c Case, got map[string][]float64) {
		g, w := lastRoots(got["re"], got["im"]), lastRoots(c.Out["re"], c.Out["im"])
		if len(g) != len(w) {
			t.Fatalf("%s: %d roots, numpy %d", c.Name, len(g), len(w))
		}
		if len(w) == 0 {
			return
		}
		mt, pos := matchErr(g, w)
		total++
		if pos == 0 {
			samePos++
		}
		p := c.In("p")
		// strip for the backward error
		for len(p) > 0 && p[0] == 0 {
			p = p[1:]
		}
		for len(p) > 0 && p[len(p)-1] == 0 {
			p = p[:len(p)-1]
		}
		bg, bw := backwardErr(p, g), backwardErr(p, w)
		if r := bg / math.Max(bw, eps); r > worstBackRatio {
			worstBackRatio = r
		}
		if c.Name == "random" && len(w) <= 50 {
			worstWell = math.Max(worstWell, pos)
			if pos > 1e-13 {
				t.Errorf("%s deg %d: positional error %.3g", c.Name, len(w), pos)
			}
		}
		if mt > 1e-2 || bg > 100*math.Max(bw, 4*eps) {
			t.Errorf("%s: matched error %.3g, backward error %.3g (numpy %.3g)", c.Name, mt, bg, bw)
		}
	})
	t.Logf("roots: %d of %d cases agree with numpy to the last bit in value and ORDER; well-conditioned random deg<=50: max positional error %.3g; worst (our/numpy) backward-error ratio %.3g", samePos, total, worstWell, worstBackRatio)
}

func TestOracleBasisRoots(t *testing.T) {
	for _, f := range basisFiles {
		var worst float64
		var n int
		forEach(t, f, func(c Case, got map[string][]float64) {
			if c.Op != "broots" && c.Op != "sroots" {
				return
			}
			g, w := lastRoots(got["re"], got["im"]), lastRoots(c.Out["re"], c.Out["im"])
			if len(g) != len(w) {
				t.Fatalf("%s %s: %d roots, numpy %d", f, c.Name, len(g), len(w))
			}
			if len(w) == 0 {
				return
			}
			n++
			_, pos := matchErr(g, w)
			worst = math.Max(worst, pos)
			if pos > 1e-11 {
				t.Errorf("%s %s: positional error %.3g", f, c.Name, pos)
			}
		})
		t.Logf("%s roots (companion, sorted): %d cases, max positional error vs numpy %.3g", f, n, worst)
	}
}

// fitChecks measures a fit against numpy in backward-error terms: the
// difference of the scaled coefficient vectors, normalised by the condition
// number bound eps*|y|*(kappa + kappa^2*rho) of the least-squares problem.
func fitCheck(t *testing.T, c Case, got map[string][]float64, name string, worst *float64, worstSSR *float64) {
	t.Helper()
	wc, gc := c.Out["c"], got["c"]
	if rk := c.Out["rank"]; len(rk) == 1 && float64(int(got["rank"][0])) != rk[0] {
		t.Errorf("%s: rank %v, numpy %v", name, got["rank"], rk)
	}
	if len(gc) != len(wc) {
		t.Fatalf("%s: %d coefficients, numpy %d", name, len(gc), len(wc))
	}
	if len(wc) > 0 && math.IsNaN(wc[0]) { // NaN data: numpy returns NaN
		for i := range gc {
			if !math.IsNaN(gc[i]) {
				t.Errorf("%s: coefficient %v, numpy NaN", name, gc[i])
			}
		}
		return
	}
	a, scale, k := designOf(c)
	_ = a
	ny := math.Sqrt(sumSq(weighted(c.In("y"), c.In("w"))))
	d := make([]float64, len(gc))
	for j := range d {
		d[j] = (gc[j] - wc[j]) * scale[j]
	}
	cs := make([]float64, len(wc)) // numpy's scaled coefficients
	for j := range cs {
		cs[j] = wc[j] * scale[j]
	}
	rr := 0.0 // numpy's residual norm
	if r := c.Out["resid"]; len(r) == 1 {
		rr = math.Sqrt(r[0])
		if e := math.Abs(got["resid"][0]-r[0]) / (eps * ny * ny); e > *worstSSR && k <= 1e10 {
			*worstSSR = e
		}
	}
	if len(c.Out["resid"]) != len(got["resid"]) {
		t.Errorf("%s: residual presence differs (numpy %d values, ours %d)", name, len(c.Out["resid"]), len(got["resid"]))
	}
	if math.IsInf(k, 0) || k > 1e10 { // beyond first-order theory: only rank/residual presence are compared
		return
	}
	// First-order bound for least squares: eps*(kappa*|c| + kappa^2*|r|).
	e := math.Sqrt(sumSq(d)) / (eps * (k*math.Sqrt(sumSq(cs)) + k*k*rr))
	if ny == 0 {
		e = 0
	}
	if e > *worst {
		*worst = e
	}
}

func TestOraclePolyfit(t *testing.T) {
	var worst, worstSSR float64
	n := 0
	forEach(t, "polyfit.json", func(c Case, got map[string][]float64) {
		switch c.Op {
		case "polyfit":
			n++
			fitCheck(t, c, got, c.Name, &worst, &worstSSR)
		case "polyfitcov":
			if e := normErr(got["c"], c.Out["c"]); e > 1e-11 {
				t.Errorf("cov %s: coefficient error %.3g", c.Name, e)
			}
			if e := normErr(got["V"], c.Out["V"]); e > 1e-9 {
				t.Errorf("cov %s: covariance error %.3g", c.Name, e)
			}
		}
	})
	t.Logf("polyfit: %d fits; max |dc*scale|/(eps(kappa|c*scale|+kappa^2|r|)) = %.3g; max |dSSR|/(eps|y|^2) = %.3g", n, worst, worstSSR)
	if worst > 40 {
		t.Errorf("coefficient deviation %.3g (units of the conditioning bound)", worst)
	}
	if worstSSR > 1000 {
		t.Errorf("residual deviation %.3g", worstSSR)
	}
}

func TestOracleBasisFit(t *testing.T) {
	for _, f := range basisFiles {
		var worst, worstSSR float64
		n := 0
		forEach(t, f, func(c Case, got map[string][]float64) {
			switch c.Op {
			case "fit":
				n++
				fitCheck(t, c, got, f+" "+c.Name, &worst, &worstSSR)
			case "sfit":
				if e := normErr(got["dom"], c.Out["dom"]); e != 0 {
					t.Errorf("%s %s: domain differs by %.3g", f, c.Name, e)
				}
				if e := normErr(got["c"], c.Out["c"]); e > 1e-6 {
					t.Errorf("%s %s: coefficient error %.3g", f, c.Name, e)
				}
			}
		})
		t.Logf("%s fit: %d fits; max coefficient deviation %.3g conditioning units; max |dSSR|/(eps|y|^2) %.3g", f, n, worst, worstSSR)
		if worst > 40 {
			t.Errorf("%s: coefficient deviation %.3g", f, worst)
		}
		if worstSSR > 1000 {
			t.Errorf("%s: residual deviation %.3g", f, worstSSR)
		}
	}
}

// TestFixtureInventory documents what the fixtures contain: how many cases,
// how many of them make NumPy raise, and the extreme sizes.
func TestFixtureInventory(t *testing.T) {
	for _, f := range append([]string{"poly1d.json", "polyfit.json", "roots.json"}, basisFiles...) {
		cs := loadCases(t, f)
		raised, big := 0, 0
		kinds := map[string]bool{}
		for _, c := range cs {
			kinds[c.Op] = true
			if c.Err != "" {
				raised++
			}
			if c.Gens != nil {
				big++
			}
		}
		t.Logf("%s: %d cases (%d ops), %d where numpy raises, %d regenerated large-n cases", f, len(cs), len(kinds), raised, big)
		if raised == 0 || len(cs) < 30 {
			t.Errorf("%s: fixture too thin", f)
		}
	}
}
