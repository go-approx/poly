package poly

import (
	"errors"
	"math"
	"testing"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/linalg"
)

func TestBasisNames(t *testing.T) {
	want := []string{"Chebyshev", "Legendre", "Hermite", "HermiteE", "Laguerre", "Basis(?)"}
	for i, w := range want {
		if got := Basis(i).String(); got != w {
			t.Errorf("Basis(%d) = %q, want %q", i, got, w)
		}
	}
	if d := Laguerre.Domain(); d != [2]float64{0, 1} || Laguerre.Window() != d {
		t.Errorf("laguerre domain %v", d)
	}
	if d := Hermite.Domain(); d != [2]float64{-1, 1} {
		t.Errorf("hermite domain %v", d)
	}
}

func TestBadBasis(t *testing.T) {
	bad := Basis(42)
	c := []float64{1, 2, 3}
	if _, err := Val(bad, c, 1); !errors.Is(err, ErrBasis) {
		t.Error("Val", err)
	}
	if _, err := ValAll(bad, nil, c, []float64{1}); !errors.Is(err, ErrBasis) {
		t.Error("ValAll", err)
	}
	if _, err := Vander(bad, []float64{1}, 2); !errors.Is(err, ErrBasis) {
		t.Error("Vander", err)
	}
	if _, err := Der(bad, c, 1, 1); !errors.Is(err, ErrBasis) {
		t.Error("Der", err)
	}
	if _, err := Int(bad, c, 1, nil, 0, 1); !errors.Is(err, ErrBasis) {
		t.Error("Int", err)
	}
	if _, err := RootsBasis(bad, c); !errors.Is(err, ErrBasis) {
		t.Error("RootsBasis", err)
	}
	if _, err := NewSeries(bad, c); !errors.Is(err, ErrBasis) {
		t.Error("NewSeries", err)
	}
	if _, err := FitBasis(bad, []float64{1, 2}, []float64{1, 2}, 1, nil, 0); !errors.Is(err, ErrBasis) {
		t.Error("FitBasis", err)
	}
}

func TestBasisArgumentErrors(t *testing.T) {
	if _, err := Val(Chebyshev, nil, 1); !errors.Is(err, ErrEmpty) {
		t.Error(err)
	}
	if _, err := ValAll(Chebyshev, nil, nil, []float64{1}); !errors.Is(err, ErrEmpty) {
		t.Error(err)
	}
	if _, err := Vander(Chebyshev, []float64{1}, -1); !errors.Is(err, ErrArg) {
		t.Error(err)
	}
	if _, err := Der(Chebyshev, []float64{1}, -1, 1); !errors.Is(err, ErrArg) {
		t.Error(err)
	}
	if _, err := Int(Chebyshev, []float64{1}, -1, nil, 0, 1); !errors.Is(err, ErrArg) {
		t.Error(err)
	}
	if _, err := Int(Chebyshev, []float64{1}, 1, []float64{1, 2}, 0, 1); !errors.Is(err, ErrArg) {
		t.Error(err)
	}
	if _, err := Int(Chebyshev, nil, 1, nil, 0, 1); !errors.Is(err, ErrEmpty) {
		t.Error(err)
	}
	if _, err := RootsBasis(Chebyshev, nil); !errors.Is(err, ErrEmpty) {
		t.Error(err)
	}
	if _, err := NewSeries(Chebyshev, nil); !errors.Is(err, ErrEmpty) {
		t.Error(err)
	}
	if _, err := GetDomain(nil); !errors.Is(err, ErrEmpty) {
		t.Error(err)
	}
	if d, _ := GetDomain([]float64{3, math.NaN(), 1}); !math.IsNaN(d[0]) || !math.IsNaN(d[1]) {
		t.Errorf("NaN domain %v", d)
	}
	if d, _ := GetDomain([]float64{3, -1, 5}); d != [2]float64{-1, 5} {
		t.Errorf("domain %v", d)
	}
	// numpy: the derivative of an empty series is an empty series
	if d, err := Der(Legendre, nil, 2, 1); err != nil || len(d) != 0 {
		t.Errorf("%v %v", d, err)
	}
	// dst reuse
	buf := make([]float64, 8)
	out, _ := ValAll(Legendre, buf, []float64{1, 1}, []float64{0.5, 2})
	if &out[0] != &buf[0] || out[0] != 1.5 || out[1] != 3 {
		t.Errorf("%v", out)
	}
}

func TestSeriesMethods(t *testing.T) {
	s, err := NewSeries(Chebyshev, []float64{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	s.Domain = [2]float64{0, 4} // x -> x/2 - 1
	// T0 + 2 T1 + 3 T2 at x = 3 -> u = 0.5: 1 + 1 + 3*(2*0.25-1) = 0.5
	v, err := s.Val(3)
	if err != nil || v != 0.5 {
		t.Errorf("Val = %v %v", v, err)
	}
	all, err := s.ValAll(nil, []float64{3, 4})
	if err != nil || all[0] != 0.5 || all[1] != 1+2+3 {
		t.Errorf("ValAll = %v %v", all, err)
	}
	// derivative then integral (lbnd at the domain origin) gives the series back up to the constant
	d, err := s.Der(1)
	if err != nil {
		t.Fatal(err)
	}
	lb := 0.0
	back, err := d.Int(1, []float64{1}, &lb)
	if err != nil {
		t.Fatal(err)
	}
	a0, _ := s.Val(0)
	b0, _ := back.Val(0)
	if b0 != 1 { // the constant k fixes the value at lbnd
		t.Errorf("back(lbnd) = %v, want 1", b0)
	}
	for _, x := range []float64{0, 1, 2.5, 4} {
		a, _ := s.Val(x)
		b, _ := back.Val(x)
		if math.Abs((a-a0)-(b-b0)) > 1e-14 {
			t.Errorf("round trip at %v: %v vs %v", x, a-a0, b-b0)
		}
	}
	if _, err := (Series{Basis: Chebyshev, Domain: [2]float64{-1, 1}, Window: [2]float64{-1, 1}}).Der(1); err != nil {
		t.Errorf("empty series Der: %v", err)
	}
	if _, err := (Series{Basis: Chebyshev}).Int(1, nil, nil); !errors.Is(err, ErrEmpty) {
		t.Error(err)
	}
	if _, err := (Series{Basis: Chebyshev}).Roots(); !errors.Is(err, ErrEmpty) {
		t.Error(err)
	}
	// roots of T2 = 2x^2-1 on the domain [0,4]: u = ±1/sqrt2 -> x = 2(u+1)
	r, err := Series{Coef: []float64{0, 0, 1}, Domain: [2]float64{0, 4}, Window: [2]float64{-1, 1}, Basis: Chebyshev}.Roots()
	if err != nil || len(r) != 2 || math.Abs(real(r[0])-(2-math.Sqrt2)) > 1e-14 || math.Abs(real(r[1])-(2+math.Sqrt2)) > 1e-14 {
		t.Errorf("roots %v %v", r, err)
	}
	if _, err := (Series{Coef: []float64{}, Basis: Chebyshev}).Val(0); !errors.Is(err, ErrEmpty) {
		t.Error(err)
	}
	if _, err := (Series{Coef: []float64{}, Basis: Chebyshev}).ValAll(nil, []float64{0}); !errors.Is(err, ErrEmpty) {
		t.Error(err)
	}
	if _, err := (Series{Coef: []float64{}, Basis: Chebyshev}).Der(0); err != nil {
		t.Error(err)
	}
}

func TestFitSeries(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6}
	y := []float64{1, 4, 9, 16, 25, 36}
	s, f, err := FitSeries(Legendre, x, y, 2, nil)
	if err != nil || f.Rank != 3 || s.Domain != [2]float64{1, 6} || s.Window != [2]float64{-1, 1} {
		t.Fatalf("%+v %+v %v", s, f, err)
	}
	for i, xv := range x {
		if v, _ := s.Val(xv); math.Abs(v-y[i]) > 1e-12 {
			t.Errorf("fit(%v) = %v want %v", xv, v, y[i])
		}
	}
	dom, win := [2]float64{0, 10}, [2]float64{0, 1}
	s, _, err = FitSeries(Laguerre, x, y, 2, &SeriesFitOptions{Domain: &dom, Window: &win, W: []float64{1, 1, 1, 1, 1, 1}})
	if err != nil || s.Domain != dom || s.Window != win {
		t.Errorf("%+v %v", s, err)
	}
	// equal x widen the domain by one on each side
	s, _, err = FitSeries(Chebyshev, []float64{2, 2, 2}, []float64{1, 2, 3}, 1, nil)
	if err != nil || s.Domain != [2]float64{1, 3} {
		t.Errorf("%+v %v", s, err)
	}
	if _, _, err := FitSeries(Chebyshev, nil, nil, 1, nil); !errors.Is(err, ErrEmpty) {
		t.Error(err)
	}
	if _, _, err := FitSeries(Chebyshev, x, y[:2], 1, nil); !errors.Is(err, ErrArg) {
		t.Error(err)
	}
	if _, err := FitBasis(Chebyshev, x, y, -1, nil, 0); !errors.Is(err, ErrArg) {
		t.Error(err)
	}
	if _, err := FitBasis(Chebyshev, x, y, 1, []float64{1}, 0); !errors.Is(err, ErrArg) {
		t.Error(err)
	}
	if _, err := FitBasis(Chebyshev, x, y, -1, nil, 0); err == nil {
		t.Error("negative degree accepted")
	}
	if _, err := FitBasis(Chebyshev, x, y, 1, []float64{1, 2, 3, 4, 5, math.NaN()}, 0); !errors.Is(err, ErrNonFinite) {
		t.Error(err)
	}
}

func TestPolyfitOptionsAndErrors(t *testing.T) {
	x := []float64{0, 1, 2, 3, 4}
	y := []float64{1, 3, 7, 13, 21} // x^2 + x + 1
	f, err := Polyfit(x, y, 2, nil)
	if err != nil || f.Rank != 3 || f.Cov != nil || len(f.Residuals) != 1 {
		t.Fatalf("%+v %v", f, err)
	}
	for i, w := range []float64{1, 1, 1} {
		if math.Abs(f.Coef[i]-w) > 1e-12 {
			t.Errorf("coef %v", f.Coef)
		}
	}
	for _, tc := range []struct {
		name string
		x, y []float64
		deg  int
		o    *FitOptions
	}{
		{"neg deg", x, y, -1, nil},
		{"empty", nil, nil, 1, nil},
		{"mismatch", x, y[:3], 1, nil},
		{"weights mismatch", x, y, 1, &FitOptions{W: []float64{1}}},
		{"scaled cov needs m > n", x[:3], y[:3], 2, &FitOptions{Cov: CovScaled}},
	} {
		if _, err := Polyfit(tc.x, tc.y, tc.deg, tc.o); !errors.Is(err, ErrArg) {
			t.Errorf("%s: %v", tc.name, err)
		}
	}
	// an out-of-range rcond means machine precision
	if f, err := Polyfit(x, y, 2, &FitOptions{Rcond: -1}); err != nil || f.Rank != 3 {
		t.Errorf("%+v %v", f, err)
	}
	// unscaled covariance of a square (m == n) system is fine
	f, err = Polyfit(x[:3], y[:3], 2, &FitOptions{Cov: CovUnscaled})
	if err != nil || len(f.Cov) != 9 || f.Residuals != nil {
		t.Errorf("%+v %v", f, err)
	}
	// rank-deficient fits have no covariance
	if _, err := Polyfit([]float64{1, 1, 1, 1}, []float64{1, 2, 3, 4}, 2, &FitOptions{Cov: CovUnscaled}); !errors.Is(err, ErrRankDeficient) {
		t.Errorf("%v", err)
	}
	// NaN data: NaN in x is an error, NaN in y gives NaN coefficients
	if _, err := Polyfit([]float64{0, math.NaN(), 2}, []float64{1, 2, 3}, 1, nil); !errors.Is(err, ErrNonFinite) {
		t.Error(err)
	}
	f, err = Polyfit(x, []float64{1, math.NaN(), 3, 4, 5}, 1, nil)
	if err != nil || !math.IsNaN(f.Coef[0]) || !math.IsNaN(f.Residuals[0]) || f.Rank != 2 {
		t.Errorf("%+v %v", f, err)
	}
	// underdetermined: minimum-norm solution through the SVD fallback
	f, err = Polyfit([]float64{2}, []float64{6}, 2, nil)
	if err != nil || f.Rank != 1 || math.Abs(f.Coef[0]*4+f.Coef[1]*2+f.Coef[2]-6) > 1e-12 || f.Residuals != nil {
		t.Errorf("%+v %v", f, err)
	}
	// a failing SVD is reported
	boom := errors.New("svd failed")
	old := lstsqFn
	lstsqFn = func(a, b *ndarray.Array, rcond float64) (linalg.LstsqResult, error) {
		return linalg.LstsqResult{}, boom
	}
	defer func() { lstsqFn = old }()
	if _, err := Polyfit([]float64{2}, []float64{6}, 2, nil); !errors.Is(err, boom) {
		t.Errorf("%v", err)
	}
}

func TestRootsEigenFailure(t *testing.T) {
	boom := errors.New("no convergence")
	old := eigvalsFn
	eigvalsFn = func(*ndarray.Array) (*ndarray.Array, error) { return nil, boom }
	defer func() { eigvalsFn = old }()
	if _, err := Roots([]float64{1, 2, 3}); !errors.Is(err, boom) {
		t.Error(err)
	}
	if _, err := RootsBasis(Chebyshev, []float64{1, 2, 3}); !errors.Is(err, boom) {
		t.Error(err)
	}
}

func TestRootsSmall(t *testing.T) {
	r, err := Roots([]float64{1, 0, -4, 0, 0})
	if err != nil || len(r) != 4 || r[2] != 0 || r[3] != 0 {
		t.Fatalf("%v %v", r, err)
	}
	if got := math.Abs(real(r[0])) + math.Abs(real(r[1])); math.Abs(got-4) > 1e-14 {
		t.Errorf("%v", r)
	}
	// two-term series use the closed form, per family
	for b, want := range map[Basis]float64{Chebyshev: -1.5, Legendre: -1.5, Hermite: -0.75, HermiteE: -1.5, Laguerre: 2.5} {
		r, err := RootsBasis(b, []float64{3, 2, 0})
		if err != nil || len(r) != 1 || real(r[0]) != want {
			t.Errorf("%v: %v %v", b, r, err)
		}
	}
	if r, _ := RootsBasis(Hermite, []float64{5}); len(r) != 0 {
		t.Errorf("%v", r)
	}
}

func TestPolyBasics(t *testing.T) {
	if got := Polyval([]float64{2, -3, 1}, 4); got != 21 {
		t.Error(got)
	}
	if got := PolyvalComp(nil, 4); got != 0 {
		t.Error(got)
	}
	if got := PolyvalComp([]float64{2, -3, 1}, 4); got != 21 {
		t.Error(got)
	}
	buf := make([]float64, 4)
	if out := PolyvalAll(buf, []float64{1, 0}, []float64{5, 6}); &out[0] != &buf[0] || out[1] != 6 || len(out) != 2 {
		t.Error(out)
	}
	if out := PolyvalCompAll(buf, []float64{1, 0}, []float64{5, 6}); &out[0] != &buf[0] || out[1] != 6 {
		t.Error(out)
	}
	if d, err := Polyder([]float64{1, 2, 3}, 1); err != nil || len(d) != 2 || d[0] != 2 || d[1] != 2 {
		t.Error(d, err)
	}
	if _, err := Polyder(nil, -1); !errors.Is(err, ErrArg) {
		t.Error(err)
	}
	if _, err := Polyint(nil, -1, nil); !errors.Is(err, ErrArg) {
		t.Error(err)
	}
	if _, err := Conv(nil, []float64{1}); !errors.Is(err, ErrEmpty) {
		t.Error(err)
	}
	if _, _, err := Deconv([]float64{1}, nil); !errors.Is(err, ErrEmpty) {
		t.Error(err)
	}
	q, r, err := Deconv([]float64{1, 3, 3, 1}, []float64{1, 1})
	if err != nil || len(q) != 3 || q[0] != 1 || q[1] != 2 || q[2] != 1 || len(r) != 1 || r[0] != 0 {
		t.Error(q, r, err)
	}
	if got := Polymul([]float64{0, 1, 1}, []float64{1, -1}); len(got) != 2+0 && len(got) != 3 || got[0] != 1 {
		t.Error(got)
	}
	if got := Polyadd([]float64{1, 2, 3}, []float64{4}); got[2] != 7 || got[0] != 1 {
		t.Error(got)
	}
	if got := Polysub([]float64{4}, []float64{1, 2, 3}); got[0] != -1 || got[2] != 1 {
		t.Error(got)
	}
	if got := PolyFromRoots([]float64{1, 2}); got[0] != 1 || got[1] != -3 || got[2] != 2 {
		t.Error(got)
	}
	// numpy.poly: conjugate-closed complex roots give real coefficients
	c, realOnly := PolyFromComplexRoots([]complex128{complex(0, 1), complex(0, -1)})
	if !realOnly || c[0] != 1 || c[1] != 0 || c[2] != 1 {
		t.Error(c, realOnly)
	}
	c, realOnly = PolyFromComplexRoots([]complex128{complex(0, 1)})
	if realOnly || c[1] != complex(0, -1) {
		t.Error(c, realOnly)
	}
}
