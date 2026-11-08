package bench

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/go-approx/poly"
	"github.com/go-scikit/refbench"
	"github.com/go-scikit/refbench/rbtest"
)

// The speed comparison with NumPy (docs/perf.md).
//
//	cd bench
//	go test -run TestPerf .                          CI mode: allocations, table, inventory
//	go test -run TestPerf -cpu 1 -perf.run .         measure (refuses at a 1-minute load of 8 or more)
//	go test -run TestPerf -perf.record .             allocation counts only (a deferred report)
//	go test -run TestDataParity -perf.parity .       both sides get the same bytes (needs uv)
//	go test -run TestRefSmoke -perf.smoke .          every NumPy call runs once (needs uv)
//
// What is compared. Every row times one call on both sides, on the same data:
// the Go body is the public function, the NumPy body is the numpy / numpy.polynomial
// function it ports (the reference is numpy 2.5.4, pinned in refs.txt). The "loop"
// rows call a scalar Go function over the n points and compare with the one array
// call of NumPy, as the scalar function has no array counterpart there. The arrays
// have n = 10 000 points (80 kB, so that both sides run from cache), the series
// and polynomials 21 to 1000 coefficients, as each row says. Parity is the claim
// of the package for that function (docs/accuracy.md of the repository).

const n = 10_000

const (
	gold    = 0.6180339887498949 // x
	plastic = 0.7548776662466927 // y
)

// pts reproduces the Python setup lo + (hi-lo)·frac(i·mult). The explicit float64
// conversion forbids a fused multiply-add, so the bytes are the same on arm64.
func pts(lo, hi, mult float64) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = lo + float64((hi-lo)*math.Mod(float64(i)*mult, 1))
	}
	return x
}

func ptsPy(lo, hi, mult float64) string {
	return fmt.Sprintf("%v + (%v - %v) * np.mod(np.arange(n, dtype=np.float64) * %v, 1.0)", lo, hi, lo, mult)
}

// coefs are the alternating 1/(i+1): the same bytes on both sides (one division).
func coefs(m int) []float64 {
	c := make([]float64, m)
	for i := range c {
		s := 1.0
		if i%2 == 1 {
			s = -1
		}
		c[i] = s / float64(i+1)
	}
	return c
}

func coefsPy(m int) string {
	return fmt.Sprintf("(1.0 - 2.0 * (np.arange(%d) %% 2)) / (np.arange(%d) + 1.0)", m, m)
}

// conj builds 2m roots closed under conjugation: re ± i·im.
func conj(re, im []float64) []complex128 {
	r := make([]complex128, 0, 2*len(re))
	for i := range re {
		r = append(r, complex(re[i], im[i]))
	}
	for i := range re {
		r = append(r, complex(re[i], -im[i]))
	}
	return r
}

const conjPy = "np.concatenate([a + 1j * b, a - 1j * b])"

var (
	sink    float64
	sinkS   []float64
	sinkC   []complex128
	sinkErr error
)

// arg is one array handed to both sides: the Go values (a real array, or a complex
// one flattened as re, im pairs) and the Python expression that builds them.
type arg struct {
	data []float64
	py   string
}

func real_(d []float64, py string) arg { return arg{d, py} }

func cplx(r []complex128, py string) arg {
	d := make([]float64, 0, 2*len(r))
	for _, z := range r {
		d = append(d, real(z), imag(z))
	}
	return arg{d, py}
}

// row is one case: it builds the Go body once (its buffers are allocated outside
// the timed loop).
type row struct {
	name   string
	fn     string // the public function
	ref    string // the NumPy callable
	parity string
	args   []arg
	series string        // for a series row: what is applied to the NumPy series
	mk     func() func() // one call of the Go function
}

// label is the reference as the table shows it: the callable, without the
// lambda wrapper, and for a series row the method of a Chebyshev series.
func (r row) label() string {
	if strings.Contains(r.ref, serPy) {
		return "np.polynomial.Chebyshev(c)" + r.series
	}
	if strings.HasPrefix(r.ref, "lambda") {
		_, l, _ := strings.Cut(r.ref, ": ")
		return l
	}
	return r.ref
}

func (r row) setup() string {
	p := make([]string, len(r.args))
	for i, a := range r.args {
		p[i] = a.py
	}
	return "[" + strings.Join(p, ", ") + "]"
}

var (
	x  = pts(-1, 1, gold)
	y  = pts(-1, 1, plastic)
	xa = real_(x, ptsPy(-1, 1, gold))
	ya = real_(y, ptsPy(-1, 1, plastic))
)

func ca(m int) arg { return real_(coefs(m), coefsPy(m)) }

// bases: the five families of numpy.polynomial.
var bases = []struct {
	b        poly.Basis
	name     string
	mod, pre string
}{
	{poly.Chebyshev, "chebyshev", "chebyshev", "cheb"},
	{poly.Legendre, "legendre", "legendre", "leg"},
	{poly.Hermite, "hermite", "hermite", "herm"},
	{poly.HermiteE, "hermite_e", "hermite_e", "herme"},
	{poly.Laguerre, "laguerre", "laguerre", "lag"},
}

func npf(mod, name string) string { return "np.polynomial." + mod + "." + name }

// ser is the NumPy series the Series rows are built from (a Chebyshev series of 31
// coefficients on its default domain).
const serPy = "np.polynomial.Chebyshev(" + "(1.0 - 2.0 * (np.arange(31) % 2)) / (np.arange(31) + 1.0)" + ")"

func rows() []row {
	var rs []row
	add := func(r row) { rs = append(rs, r) }
	const ex, tol = refbench.ParityExact, refbench.ParityTolerance

	// numpy.polyval: coefficients highest degree first.
	p6, p21 := ca(6), ca(21)
	dst := make([]float64, n)
	for _, p := range []struct {
		tag string
		c   arg
	}{{"deg5", p6}, {"deg20", p21}} {
		c := p.c.data
		add(row{name: "Polyval loop " + p.tag, fn: "Polyval", ref: "np.polyval", parity: ex, args: []arg{p.c, xa},
			mk: func() func() {
				return func() {
					for i, v := range x {
						dst[i] = poly.Polyval(c, v)
					}
					sink = dst[0]
				}
			}})
		add(row{name: "PolyvalAll " + p.tag, fn: "PolyvalAll", ref: "np.polyval", parity: ex, args: []arg{p.c, xa},
			mk: func() func() {
				return func() { sinkS = poly.PolyvalAll(dst, c, x) }
			}})
	}

	for _, bs := range bases {
		bs := bs
		c21 := ca(21)
		add(row{name: "Val loop " + bs.name, fn: "Val", ref: npf(bs.mod, bs.pre+"val"), parity: ex, args: []arg{xa, c21},
			mk: func() func() {
				return func() {
					for i, v := range x {
						dst[i], sinkErr = poly.Val(bs.b, c21.data, v)
					}
					sink = dst[0]
				}
			}})
		add(row{name: "ValAll " + bs.name, fn: "ValAll", ref: npf(bs.mod, bs.pre+"val"), parity: ex, args: []arg{xa, c21},
			mk: func() func() {
				return func() { sinkS, sinkErr = poly.ValAll(bs.b, dst, c21.data, x) }
			}})
		add(row{name: "Vander " + bs.name + " deg10", fn: "Vander", parity: ex, args: []arg{xa},
			ref: "lambda x: " + npf(bs.mod, bs.pre+"vander") + "(x, 10)",
			mk: func() func() {
				return func() { sinkS, sinkErr = poly.Vander(bs.b, x, 10) }
			}})
		c200 := ca(200)
		add(row{name: "Der " + bs.name + " 200 coef, m=2", fn: "Der", parity: ex, args: []arg{c200},
			ref: "lambda c: " + npf(bs.mod, bs.pre+"der") + "(c, 2, 1.0)",
			mk: func() func() {
				return func() { sinkS, sinkErr = poly.Der(bs.b, c200.data, 2, 1) }
			}})
		k := []float64{0, 0}
		add(row{name: "Int " + bs.name + " 200 coef, m=2", fn: "Int", parity: ex, args: []arg{c200},
			ref: "lambda c: " + npf(bs.mod, bs.pre+"int") + "(c, 2, [0.0, 0.0], 0.0, 1.0)",
			mk: func() func() {
				return func() { sinkS, sinkErr = poly.Int(bs.b, c200.data, 2, k, 0, 1) }
			}})
	}

	// polyutils.
	dom, win := [2]float64{-1, 1}, [2]float64{0, 10}
	add(row{name: "MapParms", fn: "MapParms", parity: ex, ref: "lambda: np.polynomial.polyutils.mapparms([-1.0, 1.0], [0.0, 10.0])",
		mk: func() func() {
			return func() { off, scl := poly.MapParms(dom, win); sink = off + scl }
		}})
	add(row{name: "MapDomain", fn: "MapDomain", parity: ex, args: []arg{xa, real_([]float64{-1, 1}, "np.array([-1.0, 1.0])"), real_([]float64{0, 10}, "np.array([0.0, 10.0])")},
		ref: "np.polynomial.polyutils.mapdomain",
		mk: func() func() {
			return func() { sinkS = poly.MapDomain(dst, x, dom, win) }
		}})
	add(row{name: "GetDomain", fn: "GetDomain", parity: ex, args: []arg{xa}, ref: "np.polynomial.polyutils.getdomain",
		mk: func() func() {
			return func() { d, _ := poly.GetDomain(x); sink = d[0] + d[1] }
		}})

	// numpy.poly1d-style functions, highest degree first.
	p1000, p100 := ca(1000), ca(100)
	add(row{name: "Polyder 1000 coef, m=2", fn: "Polyder", parity: ex, args: []arg{p1000}, ref: "lambda p: np.polyder(p, 2)",
		mk: func() func() { return func() { sinkS, sinkErr = poly.Polyder(p1000.data, 2) } }})
	add(row{name: "Polyint 1000 coef, m=2", fn: "Polyint", parity: ex, args: []arg{p1000}, ref: "lambda p: np.polyint(p, 2)",
		mk: func() func() { return func() { sinkS, sinkErr = poly.Polyint(p1000.data, 2, nil) } }})
	add(row{name: "Conv 1000x100", fn: "Conv", parity: tol, args: []arg{p1000, p100}, ref: "np.convolve",
		mk: func() func() { return func() { sinkS, sinkErr = poly.Conv(p1000.data, p100.data) } }})
	add(row{name: "Polymul 1000x100", fn: "Polymul", parity: tol, args: []arg{p1000, p100}, ref: "np.polymul",
		mk: func() func() { return func() { sinkS = poly.Polymul(p1000.data, p100.data) } }})
	add(row{name: "Polyadd 1000+100", fn: "Polyadd", parity: ex, args: []arg{p1000, p100}, ref: "np.polyadd",
		mk: func() func() { return func() { sinkS = poly.Polyadd(p1000.data, p100.data) } }})
	add(row{name: "Polysub 1000-100", fn: "Polysub", parity: ex, args: []arg{p1000, p100}, ref: "np.polysub",
		mk: func() func() { return func() { sinkS = poly.Polysub(p1000.data, p100.data) } }})
	u200, v20 := ca(200), ca(20)
	add(row{name: "Deconv 200/20", fn: "Deconv", parity: tol, args: []arg{u200, v20}, ref: "np.polydiv",
		mk: func() func() {
			return func() { sinkS, _, sinkErr = poly.Deconv(u200.data, v20.data) }
		}})
	r100 := real_(x[:100], "("+ptsPy(-1, 1, gold)+")[:100]")
	add(row{name: "PolyFromRoots 100", fn: "PolyFromRoots", parity: tol, args: []arg{r100}, ref: "np.poly",
		mk: func() func() { return func() { sinkS = poly.PolyFromRoots(r100.data) } }})
	re, im := x[:50], pts(0.1, 1.1, plastic)[:50]
	cr := cplx(conj(re, im), "(lambda a, b: "+conjPy+")(("+ptsPy(-1, 1, gold)+")[:50], ("+ptsPy(0.1, 1.1, plastic)+")[:50])")
	croots := conj(re, im)
	add(row{name: "PolyFromComplexRoots 100", fn: "PolyFromComplexRoots", parity: tol, args: []arg{cr}, ref: "np.poly",
		mk: func() func() { return func() { sinkC, _ = poly.PolyFromComplexRoots(croots) } }})
	for _, deg := range []int{3, 10} {
		deg := deg
		add(row{name: fmt.Sprintf("Polyfit deg%d", deg), fn: "Polyfit", parity: tol, args: []arg{xa, ya},
			ref: fmt.Sprintf("lambda x, y: np.polyfit(x, y, %d)", deg),
			mk: func() func() {
				return func() { f, _ := poly.Polyfit(x, y, deg, nil); sinkS = f.Coef }
			}})
	}
	p31 := ca(31)
	add(row{name: "Roots deg30", fn: "Roots", parity: tol, args: []arg{p31}, ref: "np.roots",
		mk: func() func() { return func() { sinkC, sinkErr = poly.Roots(p31.data) } }})

	// numpy.polynomial series.
	s31 := coefs(31)
	ser, _ := poly.NewSeries(poly.Chebyshev, s31)
	add(row{name: "NewSeries 31 coef", fn: "NewSeries", parity: ex, args: []arg{p31}, ref: "np.polynomial.Chebyshev",
		mk: func() func() { return func() { s, _ := poly.NewSeries(poly.Chebyshev, s31); sinkS = s.Coef } }})
	sv := "(lambda s: lambda x: s(x))(" + serPy + ")"
	add(row{name: "Series.Val loop", fn: "Series.Val", parity: tol, args: []arg{xa}, ref: sv, series: "(x)",
		mk: func() func() {
			return func() {
				for i, v := range x {
					dst[i], sinkErr = ser.Val(v)
				}
				sink = dst[0]
			}
		}})
	add(row{name: "Series.ValAll", fn: "Series.ValAll", parity: tol, args: []arg{xa}, ref: sv, series: "(x)",
		mk: func() func() { return func() { sinkS, sinkErr = ser.ValAll(dst, x) } }})
	add(row{name: "Series.Der", fn: "Series.Der", parity: ex, series: ".deriv()", ref: "(lambda s: lambda: s.deriv())(" + serPy + ")",
		mk: func() func() { return func() { s, _ := ser.Der(1); sinkS = s.Coef } }})
	add(row{name: "Series.Int", fn: "Series.Int", parity: ex, series: ".integ()", ref: "(lambda s: lambda: s.integ())(" + serPy + ")",
		mk: func() func() { return func() { s, _ := ser.Int(1, nil, nil); sinkS = s.Coef } }})
	add(row{name: "Series.Roots deg30", fn: "Series.Roots", parity: tol, series: ".roots()", ref: "(lambda s: lambda: s.roots())(" + serPy + ")",
		mk: func() func() { return func() { sinkC, sinkErr = ser.Roots() } }})
	add(row{name: "FitBasis chebyshev deg10", fn: "FitBasis", parity: tol, args: []arg{xa, ya},
		ref: "lambda x, y: np.polynomial.chebyshev.chebfit(x, y, 10)",
		mk: func() func() {
			return func() { f, _ := poly.FitBasis(poly.Chebyshev, x, y, 10, nil, 0); sinkS = f.Coef }
		}})
	add(row{name: "FitSeries chebyshev deg10", fn: "FitSeries", parity: tol, args: []arg{xa, ya},
		ref: "lambda x, y: np.polynomial.Chebyshev.fit(x, y, 10)",
		mk: func() func() {
			return func() { s, _, _ := poly.FitSeries(poly.Chebyshev, x, y, 10, nil); sinkS = s.Coef }
		}})
	add(row{name: "RootsBasis chebyshev deg30", fn: "RootsBasis", parity: tol, args: []arg{p31},
		ref: "np.polynomial.chebyshev.chebroots",
		mk:  func() func() { return func() { sinkC, sinkErr = poly.RootsBasis(poly.Chebyshev, p31.data) } }})
	return rs
}

// worker is the NumPy process, started once per test binary run that times.
func startWorker(t testing.TB) *refbench.Worker {
	t.Helper()
	uv, err := exec.LookPath("uv")
	if err != nil {
		home, _ := os.UserHomeDir()
		uv = filepath.Join(home, ".local", "bin", "uv")
	}
	refs, err := filepath.Abs("refs.txt")
	if err != nil {
		t.Fatal(err)
	}
	w, err := refbench.StartPython(context.Background(), uv, refs)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { w.Close() })
	return w
}

// cases builds the registry. ref is nil unless a worker is given (the CI mode
// and -perf.record need none).
func cases(w *refbench.Worker) []refbench.Case {
	var out []refbench.Case
	for _, r := range rows() {
		body := r.mk()
		var ref refbench.Timer
		if w != nil {
			ref = w.Timer(r.setup(), r.ref, n)
		}
		out = append(out, refbench.Case{
			Name:      r.name,
			Func:      r.fn,
			Reference: "numpy 2.5.4 " + r.label(),
			Parity:    r.parity,
			Go: func(iters int) {
				for ; iters > 0; iters-- {
					body()
				}
			},
			Ref: ref,
		})
	}
	return out
}

var (
	parityFlag = flag.Bool("perf.parity", false, "TestDataParity: compare the data digests with NumPy's")
	smokeFlag  = flag.Bool("perf.smoke", false, "TestRefSmoke: run every NumPy call once")
)

// running reports rbtest's -perf.run (the worker is started only then).
func running() bool {
	f := flag.Lookup("perf.run")
	return f != nil && f.Value.String() == "true"
}

func TestPerf(t *testing.T) {
	var w *refbench.Worker
	if running() {
		w = startWorker(t)
	}
	opt := refbench.Options{Repo: "go-approx/poly"}
	if w != nil {
		v, err := w.RefVersions()
		if err != nil {
			t.Fatal(err)
		}
		opt.RefVersions = v
	}
	rbtest.Perf(t, refbench.CIConfig{Root: "..", Cases: cases(w)}, opt)
}

// TestDataParity checks, for every row, that the Go arrays and NumPy's are the
// same bytes: a ratio between two functions on different data means nothing.
func TestDataParity(t *testing.T) {
	if !*parityFlag {
		t.Skip("set -perf.parity (starts the NumPy worker through uv)")
	}
	w := startWorker(t)
	for _, r := range rows() {
		if len(r.args) == 0 {
			continue
		}
		h := sha256.New()
		for _, a := range r.args {
			for _, v := range a.data {
				var b [8]byte
				binary.LittleEndian.PutUint64(b[:], math.Float64bits(v))
				h.Write(b[:])
			}
		}
		got, err := w.DataHash(r.setup(), n)
		if err != nil {
			t.Fatalf("%s: %v", r.name, err)
		}
		if want := hex.EncodeToString(h.Sum(nil)); got != want {
			t.Errorf("%s: NumPy data %s, Go data %s", r.name, got, want)
		}
	}
}

// TestRefSmoke runs every NumPy call once: a typo in a reference expression is a
// failure here, not in a timing run.
func TestRefSmoke(t *testing.T) {
	if !*smokeFlag {
		t.Skip("set -perf.smoke (starts the NumPy worker through uv)")
	}
	w := startWorker(t)
	for _, c := range cases(w) {
		if _, err := c.Ref(1); err != nil {
			t.Errorf("%s: %v", c.Name, err)
		}
	}
}
