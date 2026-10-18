//go:build oracle

package poly

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/go-approx/poly/internal/oracle"
)

func uptime() string {
	b, err := exec.Command("uptime").Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(b))
}

func median(v []float64) float64 {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	return s[len(s)/2]
}

type perfCase struct {
	op   string
	n    int
	goFn func()
	call oracle.Call
}

// TestPerf compares the library with NumPy on one core with interleaved
// rounds and writes docs/perf.md:
//
//	GOWORK=off go test -tags oracle -run TestPerf -timeout 3h .
func TestPerf(t *testing.T) {
	runtime.GOMAXPROCS(1)
	rounds := 9
	before := uptime()
	u, err := oracle.StartUV()
	if err != nil {
		t.Fatal(err)
	}
	defer u.Close()
	cl := oracle.Client{T: u}
	if err := cl.HealthCheck(); err != nil {
		t.Fatal(err)
	}
	var cases []perfCase
	sizes := []int{1000, 10000, 100000, 1000000}
	r := newRNG(99)
	rc := unif(r, 21, -1, 1)
	p5 := unif(r, 6, -1, 1)
	for _, n := range sizes {
		x := unif(r, n, -1, 1)
		y := unif(r, n, -1, 1)
		xn, yn := fmt.Sprintf("x%d", n), fmt.Sprintf("y%d", n)
		if _, err := cl.Batch(oracle.Set{Name: xn, V: oracle.Arr(x)}, oracle.Set{Name: yn, V: oracle.Arr(y)}); err != nil {
			t.Fatal(err)
		}
		cases = append(cases,
			perfCase{"polyval deg5", n, func() { PolyvalAll(nil, p5, x) }, oracle.Call{Fn: "np.polyval", Args: []any{oracle.Arr(p5), oracle.Ref(xn)}}},
			perfCase{"polyfit deg3", n, func() { _, _ = Polyfit(x, y, 3, nil) }, oracle.Call{Fn: "np.polyfit", Args: []any{oracle.Ref(xn), oracle.Ref(yn), 3}}},
			perfCase{"polyfit deg10", n, func() { _, _ = Polyfit(x, y, 10, nil) }, oracle.Call{Fn: "np.polyfit", Args: []any{oracle.Ref(xn), oracle.Ref(yn), 10}}},
			perfCase{"chebval 21 coef", n, func() { _, _ = ValAll(Chebyshev, nil, rc, x) }, oracle.Call{Fn: "np.polynomial.chebyshev.chebval", Args: []any{oracle.Ref(xn), oracle.Arr(rc)}}},
			perfCase{"chebfit deg10", n, func() { _, _ = FitBasis(Chebyshev, x, y, 10, nil, 0) }, oracle.Call{Fn: "np.polynomial.chebyshev.chebfit", Args: []any{oracle.Ref(xn), oracle.Ref(yn), 10}}},
			perfCase{"chebvander deg10", n, func() { _, _ = Vander(Chebyshev, x, 10) }, oracle.Call{Fn: "np.polynomial.chebyshev.chebvander", Args: []any{oracle.Ref(xn), 10}}},
		)
	}
	for _, deg := range []int{10, 100} {
		p := unif(r, deg+1, -1, 1)
		cases = append(cases, perfCase{"roots", deg, func() { _, _ = Roots(p) }, oracle.Call{Fn: "np.roots", Args: []any{oracle.Arr(p)}}})
	}
	var sb strings.Builder
	fmt.Fprintf(&sb, "# Performance against NumPy %s\n\n", oracle.NumPyVersion)
	fmt.Fprintf(&sb, "One core (GOMAXPROCS=1; OPENBLAS/OMP/MKL threads=1 for NumPy), Go %s %s/%s, interleaved rounds (Go, NumPy, Go, NumPy, ...), median of %d rounds. Target: no slower than NumPy (ratio <= 1).\n\n", runtime.Version(), runtime.GOOS, runtime.GOARCH, rounds)
	fmt.Fprintf(&sb, "Load before: `%s`\n\n", before)
	fmt.Fprintf(&sb, "| operation | n (or degree) | Go | NumPy | Go / NumPy |\n|---|---:|---:|---:|---:|\n")
	for _, c := range cases {
		// calibrate: repeat so that a sample lasts >= 20 ms (>= 1 repetition)
		c.goFn()
		t0 := time.Now()
		c.goFn()
		one := time.Since(t0)
		rep := max(1, int(20*time.Millisecond/max(one, time.Microsecond)))
		c.call.Rep = rep
		if _, err := cl.Do(c.call); err != nil { // warm-up
			t.Fatal(err)
		}
		var gt, nt []float64
		for i := 0; i < rounds; i++ {
			t0 := time.Now()
			for k := 0; k < rep; k++ {
				c.goFn()
			}
			gt = append(gt, time.Since(t0).Seconds()/float64(rep))
			res, err := cl.Do(c.call)
			if err != nil || res.Err != "" {
				t.Fatalf("%s: %v %s", c.op, err, res.Err)
			}
			nt = append(nt, res.Time)
		}
		g, n := median(gt), median(nt)
		fmt.Fprintf(&sb, "| %s | %d | %s | %s | %.2f |\n", c.op, c.n, dur(g), dur(n), g/n)
		t.Logf("%s n=%d go=%s numpy=%s ratio=%.2f (rep %d)", c.op, c.n, dur(g), dur(n), g/n, rep)
	}
	fmt.Fprintf(&sb, "\nLoad after: `%s`\n", uptime())
	if err := os.MkdirAll("docs", 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile("docs/perf.md", []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}

func dur(s float64) string {
	switch {
	case s < 1e-6:
		return fmt.Sprintf("%.0f ns", s*1e9)
	case s < 1e-3:
		return fmt.Sprintf("%.1f µs", s*1e6)
	case s < 1:
		return fmt.Sprintf("%.2f ms", s*1e3)
	}
	return fmt.Sprintf("%.2f s", s)
}

var _ = math.Pi
