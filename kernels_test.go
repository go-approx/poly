package poly

import (
	"math"
	"testing"
)

// The batch evaluators run go-simd/floats' Polyval and Chebval kernels. Their
// contract here: the bits of the scalar functions (which are numpy.polyval's and
// numpy.polynomial.chebyshev.chebval's operation order), for every coefficient
// count, with Inf, NaN and signed zeros among the points.

func kernelPoints() []float64 {
	x := unif(newRNG(21), 37, -2, 2)
	return append(x, math.Inf(1), math.Inf(-1), math.NaN(), 0, math.Copysign(0, -1), 1, -1, 1e300, -1e-300)
}

func sameBits(a, b float64) bool {
	return math.Float64bits(a) == math.Float64bits(b) || (math.IsNaN(a) && math.IsNaN(b))
}

func TestPolyvalAllMatchesScalar(t *testing.T) {
	x := kernelPoints()
	r := newRNG(22)
	for n := 1; n <= 70; n++ { // crosses the pooled buffer's initial capacity of 64
		p := unif(r, n, -3, 3)
		if n == 3 {
			p[0], p[2] = 0, math.Copysign(0, -1)
		}
		got := PolyvalAll(nil, p, x)
		for i, xv := range x {
			if w := Polyval(p, xv); !sameBits(got[i], w) {
				t.Fatalf("deg %d x=%v: PolyvalAll %v, Polyval %v", n-1, xv, got[i], w)
			}
		}
	}
}

func TestPolyvalAllDegenerate(t *testing.T) {
	dst := []float64{7, 7, 7}
	if out := PolyvalAll(dst, nil, []float64{1, 2}); out[0] != 0 || out[1] != 0 || len(out) != 2 {
		t.Errorf("empty p: %v", out)
	}
	if out := PolyvalAll(dst, []float64{1, 2}, nil); len(out) != 0 {
		t.Errorf("empty x: %v", out)
	}
}

func TestValAllChebyshevMatchesScalar(t *testing.T) {
	x := kernelPoints()
	r := newRNG(23)
	for n := 1; n <= 40; n++ {
		c := unif(r, n, -3, 3)
		got, err := ValAll(Chebyshev, nil, c, x)
		if err != nil {
			t.Fatal(err)
		}
		for i, xv := range x {
			if w := chebval(c, xv); !sameBits(got[i], w) {
				t.Fatalf("%d coefficients x=%v: ValAll %v, chebval %v", n, xv, got[i], w)
			}
		}
	}
}

// A call with a large enough dst allocates nothing: the reversed coefficients
// live in a pooled buffer.
func TestBatchEvaluatorsAllocateNothing(t *testing.T) {
	if raceOn {
		t.Skip("sync.Pool drops items at random under the race detector")
	}
	x := unif(newRNG(24), 1000, -1, 1)
	dst := make([]float64, len(x))
	for _, n := range []int{6, 200} {
		p := unif(newRNG(25), n, -1, 1)
		if a := testing.AllocsPerRun(20, func() { PolyvalAll(dst, p, x) }); a != 0 {
			t.Errorf("PolyvalAll, %d coefficients: %v allocations per call", n, a)
		}
	}
	c := unif(newRNG(26), 21, -1, 1)
	if a := testing.AllocsPerRun(20, func() { _, _ = ValAll(Chebyshev, dst, c, x) }); a != 0 {
		t.Errorf("ValAll Chebyshev: %v allocations per call", a)
	}
}
