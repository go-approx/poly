package poly

import (
	"encoding/json"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/go-approx/poly/internal/oracle"
)

// Case is one oracle fixture: the inputs given to NumPy and what it returned.
// Small inputs are stored; large ones are regenerated from Gens (a fixed-seed
// PCG stream) and only a strided sample of the output is stored.
type Case struct {
	Name   string                   `json:"name"`
	Op     string                   `json:"op"`
	Class  string                   `json:"class,omitempty"` // "ill": compare fitted values, not coefficients
	Arr    map[string]oracle.Floats `json:"arr,omitempty"`
	Num    map[string]oracle.Float  `json:"num,omitempty"`
	Gens   map[string]Gen           `json:"gens,omitempty"`
	Err    string                   `json:"err,omitempty"` // exception NumPy raised
	Out    map[string]oracle.Floats `json:"out,omitempty"`
	Stride int                      `json:"stride,omitempty"`
}

// Gen is a deterministic input generator: N uniform values in [Lo, Hi).
type Gen struct {
	Seed uint64  `json:"seed"`
	N    int     `json:"n"`
	Lo   float64 `json:"lo"`
	Hi   float64 `json:"hi"`
}

func newRNG(seed uint64) *rand.Rand {
	return rand.New(rand.NewPCG(seed, seed*0x9E3779B97F4A7C15+1))
}

// Make returns the generated values.
func (g Gen) Make() []float64 {
	r := newRNG(g.Seed)
	v := make([]float64, g.N)
	for i := range v {
		v[i] = g.Lo + float64((g.Hi-g.Lo)*r.Float64()) // rounded product: no FMA, same bits on every architecture
	}
	return v
}

// In returns the named input array (stored or generated).
func (c Case) In(name string) []float64 {
	if g, ok := c.Gens[name]; ok {
		return g.Make()
	}
	if a, ok := c.Arr[name]; ok {
		return a
	}
	return nil
}

// huge reports whether the case regenerates a million points or more.
func (c Case) huge() bool {
	for _, g := range c.Gens {
		if g.N >= 1000000 {
			return true
		}
	}
	return false
}

func (c Case) has(name string) bool {
	_, a := c.Arr[name]
	_, g := c.Gens[name]
	return a || g
}

// N returns a scalar parameter.
func (c Case) N(name string) float64 { return float64(c.Num[name]) }

func (c Case) hasNum(name string) bool { _, ok := c.Num[name]; return ok }

func loadCases(t testing.TB, file string) []Case {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", file))
	if err != nil {
		t.Fatalf("fixture %s: %v (generate with: GOWORK=off go test -tags oracle -run TestGenerate .)", file, err)
	}
	var cs []Case
	if err := json.Unmarshal(b, &cs); err != nil {
		t.Fatal(err)
	}
	return cs
}

// ordered maps a float to an integer such that adjacent floats differ by 1.
func ordered(f float64) int64 {
	b := int64(math.Float64bits(f))
	if b < 0 {
		return math.MinInt64 - b
	}
	return b
}

// ulpDist is the distance in units in the last place; NaN==NaN counts as 0 and
// NaN vs a number as MaxInt64.
func ulpDist(a, b float64) int64 {
	switch {
	case math.IsNaN(a) && math.IsNaN(b):
		return 0
	case math.IsNaN(a) || math.IsNaN(b):
		return math.MaxInt64
	case a == b:
		return 0
	case math.IsInf(a, 0) || math.IsInf(b, 0):
		return math.MaxInt64
	}
	d := ordered(a) - ordered(b)
	if d < 0 {
		d = -d
	}
	return d
}

func unif(r *rand.Rand, n int, lo, hi float64) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = lo + (hi-lo)*r.Float64()
	}
	return v
}
