package poly

import (
	"fmt"
	"testing"
)

func benchData(n int) (x, y []float64) {
	r := newRNG(5)
	return unif(r, n, -1, 1), unif(r, n, -1, 1)
}

func BenchmarkPolyval(b *testing.B) {
	for _, n := range []int{1000, 100000} {
		x, _ := benchData(n)
		p := []float64{1, -2, 3, -4, 5, -6}
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			dst := make([]float64, n)
			for i := 0; i < b.N; i++ {
				PolyvalAll(dst, p, x)
			}
		})
	}
}

func BenchmarkChebval(b *testing.B) {
	x, _ := benchData(100000)
	c := make([]float64, 21)
	for i := range c {
		c[i] = 1 / float64(i+1)
	}
	dst := make([]float64, len(x))
	for i := 0; i < b.N; i++ {
		_, _ = ValAll(Chebyshev, dst, c, x)
	}
}

func BenchmarkPolyfit(b *testing.B) {
	x, y := benchData(100000)
	for _, deg := range []int{3, 10} {
		b.Run(fmt.Sprint("deg", deg), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = Polyfit(x, y, deg, nil)
			}
		})
	}
}

func BenchmarkRoots(b *testing.B) {
	for _, deg := range []int{10, 100} {
		p := make([]float64, deg+1)
		r := newRNG(6)
		for i := range p {
			p[i] = 2*r.Float64() - 1
		}
		b.Run(fmt.Sprint("deg", deg), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = Roots(p)
			}
		})
	}
}
