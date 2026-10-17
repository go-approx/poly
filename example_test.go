package poly_test

import (
	"fmt"
	"math"
	"math/big"

	"github.com/go-approx/poly"
)

func ExamplePolyval() {
	// 2x^2 - 3x + 1, coefficients highest degree first.
	fmt.Println(poly.Polyval([]float64{2, -3, 1}, 4))
	// Output: 21
}

func ExamplePolyvalComp() {
	// (x-1)^8 at x = 1.01 is 1e-16, but Horner works with terms of size 1:
	// it loses every digit to cancellation, the compensated scheme keeps
	// them. The exact value comes from math/big.
	p := []float64{1, -8, 28, -56, 70, -56, 28, -8, 1}
	x := 1.01
	exact := new(big.Float).SetPrec(2000)
	bx := new(big.Float).SetPrec(2000).SetFloat64(x)
	for _, c := range p {
		exact.Mul(exact, bx).Add(exact, big.NewFloat(c))
	}
	ex, _ := exact.Float64()
	rel := func(v float64) float64 { return math.Abs(v-ex) / ex }
	fmt.Println("Horner is off by more than 100%:", rel(poly.Polyval(p, x)) > 1)
	fmt.Println("compensated is within 1e-14:   ", rel(poly.PolyvalComp(p, x)) < 1e-14)
	// Output:
	// Horner is off by more than 100%: true
	// compensated is within 1e-14:    true
}

func ExamplePolyfit() {
	x := []float64{0, 1, 2, 3, 4}
	y := []float64{1, 3, 7, 13, 21} // x^2 + x + 1
	fit, err := poly.Polyfit(x, y, 2, nil)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%.6f rank=%d\n", fit.Coef, fit.Rank)
	// Output: [1.000000 1.000000 1.000000] rank=3
}

func ExamplePolyfit_covariance() {
	x := []float64{0, 1, 2, 3, 4, 5}
	y := []float64{0.1, 1.1, 1.9, 3.2, 3.9, 5.1}
	fit, err := poly.Polyfit(x, y, 1, &poly.FitOptions{Cov: poly.CovScaled})
	if err != nil {
		panic(err)
	}
	fmt.Printf("slope %.4f +- %.4f\n", fit.Coef[0], sqrt(fit.Cov[0]))
	// Output: slope 0.9914 +- 0.0325
}

func sqrt(v float64) float64 {
	z := v
	for i := 0; i < 60; i++ {
		z = (z + v/z) / 2
	}
	return z
}

func ExampleRoots() {
	roots, err := poly.Roots([]float64{1, -3, 2, 0}) // x^3 - 3x^2 + 2x
	if err != nil {
		panic(err)
	}
	for _, r := range roots {
		fmt.Printf("%.4f ", real(r))
	}
	fmt.Println()
	// Output: 2.0000 1.0000 0.0000
}

func ExampleDeconv() {
	q, r, _ := poly.Deconv([]float64{1, 3, 3, 1}, []float64{1, 1}) // (x+1)^3 / (x+1)
	fmt.Println(q, r)
	// Output: [1 2 1] [0]
}

func ExampleFitSeries() {
	// Fit y = x^2 on [0, 2] with Chebyshev polynomials: the domain [0, 2]
	// is mapped to the window [-1, 1].
	x := []float64{0, 0.5, 1, 1.5, 2}
	y := []float64{0, 0.25, 1, 2.25, 4}
	s, _, err := poly.FitSeries(poly.Chebyshev, x, y, 2, nil)
	if err != nil {
		panic(err)
	}
	v, _ := s.Val(1.5)
	fmt.Printf("domain %v coef %.4f value %.4f\n", s.Domain, s.Coef, v)
	// Output: domain [0 2] coef [1.5000 2.0000 0.5000] value 2.2500
}

func ExampleSeries_Roots() {
	// 1 - 2x^2 in the Legendre basis is not what it looks like: P2 = (3x^2-1)/2.
	s, _ := poly.NewSeries(poly.Legendre, []float64{0, 0, 1})
	roots, _ := s.Roots()
	fmt.Printf("%.4f %.4f\n", real(roots[0]), real(roots[1]))
	// Output: -0.5774 0.5774
}

func ExampleVander() {
	v, _ := poly.Vander(poly.Chebyshev, []float64{0.5, 1}, 3) // rows: T0..T3
	fmt.Println(v)
	// Output: [1 0.5 -0.5 -1 1 1 1 1]
}
