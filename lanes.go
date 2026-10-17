package poly

// Four-lane kernels: the recurrences of val are latency-bound (each step
// depends on the previous one), so four independent evaluation points are
// interleaved to keep the floating-point units busy. Every lane performs
// exactly the operations of the scalar functions in basis.go, in the same
// order, so the results are bit-identical to them (and to NumPy).

// polyval4 is Polyval at four points.
func polyval4(p []float64, x0, x1, x2, x3 float64) (y0, y1, y2, y3 float64) {
	for _, c := range p {
		y0 = float64(y0*x0) + c
		y1 = float64(y1*x1) + c
		y2 = float64(y2*x2) + c
		y3 = float64(y3*x3) + c
	}
	return
}

// chebval4 needs len(c) >= 3.
func chebval4(c []float64, x0, x1, x2, x3 float64) (r0, r1, r2, r3 float64) {
	n := len(c)
	a0, a1, a2, a3 := 2*x0, 2*x1, 2*x2, 2*x3
	u0, u1, u2, u3 := c[n-2], c[n-2], c[n-2], c[n-2]
	v0, v1, v2, v3 := c[n-1], c[n-1], c[n-1], c[n-1]
	for i := 3; i <= n; i++ {
		ci := c[n-i]
		t0, t1, t2, t3 := u0, u1, u2, u3
		u0, u1, u2, u3 = ci-v0, ci-v1, ci-v2, ci-v3
		v0 = t0 + float64(v0*a0)
		v1 = t1 + float64(v1*a1)
		v2 = t2 + float64(v2*a2)
		v3 = t3 + float64(v3*a3)
	}
	return u0 + float64(v0*x0), u1 + float64(v1*x1), u2 + float64(v2*x2), u3 + float64(v3*x3)
}

// legval4 needs len(c) >= 3.
func legval4(c []float64, x0, x1, x2, x3 float64) (r0, r1, r2, r3 float64) {
	n := len(c)
	nd := n
	u0, u1, u2, u3 := c[n-2], c[n-2], c[n-2], c[n-2]
	v0, v1, v2, v3 := c[n-1], c[n-1], c[n-1], c[n-1]
	for i := 3; i <= n; i++ {
		ci := c[n-i]
		nd--
		f1 := float64(nd-1) / float64(nd)
		f2 := float64(2*nd-1) / float64(nd)
		t0, t1, t2, t3 := u0, u1, u2, u3
		u0 = ci - float64(v0*f1)
		u1 = ci - float64(v1*f1)
		u2 = ci - float64(v2*f1)
		u3 = ci - float64(v3*f1)
		v0 = t0 + float64(float64(v0*x0)*f2)
		v1 = t1 + float64(float64(v1*x1)*f2)
		v2 = t2 + float64(float64(v2*x2)*f2)
		v3 = t3 + float64(float64(v3*x3)*f2)
	}
	return u0 + float64(v0*x0), u1 + float64(v1*x1), u2 + float64(v2*x2), u3 + float64(v3*x3)
}

// hermval4 needs len(c) >= 3.
func hermval4(c []float64, x0, x1, x2, x3 float64) (r0, r1, r2, r3 float64) {
	n := len(c)
	nd := n
	a0, a1, a2, a3 := x0*2, x1*2, x2*2, x3*2
	u0, u1, u2, u3 := c[n-2], c[n-2], c[n-2], c[n-2]
	v0, v1, v2, v3 := c[n-1], c[n-1], c[n-1], c[n-1]
	for i := 3; i <= n; i++ {
		ci := c[n-i]
		nd--
		f := float64(2 * (nd - 1))
		t0, t1, t2, t3 := u0, u1, u2, u3
		u0 = ci - float64(v0*f)
		u1 = ci - float64(v1*f)
		u2 = ci - float64(v2*f)
		u3 = ci - float64(v3*f)
		v0 = t0 + float64(v0*a0)
		v1 = t1 + float64(v1*a1)
		v2 = t2 + float64(v2*a2)
		v3 = t3 + float64(v3*a3)
	}
	return u0 + float64(v0*a0), u1 + float64(v1*a1), u2 + float64(v2*a2), u3 + float64(v3*a3)
}

// hermeval4 needs len(c) >= 3.
func hermeval4(c []float64, x0, x1, x2, x3 float64) (r0, r1, r2, r3 float64) {
	n := len(c)
	nd := n
	u0, u1, u2, u3 := c[n-2], c[n-2], c[n-2], c[n-2]
	v0, v1, v2, v3 := c[n-1], c[n-1], c[n-1], c[n-1]
	for i := 3; i <= n; i++ {
		ci := c[n-i]
		nd--
		f := float64(nd - 1)
		t0, t1, t2, t3 := u0, u1, u2, u3
		u0 = ci - float64(v0*f)
		u1 = ci - float64(v1*f)
		u2 = ci - float64(v2*f)
		u3 = ci - float64(v3*f)
		v0 = t0 + float64(v0*x0)
		v1 = t1 + float64(v1*x1)
		v2 = t2 + float64(v2*x2)
		v3 = t3 + float64(v3*x3)
	}
	return u0 + float64(v0*x0), u1 + float64(v1*x1), u2 + float64(v2*x2), u3 + float64(v3*x3)
}

// lagval4 needs len(c) >= 3.
func lagval4(c []float64, x0, x1, x2, x3 float64) (r0, r1, r2, r3 float64) {
	n := len(c)
	nd := n
	u0, u1, u2, u3 := c[n-2], c[n-2], c[n-2], c[n-2]
	v0, v1, v2, v3 := c[n-1], c[n-1], c[n-1], c[n-1]
	for i := 3; i <= n; i++ {
		ci := c[n-i]
		nd--
		f, g, d := float64(nd-1), float64(2*nd-1), float64(nd)
		t0, t1, t2, t3 := u0, u1, u2, u3
		u0 = ci - float64(v0*f)/d
		u1 = ci - float64(v1*f)/d
		u2 = ci - float64(v2*f)/d
		u3 = ci - float64(v3*f)/d
		v0 = t0 + float64(v0*(g-x0))/d
		v1 = t1 + float64(v1*(g-x1))/d
		v2 = t2 + float64(v2*(g-x2))/d
		v3 = t3 + float64(v3*(g-x3))/d
	}
	return u0 + float64(v0*(1-x0)), u1 + float64(v1*(1-x1)), u2 + float64(v2*(1-x2)), u3 + float64(v3*(1-x3))
}
