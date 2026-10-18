package oracle

import (
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"
)

// fake is a Transport that records the request and replays a canned answer.
type fake struct {
	req []byte
	ans string
	err error
}

func (f *fake) Do(req []byte) ([]byte, error) {
	f.req = req
	return []byte(f.ans + "\n"), f.err
}

func TestFloatsRoundTrip(t *testing.T) {
	in := Floats{1, -0.5, 1e-300, 1.7976931348623157e308, math.NaN(), math.Inf(1), math.Inf(-1), math.Copysign(0, -1)}
	b, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"nan","inf","-inf"`) {
		t.Fatalf("non-finite tokens missing: %s", b)
	}
	var out Floats
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	for i := range in {
		if math.Float64bits(in[i]) != math.Float64bits(out[i]) && !(math.IsNaN(in[i]) && math.IsNaN(out[i])) {
			t.Errorf("element %d: %v != %v", i, in[i], out[i])
		}
	}
	var f Floats
	if err := json.Unmarshal([]byte(`["bogus"]`), &f); err == nil {
		t.Error("bad token accepted")
	}
	if err := json.Unmarshal([]byte(`{}`), &f); err == nil {
		t.Error("object accepted as float array")
	}
}

func TestFloat(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want float64
		bad  bool
	}{{`1.5`, 1.5, false}, {`"nan"`, math.NaN(), false}, {`"inf"`, math.Inf(1), false}, {`"-inf"`, math.Inf(-1), false}, {`"x"`, 0, true}, {`"unterminated`, 0, true}, {`1.2.3`, 0, true}} {
		var f Float
		err := f.UnmarshalJSON([]byte(tc.in))
		if (err != nil) != tc.bad {
			t.Errorf("%s: err=%v", tc.in, err)
		}
		if !tc.bad && !(float64(f) == tc.want || math.IsNaN(float64(f)) && math.IsNaN(tc.want)) {
			t.Errorf("%s: %v", tc.in, float64(f))
		}
	}
	for _, v := range []float64{2, math.NaN(), math.Inf(1), math.Inf(-1)} {
		b, _ := Float(v).MarshalJSON()
		var f Float
		if err := f.UnmarshalJSON(b); err != nil || differ(float64(f), v) {
			t.Errorf("%v -> %s -> %v (%v)", v, b, float64(f), err)
		}
	}
}

func differ(a, b float64) bool { // true when different
	return !(a == b || math.IsNaN(a) && math.IsNaN(b))
}

func TestArguments(t *testing.T) {
	enc := func(v any) string {
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	if got := enc(Arr([]float64{1, math.NaN()})); got != `{"a":[1,"nan"]}` {
		t.Error(got)
	}
	if got := enc(F(1.5)); got != `1.5` {
		t.Error(got)
	}
	if got := enc(F(math.Inf(-1))); got != `{"f":"-inf"}` {
		t.Error(got)
	}
	if got := enc(F(math.NaN())); got != `{"f":"nan"}` {
		t.Error(got)
	}
	if got := enc(Cplx([]complex128{complex(1, -2), complex(math.NaN(), math.Inf(1))})); got != `{"z":[[1,-2],["nan","inf"]]}` {
		t.Error(got)
	}
	if got := enc(Ref("x")); got != `{"ref":"x"}` {
		t.Error(got)
	}
	if got := enc(Call{Fn: "np.roots"}); got != `{"args":[],"fn":"np.roots"}` {
		t.Error(got)
	}
	if got := enc(Call{Fn: "np.polyfit", Args: []any{Ref("a"), 2}, Kw: map[string]any{"full": true}}); got != `{"args":[{"ref":"a"},2],"fn":"np.polyfit","kw":{"full":true}}` {
		t.Error(got)
	}
	if got := enc(Call{Fn: "f", Rep: 5}); got != `{"args":[],"fn":"f","rep":5}` {
		t.Error(got)
	}
	if got := enc(Set{Name: "a", V: Arr(nil)}); got != `{"set":"a","v":{"a":[]}}` {
		t.Error(got)
	}
}

const answer = `[{},
 {"o":{"sh":[2],"re":[1,"nan"]},"time":0.5},
 {"o":{"sh":[2],"re":[1,2],"im":["inf",0]}},
 {"o":{"t":[{"sh":[],"re":[3]},{"s":"txt"}]}},
 {"err":"ValueError: boom"}]`

func TestBatch(t *testing.T) {
	f := &fake{ans: strings.ReplaceAll(answer, "\n", "")}
	c := Client{T: f}
	items := []any{Set{Name: "a", V: 1}, Call{Fn: "f"}, Call{Fn: "g"}, Call{Fn: "h"}, Call{Fn: "i"}}
	res, err := c.Batch(items...)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(string(f.req), `[{"set":"a"`) {
		t.Errorf("request: %s", f.req)
	}
	if len(res) != 5 || res[4].Err != "ValueError: boom" {
		t.Fatalf("%+v", res)
	}
	if v := res[1].Val; len(v.Re) != 2 || !math.IsNaN(v.Re[1]) || v.Complex || res[1].Time != 0.5 {
		t.Errorf("%+v", res[1])
	}
	if v := res[2].Val; !v.Complex || !math.IsInf(v.Im[0], 1) {
		t.Errorf("%+v", v)
	} else if z := v.Complexes(); z[0] != complex(1, math.Inf(1)) || z[1] != complex(2, 0) {
		t.Errorf("%v", z)
	}
	if v := res[3].Val; !v.IsTuple || len(v.Tuple) != 2 || v.Tuple[0].Re[0] != 3 || !v.Tuple[1].IsStr || v.Tuple[1].Str != "txt" {
		t.Errorf("%+v", v)
	}
	if z := (Value{Re: []float64{4}}).Complexes(); z[0] != 4 {
		t.Errorf("%v", z)
	}
}

func TestBatchErrors(t *testing.T) {
	if _, err := (Client{T: &fake{ans: `[]`}}).Batch(func() {}); err == nil {
		t.Error("unmarshalable item accepted")
	}
	boom := errors.New("pipe closed")
	if _, err := (Client{T: &fake{err: boom}}).Batch(Call{Fn: "f"}); !errors.Is(err, boom) {
		t.Errorf("transport error lost: %v", err)
	}
	if _, err := (Client{T: &fake{ans: `not json`}}).Batch(Call{Fn: "f"}); err == nil {
		t.Error("garbage answer accepted")
	}
	if _, err := (Client{T: &fake{ans: `[{},{}]`}}).Batch(Call{Fn: "f"}); err == nil {
		t.Error("wrong answer count accepted")
	}
	if _, err := (Client{T: &fake{err: boom}}).Do(Call{Fn: "f"}); !errors.Is(err, boom) {
		t.Error("Do lost the transport error")
	}
	r, err := (Client{T: &fake{ans: `[{"o":{"s":"ok"}}]`}}).Do(Call{Fn: "f"})
	if err != nil || r.Val.Str != "ok" {
		t.Errorf("%+v %v", r, err)
	}
}

func TestHealthCheck(t *testing.T) {
	ok := `[{"o":{"s":"` + NumPyVersion + `"}},{"o":{"s":"` + SciPyVersion + `"}}]`
	if err := (Client{T: &fake{ans: ok}}).HealthCheck(); err != nil {
		t.Error(err)
	}
	bad := `[{"o":{"s":"1.0"}},{"o":{"s":"` + SciPyVersion + `"}}]`
	if err := (Client{T: &fake{ans: bad}}).HealthCheck(); !errors.Is(err, ErrVersion) {
		t.Errorf("%v", err)
	}
	if err := (Client{T: &fake{ans: `[{"err":"ImportError: x"},{}]`}}).HealthCheck(); err == nil || !strings.Contains(err.Error(), "ImportError") {
		t.Errorf("%v", err)
	}
	if err := (Client{T: &fake{err: errors.New("down")}}).HealthCheck(); err == nil {
		t.Error("transport failure ignored")
	}
}

func TestProgramShape(t *testing.T) {
	// The dispatcher must stay tiny: it is the only Python in the module.
	if n := strings.Count(Program, "\n"); n > 45 {
		t.Errorf("Program has %d lines", n)
	}
	for _, s := range []string{"numpy as np", "scipy as sp", "json.loads", "perf_counter"} {
		if !strings.Contains(Program, s) {
			t.Errorf("Program lacks %q", s)
		}
	}
}
