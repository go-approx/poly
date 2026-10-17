// Package oracle drives NumPy/SciPy as a reference implementation from Go.
//
// The only Python is the small dispatcher in Program: it resolves a dotted
// name (np.polyfit, np.polynomial.chebyshev.chebval, ...) or a method (".deriv"
// on the result of an earlier call), calls it with the decoded JSON
// arguments and encodes the result as JSON. Everything else - the requests,
// the comparison, the fixtures - is Go.
//
// Transport is injected so that the package is fully testable without
// Python; the real "uv run" transport lives behind the "oracle" build tag
// (uv.go).
package oracle

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
)

// Versions pinned for the oracle.
const (
	NumPyVersion = "2.5.4"
	SciPyVersion = "1.18.1"
)

// Program is the Python dispatcher given to "python -c". It reads one JSON
// request (a list of calls) per line on stdin and answers one JSON line.
const Program = `import sys,json,time,warnings,numpy as np,scipy as sp
from fractions import Fraction as F
from functools import reduce
warnings.simplefilter('ignore');np.seterr(all='ignore')
E={}
R={'np':np,'sp':sp,'ex':lambda p,x:np.array([float(reduce(lambda r,c:r*F(v)+F(c),p,F(0))) for v in x])}
def d(v):
 if isinstance(v,dict):
  if 'a' in v:return np.array(v['a'],dtype=float)
  if 'f' in v:return float(v['f'])
  if 'z' in v:return np.array([complex(float(a),float(b)) for a,b in v['z']])
  if 'ref' in v:return E[v['ref']]
  if 'fn' in v:return c(v)
 if isinstance(v,list):return [d(i) for i in v]
 return v
def c(v):
 a=[d(i) for i in v.get('args',[])];k={u:d(w) for u,w in v.get('kw',{}).items()};f=v['fn']
 if f[0]=='.':return getattr(a[0],f[1:])(*a[1:],**k)
 o=R[f.split('.')[0]]
 for p in f.split('.')[1:]:o=getattr(o,p)
 return o(*a,**k) if callable(o) else o
def fl(x):return x if x==x and abs(x)!=float('inf') else str(x)
def e(v):
 if isinstance(v,(tuple,list)):return {'t':[e(i) for i in v]}
 if hasattr(v,'coef'):return {'t':[e(v.coef),e(v.domain),e(v.window)]}
 if isinstance(v,str):return {'s':v}
 a=np.asarray(v);r={'sh':list(a.shape),'re':[fl(i) for i in a.real.ravel().tolist()]}
 if a.dtype.kind=='c':r['im']=[fl(i) for i in a.imag.ravel().tolist()]
 return r
for line in sys.stdin:
 o=[]
 for r in json.loads(line):
  try:
   if 'set' in r:E[r['set']]=d(r['v']);o.append({})
   else:
    n=r.get('rep',1);t=time.perf_counter()
    for _ in range(n):x=c(r)
    t=(time.perf_counter()-t)/n
    o.append({'o':e(x),'time':t})
  except Exception as x:o.append({'err':type(x).__name__+': '+str(x)})
 print(json.dumps(o),flush=True)
`

// Transport sends one request line and returns the one-line answer.
type Transport interface {
	Do(request []byte) ([]byte, error)
}

// Client talks to the oracle through a Transport.
type Client struct{ T Transport }

// Floats is a []float64 whose JSON form keeps NaN and +-Inf as the strings
// "nan", "inf", "-inf" and every finite value as a shortest round-trip number.
type Floats []float64

// MarshalJSON implements json.Marshaler.
func (f Floats) MarshalJSON() ([]byte, error) {
	b := make([]byte, 0, len(f)*12+2)
	b = append(b, '[')
	for i, v := range f {
		if i > 0 {
			b = append(b, ',')
		}
		b = appendFloat(b, v)
	}
	return append(b, ']'), nil
}

func appendFloat(b []byte, v float64) []byte {
	switch {
	case math.IsNaN(v):
		return append(b, `"nan"`...)
	case math.IsInf(v, 1):
		return append(b, `"inf"`...)
	case math.IsInf(v, -1):
		return append(b, `"-inf"`...)
	}
	return strconv.AppendFloat(b, v, 'g', -1, 64)
}

// Float is a float64 with the same JSON convention as Floats.
type Float float64

// MarshalJSON implements json.Marshaler.
func (f Float) MarshalJSON() ([]byte, error) { return appendFloat(nil, float64(f)), nil }

// UnmarshalJSON implements json.Unmarshaler.
func (f *Float) UnmarshalJSON(b []byte) error {
	if len(b) > 0 && b[0] == '"' {
		var s string
		if err := json.Unmarshal(b, &s); err != nil {
			return err
		}
		switch s {
		case "nan":
			*f = Float(math.NaN())
		case "inf":
			*f = Float(math.Inf(1))
		case "-inf":
			*f = Float(math.Inf(-1))
		default:
			return fmt.Errorf("oracle: bad float token %q", s)
		}
		return nil
	}
	v, err := strconv.ParseFloat(string(b), 64)
	*f = Float(v)
	return err
}

// UnmarshalJSON implements json.Unmarshaler.
func (f *Floats) UnmarshalJSON(b []byte) error {
	var raw []Float
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	*f = make(Floats, len(raw))
	for i, v := range raw {
		(*f)[i] = float64(v)
	}
	return nil
}

// Arr encodes a float64 array argument.
func Arr(v []float64) any { return map[string]any{"a": Floats(v)} }

// F encodes a float64 scalar argument (NaN/Inf allowed).
func F(v float64) any {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return map[string]any{"f": Float(v)}
	}
	return v
}

// Cplx encodes a complex128 array argument.
func Cplx(v []complex128) any {
	z := make([][2]Float, len(v))
	for i, c := range v {
		z[i] = [2]Float{Float(real(c)), Float(imag(c))}
	}
	return map[string]any{"z": z}
}

// Ref refers to a value stored earlier with Set.
func Ref(name string) any { return map[string]any{"ref": name} }

// Call is a function call; Fn is a dotted name rooted at np, sp or ex, or
// ".method" to call a method on the first argument. Args may be nested Calls.
type Call struct {
	Fn   string
	Args []any
	Kw   map[string]any
	// Rep, when > 1, makes the oracle run the call Rep times and report the
	// mean time per call (Result.Time).
	Rep int
}

// MarshalJSON implements json.Marshaler.
func (c Call) MarshalJSON() ([]byte, error) {
	m := map[string]any{"fn": c.Fn, "args": c.Args}
	if c.Args == nil {
		m["args"] = []any{}
	}
	if len(c.Kw) > 0 {
		m["kw"] = c.Kw
	}
	if c.Rep > 1 {
		m["rep"] = c.Rep
	}
	return json.Marshal(m)
}

// Set stores the value V in the oracle process under Name.
type Set struct {
	Name string
	V    any
}

// MarshalJSON implements json.Marshaler.
func (s Set) MarshalJSON() ([]byte, error) {
	return json.Marshal(map[string]any{"set": s.Name, "v": s.V})
}

// Value is a decoded result: a (possibly 0-d) array, a tuple, or a string.
type Value struct {
	Shape   []int
	Re, Im  []float64
	Complex bool
	Tuple   []Value
	Str     string
	IsStr   bool
	IsTuple bool
}

// Result is the answer to one item of a batch.
type Result struct {
	Err  string  // non-empty when NumPy raised: "ValueError: ..."
	Val  Value   // the result
	Time float64 // seconds spent in the call itself (time.perf_counter)
}

type rawValue struct {
	Sh []int      `json:"sh"`
	Re []Float    `json:"re"`
	Im []Float    `json:"im"`
	T  []rawValue `json:"t"`
	S  *string    `json:"s"`
}

type rawResult struct {
	Err  string    `json:"err"`
	O    *rawValue `json:"o"`
	Time float64   `json:"time"`
}

func (r rawValue) value() Value {
	switch {
	case r.S != nil:
		return Value{Str: *r.S, IsStr: true}
	case r.T != nil:
		v := Value{IsTuple: true, Tuple: make([]Value, len(r.T))}
		for i, e := range r.T {
			v.Tuple[i] = e.value()
		}
		return v
	}
	v := Value{Shape: r.Sh, Re: make([]float64, len(r.Re)), Complex: r.Im != nil}
	for i, e := range r.Re {
		v.Re[i] = float64(e)
	}
	if v.Complex {
		v.Im = make([]float64, len(r.Im))
		for i, e := range r.Im {
			v.Im[i] = float64(e)
		}
	}
	return v
}

// Complexes returns the value as a complex slice (imaginary parts 0 for a
// real array).
func (v Value) Complexes() []complex128 {
	out := make([]complex128, len(v.Re))
	for i, re := range v.Re {
		out[i] = complex(re, 0)
		if v.Complex {
			out[i] = complex(re, v.Im[i])
		}
	}
	return out
}

// Batch sends the items (Call or Set values) as one request and returns one
// Result per item.
func (c Client) Batch(items ...any) ([]Result, error) {
	req, err := json.Marshal(items)
	if err != nil {
		return nil, err
	}
	ans, err := c.T.Do(req)
	if err != nil {
		return nil, err
	}
	var raw []rawResult
	if err := json.Unmarshal(bytes.TrimSpace(ans), &raw); err != nil {
		return nil, fmt.Errorf("oracle: bad answer: %w", err)
	}
	if len(raw) != len(items) {
		return nil, fmt.Errorf("oracle: %d answers for %d requests", len(raw), len(items))
	}
	out := make([]Result, len(raw))
	for i, r := range raw {
		out[i] = Result{Err: r.Err, Time: r.Time}
		if r.O != nil {
			out[i].Val = r.O.value()
		}
	}
	return out, nil
}

// Do runs a single call.
func (c Client) Do(call Call) (Result, error) {
	r, err := c.Batch(call)
	if err != nil {
		return Result{}, err
	}
	return r[0], nil
}

// ErrVersion is returned by HealthCheck when the oracle's libraries differ
// from the pinned versions.
var ErrVersion = errors.New("oracle: version mismatch")

// HealthCheck verifies the oracle answers and runs numpy NumPyVersion and
// scipy SciPyVersion.
func (c Client) HealthCheck() error {
	r, err := c.Batch(Call{Fn: "np.__version__"}, Call{Fn: "sp.__version__"})
	if err != nil {
		return err
	}
	for i, want := range []string{NumPyVersion, SciPyVersion} {
		if r[i].Err != "" {
			return fmt.Errorf("oracle: %s", r[i].Err)
		}
		if got := r[i].Val.Str; got != want {
			return fmt.Errorf("%w: got %q, want %q", ErrVersion, got, want)
		}
	}
	return nil
}
