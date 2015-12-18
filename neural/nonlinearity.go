package neural

import "math"

type NonLinearity uint8

// Constants contain enums for
// representing neuron activation
// nonlinearities
const (
	ReLu NonLinearity = iota
	Sigmoid
	Tanh
	Identity
)

func (n NonLinearity) F(x float64) float64 {
	switch n {
	case ReLu:
		return ReLuF(x)
	case Sigmoid:
		return SigmoidF(x)
	case Tanh:
		return TanhF(x)
	case Identity:
		return IdentityF(x)
	}
}

// String converts a NonLinearity
// into a string representation.
func (n NonLinearity) String() string {
	switch n {
	case ReLu:
		return "ReLu"
	case Sigmoid:
		return "Sigmoid"
	case Tanh:
		return "Tanh"
	case Identity:
		return "Identity"
	}
}

func EncodeNonLinearitySlice(n []NonLinearity) []string {
	s := make([]string, len(n))
	for i := range n {
		s[i] = n[i].String()
	}

	return s
}

/* ReLu */

// ReLuF (x) = max( 0, x )
func ReLuF(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// ReLuDF (x) = 1 if x > 0; 0 otherwise
func ReLuDF(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

/* Sigmoid */

// SigmoidF (x) = 1 / ( 1 + exp( x ) )
func SigmoidF(x float64) float64 {
	return 1 / (1 + math.Exp(-1*x))
}

// SigmoidDF (x) = F(x)(1 - F(x))
func SigmoidDF(x float64) float64 {
	f := 1 / (1 + math.Exp(-1*x))
	return f * (1 - f)
}

/* Tanh */

// TanhF (x) = tanh(x)
func TanhF(x float64) float64 {
	return math.Tanh(x)
}

// TanhDF (x) = 1 - tanh^2(x)
func TanhDF(x float64) float64 {
	tanh := math.Tanh(x)
	return 1 - tanh*tanh
}

/* Identity */

// IdentityF (x) = x
func IdentityF(x float64) float64 {
	return x
}

// IdentityDF (x) = 1
func IdentityDF(x float64) float64 {
	return 1
}
