package neural

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// derivative numerically computes the first derivative
func derivative(f func(float64) float64, x, epsilon float64) float64 {
	return (f(x+epsilon) - f(x-epsilon)) / (2 * epsilon)
}

var points = []float64{
	-100, -10, -9, -8, -7, -6, -5, -1.0, -0.75, -0.25, 0.25, 0.75, 1.0, 5, 6, 7, 8, 9, 10, 100,
}

const epsilon = 0.001

func TestReLuShouldPass1(t *testing.T) {
	for _, x := range points {
		assert.InDelta(t, derivative(ReLuF, x, epsilon), ReLuDF(x), epsilon, "Derivative Should Match at %v", x)
	}
}
func TestSigmoidShouldPass1(t *testing.T) {
	for _, x := range points {
		assert.InDelta(t, derivative(SigmoidF, x, epsilon), SigmoidDF(x), epsilon, "Derivative Should Match at %v", x)
	}
}
func TestTanhShouldPass1(t *testing.T) {
	for _, x := range points {
		assert.InDelta(t, derivative(TanhF, x, epsilon), TanhDF(x), epsilon, "Derivative Should Match at %v", x)
	}
}
func TestIdentityShouldPass1(t *testing.T) {
	for _, x := range points {
		assert.InDelta(t, derivative(IdentityF, x, epsilon), IdentityDF(x), epsilon, "Derivative Should Match at %v", x)
	}
}
