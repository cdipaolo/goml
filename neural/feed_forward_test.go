package neural

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

func init() {
	rand.Seed(42)
}

/* Test gradient computations! */
func TestGradientShouldPass1(t *testing.T) {
	model := NewFeedForwardNet(2, 0.1, 10, []uint64{1}, []NonLinearity{Sigmoid}, nil, nil)
	assert.InDelta(t, model.ComputeNumericalDerivative(0, 0, 0, []float64{10, 15}, []float64{1}, 0.001),
		model.ComputeDerivative(0, 0, 0, []float64{10, 15}, []float64{1}),
		0.001, "Derivatives should be correct")

	assert.InDelta(t, model.ComputeNumericalDerivative(2, 0, 0, []float64{-3, 15}, []float64{1}, 0.001),
		model.ComputeDerivative(2, 0, 0, []float64{-3, 15}, []float64{1}),
		0.001, "Derivatives should be correct")
}

func TestGradientShouldPass2(t *testing.T) {
	model := NewFeedForwardNet(2, 0.1, 10, []uint64{10, 40, 2, 1}, []NonLinearity{Tanh, ReLu, ReLu, Sigmoid}, nil, nil)
	assert.InDelta(t, model.ComputeNumericalDerivative(0, 0, 0, []float64{10, 15}, []float64{1}, 1e-4),
		model.ComputeDerivative(0, 0, 0, []float64{10, 15}, []float64{1}),
		0.00001, "Derivatives should be correct")

	assert.InDelta(t, model.ComputeNumericalDerivative(7, 23, 1, []float64{10, 15}, []float64{1}, 1e-4),
		model.ComputeDerivative(7, 23, 1, []float64{10, 15}, []float64{1}),
		0.00001, "Derivatives should be correct")

	assert.InDelta(t, model.ComputeNumericalDerivative(2, 39, 1, []float64{-2, -32}, []float64{0}, 1e-4),
		model.ComputeDerivative(2, 39, 1, []float64{-2, -32}, []float64{0}),
		0.00001, "Derivatives should be correct")

	assert.InDelta(t, model.ComputeNumericalDerivative(2, 1, 2, []float64{0, 15}, []float64{0}, 1e-4),
		model.ComputeDerivative(2, 1, 2, []float64{0, 15}, []float64{0}),
		0.00001, "Derivatives should be correct")
}
