package neural

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

/* Test Forward Pass Validity */

func TestNetworkOutputsShouldPass1(t *testing.T) {
	model := NewFeedForwardNet(3, 0.1, 10, []uint64{2, 3, 1}, []NonLinearity{ReLu, ReLu, Sigmoid}, nil, nil)

	model.Weights = [][][]float64{
		[][]float64{ //input layer
			[]float64{0, -3, 2, 10},
			[]float64{0, 3, 5, 2},
		},
		[][]float64{ //layer 1
			[]float64{-10, -2, 6},
			[]float64{3, 3, 1},
			[]float64{-5, 12, 2},
		},
		[][]float64{ // output layer
			[]float64{-20, 0.001, 1, 0.25},
		},
	}

	yHat, err := model.Predict([]float64{1, 2, -1})
	assert.Nil(t, err, "Error should be nil")
	assert.Len(t, yHat, 1, "Length of output should be 1")
	assert.InDelta(t, 0.1552505252, yHat[0], 1e-6, "Hypothesis output should be 0.1552505252")

	yHat, err = model.Predict([]float64{-1, 3, 2})
	assert.Nil(t, err, "Error should be nil")
	assert.Len(t, yHat, 1, "Length of output should be 1")
	assert.InDelta(t, 1, yHat[0], 1e-6, "Hypothesis output should be 1")
}

/* Test gradient computations! */
const gradEpsilon = 1e-3

type coord struct {
	i int
	j int
	l int
}

var gradTests = []struct {
	inputs     uint64
	dims       []uint64
	transforms []NonLinearity
	x          [][]float64
	y          [][]float64
	coords     []coord
}{
	{2, []uint64{1}, []NonLinearity{Sigmoid}, [][]float64{
		{10, 15},
		{-10, -15},
		{0, 0},
		{-100, 6},
	}, [][]float64{
		{1},
		{1},
		{0},
		{0},
	}, []coord{
		{1, 0, 0},
		{0, 0, 0},
		{2, 0, 0},
	}},
	{2, []uint64{1, 1, 1}, []NonLinearity{ReLu, Tanh, Sigmoid}, [][]float64{
		{10, 15},
		{-10, -15},
		{0, 0},
		{-100, 6},
	}, [][]float64{
		{1},
		{1},
		{0},
		{0},
	}, []coord{
		{1, 0, 0},
		{0, 0, 0},
		{2, 0, 0},
		{1, 0, 1},
		{0, 0, 1},
		{1, 0, 1},
		{1, 0, 2},
		{0, 0, 2},
	}},
	{2, []uint64{10, 40, 2, 3}, []NonLinearity{Tanh, ReLu, ReLu, Sigmoid}, [][]float64{
		{10, 15},
		{-10, -15},
		{0, 0},
		{-100, 6},
	}, [][]float64{
		{1, 0, 1},
		{1, 1, 0},
		{0, 0, 1},
		{0, 1, 1},
	}, []coord{
		{1, 0, 0},
		{0, 0, 0},
		{2, 0, 0},
		{7, 23, 1},
		{2, 39, 1},
		{2, 1, 2},
	}},
}

/*
var gradTests = []struct {
	inputs     uint64
	dims       []uint64
	transforms []NonLinearity
	x          [][]float64
	y          [][]float64
	coords     []coord
}{...}
*/

func TestGradientComputationShouldPass(t *testing.T) {
	for i := 0; i < 10; i++ {
		rand.Seed(int64(i))
		for _, T := range gradTests {
			model := NewFeedForwardNet(T.inputs, 0.1, 10, T.dims, T.transforms, nil, nil)
			for j := range T.coords {
				for i := range T.x {
					assert.InDelta(t, model.computeNumericalDerivative(T.coords[j].i, T.coords[j].j, T.coords[j].l, T.x[i], T.y[i], 1e-5),
						model.computeDerivative(T.coords[j].i, T.coords[j].j, T.coords[j].l, T.x[i], T.y[i]),
						gradEpsilon, "Derivatives should be correct")
				}
			}
		}
	}
}
