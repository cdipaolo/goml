package linear

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

var flatX [][]float64
var flatY []float64

var increasingX [][]float64
var increasingY []float64

var threeDLineX [][]float64
var threeDLineY []float64

func init() {
	// the line y=3
	flatX = [][]float64{}
	flatY = []float64{}
	for i := -10; i < 10; i++ {
        for j := -10; j < 10; j++ {
            for k := -10; k < 10; k++ {
                flatX = append(flatX, []float64{float64(i), float64(j), float64(k)})
                flatY = append(flatY, 3.0)
            }
        }
	}

	// the line y=x
	increasingX = [][]float64{}
	increasingY = []float64{}
	for i := -10; i < 10; i++ {
		increasingX = append(increasingX, []float64{float64(i)})
		increasingY = append(increasingY, float64(i))
	}

	threeDLineX = [][]float64{}
	threeDLineY = []float64{}
	// the line z = 10 + (x/10) + (y/5)
	for i := -10; i < 10; i++ {
		for j := -10; j < 10; j++ {
			threeDLineX = append(threeDLineX, []float64{float64(i), float64(j)})
			threeDLineY = append(threeDLineY, 10+float64(i)/10+float64(j)/5)
		}
	}
}

// test y=3
func TestFlatLineShouldPass1(t *testing.T) {
	var err error

	model, err := NewLeastSquares(.000001, 800, flatX, flatY)
	assert.Nil(t, err, "You should be able to create a new Least Squares model!")

	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64

    for i := -20; i < 20; i += 10 {
        for j := -20; j < 20; j += 10 {
            for k := -20; k < 20; k += 10 {
                guess, err = model.Predict([]float64{float64(i), float64(j), float64(k)})
                assert.Len(t, guess, 1, "Length of a LeastSquares model output from the hypothesis should always be a 1 dimensional vector. Never multidimensional.")
                assert.InDelta(t, 3, guess[0], 1e-2, "Guess should be really close to 3 (within 1e-2) for y=3")
                assert.Nil(t, err, "Prediction error should be nil")
            }
        }
	}
}

// test y=x
func TestInclinedLineShouldPass1(t *testing.T) {
	var err error

	model, err := NewLeastSquares(.0001, 500, increasingX, increasingY)
	assert.Nil(t, err, "You should be able to create a new Least Squares model!")
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64

	for i := -20; i < 20; i++ {
		guess, err = model.Predict([]float64{float64(i)})
		assert.Len(t, guess, 1, "Length of a LeastSquares model output from the hypothesis should always be a 1 dimensional vector. Never multidimensional.")
		assert.InDelta(t, i, guess[0], 1e-2, "Guess should be really close to input (within 1e-2) for y=x")
		assert.Nil(t, err, "Prediction error should be nil")
	}
}

// test z = 10 + (x/10) + (y/5)
func TestThreeDimensionalLineShouldPass1(t *testing.T) {
	var err error

	model, err := NewLeastSquares(.0001, 1000, threeDLineX, threeDLineY)
	assert.Nil(t, err, "You should be able to create a new Least Squares model!")
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64

	for i := 0; i < 10; i++ {
		for j := 0; j < 10; j++ {
			guess, err = model.Predict([]float64{float64(i), float64(j)})
			assert.Len(t, guess, 1, "Length of a LeastSquares model output from the hypothesis should always be a 1 dimensional vector. Never multidimensional.")
			assert.InDelta(t, 10.0+float64(i)/10+float64(j)/5, guess[0], 1e-2, "Guess should be really close to i+x (within 1e-2) for line z=10 + (x+y)/10")
			assert.Nil(t, err, "Prediction error should be nil")
		}
	}
}
