package linear

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

var flatX *[][]float64
var flatY *[]float64

var increasingX *[][]float64
var increasingY *[]float64

var threeDLineX *[][]float64
var threeDLineY *[]float64

func init() {
	// the line y=0
	flatX = &[][]float64{
		[]float64{0},
		[]float64{1},
		[]float64{2},
		[]float64{3},
		[]float64{4},
	}
	flatY = &[]float64{0, 0, 0, 0, 0}

	// the line y=x
	increasingX = &[][]float64{
		[]float64{0},
		[]float64{1},
		[]float64{2},
		[]float64{3},
		[]float64{4},
	}
	increasingY = &[]float64{0, 1, 2, 3, 4}

	threeDLineX = &[][]float64{}
	threeDLineY = &[]float64{}
	// the line z = x + y
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			*threeDLineX = append(*threeDLineX, []float64{float64(i), float64(j)})
			*threeDLineY = append(*threeDLineY, float64(i+j))
		}
	}
}

// test y=0
func TestFlatLineShouldPass1(t *testing.T) {
	var err error

	model, err := NewLeastSquares(1, flatX, flatY)
	assert.Nil(t, err, "You should be able to create a new Least Squares model!")

	model.Learn()

	var guess float64

	guess, err = model.Predict([]float64{1})
	assert.InEpsilon(t, 0, guess, 1e-2, "Guess should be really close to 0 (within 1e-2) for a flat line")
	assert.Nil(t, err, "Prediction error should be nil")

	guess, err = model.Predict([]float64{0})
	assert.InEpsilon(t, 0, guess, 1e-2, "Guess should be really close to 0 (within 1e-2) for a flat line")
	assert.Nil(t, err, "Prediction error should be nil")

	guess, err = model.Predict([]float64{3})
	assert.InEpsilon(t, 0, guess, 1e-2, "Guess should be really close to 0 (within 1e-2) for a flat line")
	assert.Nil(t, err, "Prediction error should be nil")

	guess, err = model.Predict([]float64{-2})
	assert.InEpsilon(t, 0, guess, 1e-2, "Guess should be really close to 0 (within 1e-2) for a flat line")
	assert.Nil(t, err, "Prediction error should be nil")
}

// test y=x
func TestInclinedLineShouldPass1(t *testing.T) {
	var err error

	model, err := NewLeastSquares(1, increasingX, increasingY)
	assert.Nil(t, err, "You should be able to create a new Least Squares model!")
	model.Learn()

	var guess float64

	guess, err = model.Predict([]float64{1})
	assert.InEpsilon(t, 1, guess, 1e-2, "Guess should be really close to 1 (within 1e-2)")
	assert.Nil(t, err, "Prediction error should be nil")

	guess, err = model.Predict([]float64{0})
	assert.InEpsilon(t, 0, guess, 1e-2, "Guess should be really close to 0 (within 1e-2)")
	assert.Nil(t, err, "Prediction error should be nil")

	guess, err = model.Predict([]float64{3})
	assert.InEpsilon(t, 3, guess, 1e-2, "Guess should be really close to 3 (within 1e-2)")
	assert.Nil(t, err, "Prediction error should be nil")

	guess, err = model.Predict([]float64{-2})
	assert.InEpsilon(t, -2, guess, 1e-2, "Guess should be really close to -2 (within 1e-2)")
	assert.Nil(t, err, "Prediction error should be nil")
}

// test z = x + y
func TestThreeDimensionalLineShouldPass1(t *testing.T) {
	var err error

	model, err := NewLeastSquares(1, threeDLineX, threeDLineY)
	assert.Nil(t, err, "You should be able to create a new Least Squares model!")
	model.Learn()

	var guess float64

	guess, err = model.Predict([]float64{1, 5})
	assert.InEpsilon(t, 6, guess, 1e-2, "Guess should be really close to 6 (within 1e-2)")
	assert.Nil(t, err, "Prediction error should be nil")

	guess, err = model.Predict([]float64{0, 10})
	assert.InEpsilon(t, 10, guess, 1e-2, "Guess should be really close to 10 (within 1e-2)")
	assert.Nil(t, err, "Prediction error should be nil")

	guess, err = model.Predict([]float64{3, 0})
	assert.InEpsilon(t, 3, guess, 1e-2, "Guess should be really close to 3 (within 1e-2)")
	assert.Nil(t, err, "Prediction error should be nil")

	guess, err = model.Predict([]float64{-2, 6})
	assert.InEpsilon(t, 4, guess, 1e-2, "Guess should be really close to 4 (within 1e-2)")
	assert.Nil(t, err, "Prediction error should be nil")
}
