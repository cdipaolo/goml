package linear

import (
	"fmt"
	"os"
	"testing"

	"github.com/cdipaolo/goml/base"

	"github.com/stretchr/testify/assert"
)

var fdx [][]float64
var fdy []float64

var bdx [][]float64
var bdy []float64

var tdx [][]float64
var tdy []float64

// tests basically make a bunch of Softmaxs where
// when the input is above the Softmax the resultant
// output is 1.0, else 0.0
func init() {

	// create the /tmp/.goml/ dir for persistance testing
	// if it doesn't already exist!
	err := os.MkdirAll("/tmp/.goml", os.ModePerm)
	if err != nil {
		panic(fmt.Sprintf("You should be able to create the directory for goml model persistance testing.\n\tError returned: %v\n", err.Error()))
	}

	fdx = [][]float64{}
	fdy = []float64{}
	for i := -60; i < 40; i += 5 {
		for j := -60; j < 40; j += 5 {
			for k := -60; k < 40; k += 5 {
				fdx = append(fdx, []float64{float64(i), float64(j), float64(k)})
				if 10*i+j/20+k > 0 && -1*i-10*j-k > 0 {
					fdy = append(fdy, 2.0)
				} else if 10*i+j/20+k > 0 && -1*i-10*j-k < 0 {
					fdy = append(fdy, 1.0)
				} else {
					fdy = append(fdy, 0.0)
				}
			}
		}
	}

	bdx = [][]float64{}
	bdy = []float64{}
	for i := -10; i < 25; i++ {
		bdx = append(bdx, []float64{float64(i)})
		if i > 15 {
			bdy = append(bdy, 4.0)
		} else if i > 10 {
			bdy = append(bdy, 3.0)
		} else if i > 5 && i < 10 {
			bdy = append(bdy, 2.0)
		} else if i > 0 && i < 5 {
			bdy = append(bdy, 1.0)
		} else {
			bdy = append(bdy, 0.0)
		}
	}

	tdx = [][]float64{}
	tdy = []float64{}
	for i := -10; i < 10; i++ {
		for j := -10; j < 10; j++ {
			tdx = append(tdx, []float64{float64(i), float64(j)})

			if i+j > 5 {
				tdy = append(tdy, 1.0)
			} else {
				tdy = append(tdy, 0.0)
			}
		}
	}
}

// maxI returns the index of the maximum value
// of a slice of float64's
func maxI(array []float64) int {
	var i int
	for j := range array {
		if array[j] > array[i] {
			i = j
		}
	}
	return i
}

/*
// test ( 10*i + j/20 + k ) > 0
func TestFourDimensionalSoftmaxShouldPass1(t *testing.T) {
	var err error

	model := NewSoftmax(base.BatchGA, 1e-3, 0, 3, 800, fdx, fdy)

	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64

	for i := -20; i < 21; i += 10 {
		for j := -20; j < 21; j += 10 {
			for k := -20; k < 21; k += 10 {
				guess, err = model.Predict([]float64{float64(i), float64(j), float64(k)})
				assert.Len(t, guess, 3, "Length of Softmax hypothesis output should be 3")

				hp := maxI(guess)

				if 10*i+j/20+k > 0 && -1*i-10*j-k > 0 {
					assert.Equal(t, 2, hp, "Guess should be 2")
				} else if 10*i+j/20+k > 0 && -1*i-10*j-k < 0 {
					assert.Equal(t, 1, hp, "Guess should be 1")
				} else {
					assert.Equal(t, 0, hp, "Guess should be 0")
				}

				for _, val := range guess {
					assert.True(t, val < 1.1, "Probability (%v) should always be less than 1", val)
					assert.True(t, val > -0.1, "Probability (%v) should always be greater than 0", val)
				}

				assert.Nil(t, err, "Prediction error should be nil")
			}
		}
	}
}

// same as above but with StochasticGA
func TestFourDimensionalSoftmaxShouldPass2(t *testing.T) {
	var err error

	model := NewSoftmax(base.StochasticGA, .000001, 0, 3, 800, fdx, fdy)

	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64

	for i := -20; i < 20; i += 10 {
		for j := -20; j < 20; j += 10 {
			for k := -20; k < 20; k += 10 {
				guess, err = model.Predict([]float64{float64(i), float64(j), float64(k)})
				assert.Len(t, guess, 3, "Length of Softmax hypothesis output should be 3")

				hp := maxI(guess)

				if 10*i+j/20+k > 0 && -1*i-10*j-k > 0 {
					assert.Equal(t, 2, hp, "Guess should be 2")
				} else if 10*i+j/20+k > 0 && -1*i-10*j-k < 0 {
					assert.Equal(t, 1, hp, "Guess should be 1")
				} else {
					assert.Equal(t, 0, hp, "Guess should be 0")
				}

				for _, val := range guess {
					assert.True(t, val < 1.1, "Probability should always be less than 1")
					assert.True(t, val > -0.1, "Probability should always be greater than 0")
				}

				assert.Nil(t, err, "Prediction error should be nil")
			}
		}
	}
}

// test ( 10*i + j/20 + k ) > 0 but don't have enough iterations
func TestFourDimensionalSoftmaxShouldFail1(t *testing.T) {
	var err error

	model := NewSoftmax(base.BatchGA, .000001, 0, 3, 1, fdx, fdy)

	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64
	var faliures int

	for i := -20; i < 20; i += 10 {
		for j := -20; j < 20; j += 10 {
			for k := -20; k < 20; k += 10 {
				guess, err = model.Predict([]float64{float64(i), float64(j), float64(k)})
				assert.Len(t, guess, 3, "Length of Softmax hypothesis output should be 3")

				hp := maxI(guess)

				if 10*i+j/20+k > 0 && -1*i-10*j-k > 0 && hp != 2 {
					faliures++
				} else if 10*i+j/20+k > 0 && -1*i-10*j-k < 0 && hp != 1 {
					faliures++
				} else if hp != 0 {
					faliures++
				}

				for _, val := range guess {
					assert.True(t, val < 1.1, "Probability should always be less than 1")
					assert.True(t, val > -0.1, "Probability should always be greater than 0")
				}

				assert.Nil(t, err, "Prediction error should be nil")
			}
		}
	}

	assert.True(t, faliures > 40, "There should be more faliures than half of the training set")
}

// same as above but with StochasticGA
func TestFourDimensionalSoftmaxShouldFail2(t *testing.T) {
	var err error

	model := NewSoftmax(base.StochasticGA, .000001, 0, 3, 1, fdx, fdy)

	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64
	var faliures int

	for i := -20; i < 20; i += 10 {
		for j := -20; j < 20; j += 10 {
			for k := -20; k < 20; k += 10 {
				guess, err = model.Predict([]float64{float64(i), float64(j), float64(k)})
				assert.Len(t, guess, 3, "Length of Softmax hypothesis output should be 3")

				hp := maxI(guess)

				if 10*i+j/20+k > 0 && -1*i-10*j-k > 0 && hp != 2 {
					faliures++
				} else if 10*i+j/20+k > 0 && -1*i-10*j-k < 0 && hp != 1 {
					faliures++
				} else if hp != 0 {
					faliures++
				}

				for _, val := range guess {
					assert.True(t, val < 1.1, "Probability should always be less than 1")
					assert.True(t, val > -0.1, "Probability should always be greater than 0")
				}

				assert.Nil(t, err, "Prediction error should be nil")
			}
		}
	}

	assert.True(t, faliures > 40, "There should be more faliures than half of the training set")
}

// test ( 10*i + j/20 + k ) > 0 but include an invalid data set
func TestFourDimensionalSoftmaxShouldFail3(t *testing.T) {
	var err error

	model := NewSoftmax(base.BatchGA, 1, 0, 3, 800, [][]float64{}, fdy)

	err = model.Learn()
	assert.NotNil(t, err, "Learning error should not be nil")

	model = NewSoftmax(base.BatchGA, 1, 0, 3, 800, [][]float64{[]float64{}, []float64{}}, fdy)

	err = model.Learn()
	assert.NotNil(t, err, "Learning error should not be nil")

	model = NewSoftmax(base.BatchGA, 1, 0, 3, 800, nil, fdy)

	err = model.Learn()
	assert.NotNil(t, err, "Learning error should not be nil")
}

// same as above but with StochasticGA
func TestFourDimensionalSoftmaxShouldFail4(t *testing.T) {
	var err error

	model := NewSoftmax(base.StochasticGA, 1, 0, 3, 800, [][]float64{}, fdy)

	err = model.Learn()
	assert.NotNil(t, err, "Learning error should not be nil")

	model = NewSoftmax(base.BatchGA, 1, 0, 3, 800, [][]float64{[]float64{}, []float64{}}, fdy)

	err = model.Learn()
	assert.NotNil(t, err, "Learning error should not be nil")

	model = NewSoftmax(base.BatchGA, 1, 0, 3, 800, nil, fdy)

	err = model.Learn()
	assert.NotNil(t, err, "Learning error should not be nil")
}

// test ( 10*i + j/20 + k ) > 0 but include an invalid data set
func TestFourDimensionalSoftmaxShouldFail5(t *testing.T) {
	var err error

	model := NewSoftmax(base.BatchGA, 1, 0, 3, 800, fdx, []float64{})

	err = model.Learn()
	assert.NotNil(t, err, "Learning error should not be nil")

	model = NewSoftmax(base.BatchGA, 1, 0, 3, 800, fdx, nil)

	err = model.Learn()
	assert.NotNil(t, err, "Learning error should not be nil")
}

// same as above but with StochasticGA
func TestFourDimensionalSoftmaxShouldFail6(t *testing.T) {
	var err error

	model := NewSoftmax(base.StochasticGA, 1, 0, 3, 800, fdx, []float64{})

	err = model.Learn()
	assert.NotNil(t, err, "Learning error should not be nil")

	model = NewSoftmax(base.BatchGA, 1, 0, 3, 800, fdx, nil)

	err = model.Learn()
	assert.NotNil(t, err, "Learning error should not be nil")
}

// invalid method
func TestFourDimensionalSoftmaxShouldFail7(t *testing.T) {
	var err error

	model := NewSoftmax(base.OptimizationMethod("Not A Method!!!"), 1, 0, 3, 800, fdx, fdy)

	err = model.Learn()
	assert.NotNil(t, err, "Learning error should not be nil")
}
*/
func TestTwoDimensionalSoftmaxShouldPass1(t *testing.T) {
	var err error

	model := NewSoftmax(base.BatchGA, .0001, 0, 5, 500, bdx, bdy)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64

	for i := -20; i < 20; i += 3 {
		guess, err = model.Predict([]float64{float64(i)})

		highestP := maxI(guess)

		if i > 15 {
			assert.Equal(t, 4, highestP, "Best guess should be 4.0")
		} else if i > 10 {
			assert.Equal(t, 3, highestP, "Best guess should be 3.0")
		} else if i > 5 {
			assert.Equal(t, 2, highestP, "Best guess should be 2.0")
		} else if i > 0 {
			assert.Equal(t, 1, highestP, "Best guess should be 1.0")
		} else {
			assert.Equal(t, 0, highestP, "Best guess should be 0.0")
		}

		for _, val := range guess {
			assert.True(t, val < 1.1, "Probability should always be less than 1")
			assert.True(t, val > -0.1, "Probability should always be greater than 0")
		}

		assert.Len(t, guess, 5, "Length of a Softmax model output from hypothesis should reflect the input dimensions")
		assert.Nil(t, err, "Prediction error should be nil")
	}
}

// same as above but with StochasticGA
func TestTwoDimensionalSoftmaxShouldPass2(t *testing.T) {
	var err error

	model := NewSoftmax(base.StochasticGA, .0001, 0, 5, 1500, bdx, bdy)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64

	for i := -20; i < 20; i++ {
		guess, err = model.Predict([]float64{float64(i)})

		highestP := maxI(guess)

		if i > 15 {
			assert.Equal(t, 4, highestP, "Best guess should be 4.0")
		} else if i > 10 {
			assert.Equal(t, 3, highestP, "Best guess should be 3.0")
		} else if i > 5 {
			assert.Equal(t, 2, highestP, "Best guess should be 2.0")
		} else if i > 0 {
			assert.Equal(t, 1, highestP, "Best guess should be 1.0")
		} else {
			assert.Equal(t, 0, highestP, "Best guess should be 0.0")
		}

		for _, val := range guess {
			assert.True(t, val < 1.1, "Probability should always be less than 1")
			assert.True(t, val > -0.1, "Probability should always be greater than 0")
		}

		assert.Len(t, guess, 5, "Length of a Softmax model output from hypothesis should reflect the input dimensions")
		assert.Nil(t, err, "Prediction error should be nil")
	}
}

/*
// regularization term too large
func TestTwoDimensionalSoftmaxShouldFail1(t *testing.T) {
	var err error

	model := NewSoftmax(base.BatchGA, 1e-4, 1e3, 5, 500, bdx, bdy)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64
	var faliures int

	for i := -200; i < 200; i += 15 {
		guess, err = model.Predict([]float64{float64(i)})

		hp := maxI(guess)

		if i > 15 && hp != 4 {
			faliures++
		} else if i > 10 && hp != 3 {
			faliures++
		} else if i > 5 && hp != 2 {
			faliures++
		} else if i > 0 && hp != 1 {
			faliures++
		} else if hp != 0 {
			faliures++
		}

		assert.Len(t, guess, 5, "Length of a Softmax model output from hypothesis should reflect the input dimensions")
		assert.Nil(t, err, "Prediction error should be nil")
	}

	assert.True(t, faliures > 10, "There should be a strong majority of faliures of the training set")
}

// same as above but with StochasticGA
func TestTwoDimensionalSoftmaxShouldFail2(t *testing.T) {
	var err error

	model := NewSoftmax(base.StochasticGA, 1e-4, 1e3, 5, 300, bdx, bdy)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64
	var faliures int

	for i := -200; i < 200; i += 15 {
		guess, err = model.Predict([]float64{float64(i)})
		if i > 0 && guess[0] < 0.5 {
			faliures++
		} else if i < 0 && guess[0] > 0.5 {
			faliures++
		}

		assert.Len(t, guess, 5, "Length of a Softmax model output from hypothesis should reflect the input dimensions")
		assert.Nil(t, err, "Prediction error should be nil")
	}

	assert.True(t, faliures > 10, "There should be a strong majority of faliures of the training set")
}

// test i+j > 5
func TestThreeDimensionalSoftmaxShouldPass1(t *testing.T) {
	var err error

	model := NewSoftmax(base.BatchGA, .0001, 0, 2, 3000, tdx, tdy)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64

	for i := -20; i < 20; i += 3 {
		for j := -20; j < 20; j += 3 {
			guess, err = model.Predict([]float64{float64(i), float64(j)})

			if i+j > 5 {
				assert.True(t, guess[1] > 0.5, "Guess[1] should be more likely to be 1. %v", guess[0])
				assert.True(t, guess[1] < 1.01, "Guess[1] should not exceed 1 ever. %v", guess[0])

				assert.True(t, guess[0] < 0.5, "Guess[0] should be more likely to be 0. %v", guess[0])
				assert.True(t, guess[0] > -0.01, "Guess[0] should not be below 0 ever. %v", guess[0])
			} else {
				assert.True(t, guess[0] > 0.5, "Guess[0] should be more likely to be 1. %v", guess[0])
				assert.True(t, guess[0] < 1.01, "Guess[0] should not exceed 1 ever. %v", guess[0])

				assert.True(t, guess[1] < 0.5, "Guess[1] should be more likely to be 0. %v", guess[1])
				assert.True(t, guess[1] > -0.01, "Guess[1] should not be below 0 ever. %v", guess[0])
			}

			assert.Len(t, guess, 2, "Length of a Softmax model output from hypothesis should reflect the input dimensions")
			assert.Nil(t, err, "Prediction error should be nil")
		}
	}
}

// same as above but with StochasticGA
func TestThreeDimensionalSoftmaxShouldPass2(t *testing.T) {
	var err error

	model := NewSoftmax(base.StochasticGA, .0001, 0, 2, 3000, tdx, tdy)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64

	for i := -20; i < 20; i += 3 {
		for j := -20; j < 20; j += 3 {
			guess, err = model.Predict([]float64{float64(i), float64(j)})

			if i+j > 5 {
				assert.True(t, guess[1] > 0.5, "Guess[1] should be more likely to be 1. %v", guess[0])
				assert.True(t, guess[1] < 1.01, "Guess[1] should not exceed 1 ever. %v", guess[0])

				assert.True(t, guess[0] < 0.5, "Guess[0] should be more likely to be 0. %v", guess[0])
				assert.True(t, guess[0] > -0.01, "Guess[0] should not be below 0 ever. %v", guess[0])
			} else {
				assert.True(t, guess[0] > 0.5, "Guess[0] should be more likely to be 1. %v", guess[0])
				assert.True(t, guess[0] < 1.01, "Guess[0] should not exceed 1 ever. %v", guess[0])

				assert.True(t, guess[1] < 0.5, "Guess[1] should be more likely to be 0. %v", guess[1])
				assert.True(t, guess[1] > -0.01, "Guess[1] should not be below 0 ever. %v", guess[0])
			}

			assert.Len(t, guess, 2, "Length of a Softmax model output from hypothesis should reflect the input dimensions")
			assert.Nil(t, err, "Prediction error should be nil")
		}
	}
}

// test persisting y=x to file
func TestPersistSoftmaxShouldPass1(t *testing.T) {
	var err error

	model := NewSoftmax(base.BatchGA, .0001, 0, 5, 800, bdx, bdy)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64

	for i := -20; i < 20; i += 3 {
		guess, err = model.Predict([]float64{float64(i)})

		highestP := maxI(guess)

		if i > 15 {
			assert.Equal(t, 4, highestP, "Best guess should be 4.0")
		} else if i > 10 {
			assert.Equal(t, 3, highestP, "Best guess should be 3.0")
		} else if i > 5 {
			assert.Equal(t, 2, highestP, "Best guess should be 2.0")
		} else if i > 0 {
			assert.Equal(t, 1, highestP, "Best guess should be 1.0")
		} else {
			assert.Equal(t, 0, highestP, "Best guess should be 0.0")
		}

		for _, val := range guess {
			assert.True(t, val < 1.1, "Probability should always be less than 1")
			assert.True(t, val > -0.1, "Probability should always be greater than 0")
		}

		assert.Len(t, guess, 5, "Length of a Softmax model output from hypothesis should reflect the input dimensions")
		assert.Nil(t, err, "Prediction error should be nil")
	}

	// not that we know it works, try persisting to file,
	// then resetting the parameter vector theta, then
	// restoring it and testing that predictions are correct
	// again.

	err = model.PersistToFile("/tmp/.goml/Logistic.json")
	assert.Nil(t, err, "Persistance error should be nil")

	length := len(model.Parameters[0])
	model.Parameters = make([][]float64, len(model.Parameters))
	for i := range model.Parameters {
		model.Parameters[i] = make([]float64, length)
	}

	// make sure it WONT work now that we reset theta
	//
	// the result of Theta transpose * X should always
	// be 0.2 because the probablility is neutral
	for i := -20; i < 20; i++ {
		guess, err = model.Predict([]float64{float64(i)})
		assert.Len(t, guess, 5, "Length of a Softmax model output from hypothesis should reflect the input dimensions")
		assert.Nil(t, err, "Prediction error should be nil")

		for j := range guess {
			assert.Equal(t, 0.2, guess[j], "Guess[%v] should all be equal", j)
		}
	}

	err = model.RestoreFromFile("/tmp/.goml/Logistic.json")
	assert.Nil(t, err, "Persistance error should be nil")

	for i := -20; i < 20; i += 3 {
		guess, err = model.Predict([]float64{float64(i)})

		highestP := maxI(guess)

		if i > 15 {
			assert.Equal(t, 4, highestP, "Best guess should be 4.0")
		} else if i > 10 {
			assert.Equal(t, 3, highestP, "Best guess should be 3.0")
		} else if i > 5 {
			assert.Equal(t, 2, highestP, "Best guess should be 2.0")
		} else if i > 0 {
			assert.Equal(t, 1, highestP, "Best guess should be 1.0")
		} else {
			assert.Equal(t, 0, highestP, "Best guess should be 0.0")
		}

		for _, val := range guess {
			assert.True(t, val < 1.1, "Probability should always be less than 1")
			assert.True(t, val > -0.1, "Probability should always be greater than 0")
		}

		assert.Len(t, guess, 5, "Length of a Softmax model output from hypothesis should reflect the input dimensions")
		assert.Nil(t, err, "Prediction error should be nil")
	}
}*/
