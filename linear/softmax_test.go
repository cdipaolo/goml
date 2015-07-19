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
	for i := -2.0; i < 2.0; i += 0.2 {
		for j := -2.0; j < 2.0; j += 0.2 {
			for k := -2.0; k < 2.0; k += 0.2 {

				fdx = append(fdx, []float64{i, j, k})
				if i/2+j+2*k > 0 && -1*i-j-0.5*k > 0 {
					fdy = append(fdy, 2.0)
				} else if i/2+j+2*k > 0 && -1*i-j-0.5*k < 0 {
					fdy = append(fdy, 1.0)
				} else {
					fdy = append(fdy, 0.0)
				}
			}
		}
	}

	bdx = [][]float64{}
	bdy = []float64{}
	for i := -2.0; i < 2.0; i += 0.00073 {
		bdx = append(bdx, []float64{i})
		if i > 0.75 {
			bdy = append(bdy, 4.0)
		} else if i > 0.12 {
			bdy = append(bdy, 3.0)
		} else if i > -0.25 {
			bdy = append(bdy, 2.0)
		} else if i > -0.65 {
			bdy = append(bdy, 1.0)
		} else {
			bdy = append(bdy, 0.0)
		}
	}

	tdx = [][]float64{}
	tdy = []float64{}
	for i := -2.0; i < 2.0; i += 0.15 {
		for j := -2.0; j < 2.0; j += 0.15 {
			tdx = append(tdx, []float64{float64(i), float64(j)})

			if -2*i+j/2-0.5 > 0 && -1*i-j < 0 {
				tdy = append(tdy, 2.0)
			} else if -2*i+j/2-0.5 > 0 && -1*i-j > 0 {
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

// test ( 10*i + j/20 + k ) > 0
func TestFourDimensionalSoftmaxShouldPass1(t *testing.T) {
	var err error

	model := NewSoftmax(base.BatchGA, 1e-5, 0, 3, 10, fdx, fdy)

	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64
	var incorrect int
	var count int

	for i := -1.0; i < 1.0; i += 0.30 {
		for j := -1.0; j < 1.0; j += 0.30 {
			for k := -1.0; k < 1.0; k += 0.30 {
				guess, err = model.Predict([]float64{i, j, k})
				assert.Len(t, guess, 3, "Length of Softmax hypothesis output should be 3")

				prediction := maxI(guess)

				if i/2+j+2*k > 0 && -1*i-j-0.5*k > 0 {
					if prediction != 2 {
						incorrect++
					}

				} else if i/2+j+2*k > 0 && -1*i-j-0.5*k < 0 {
					if prediction != 1 {
						incorrect++
					}

				} else {
					if prediction != 0 {
						incorrect++
					}

				}

				for _, val := range guess {
					assert.True(t, val < 1.1, "Probability (%v) should always be less than 1", val)
					assert.True(t, val > -0.1, "Probability (%v) should always be greater than 0", val)
				}

				assert.Nil(t, err, "Prediction error should be nil")
				count++
			}
		}
	}

	fmt.Printf("Predictions: %v\n\tIncorrect: %v\n\tAccuracy Rate: %v percent\n", count, incorrect, 100*(1.0-float64(incorrect)/float64(count)))
	assert.True(t, float64(incorrect)/float64(count) < 0.14, "Accuracy should be greater than 86%")
}

// same as above but with StochasticGA
func TestFourDimensionalSoftmaxShouldPass2(t *testing.T) {
	var err error

	model := NewSoftmax(base.StochasticGA, 1e-5, 0, 3, 10, fdx, fdy)

	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64
	var incorrect int
	var count int

	for i := -1.0; i < 1.0; i += 0.3 {
		for j := -1.0; j < 1.0; j += 0.3 {
			for k := -1.0; k < 1.0; k += 0.3 {
				guess, err = model.Predict([]float64{i, j, k})
				assert.Len(t, guess, 3, "Length of Softmax hypothesis output should be 3")

				prediction := maxI(guess)

				if i/2+j+2*k > 0 && -1*i-j-0.5*k > 0 {
					if prediction != 2 {
						incorrect++
					}

				} else if i/2+j+2*k > 0 && -1*i-j-0.5*k < 0 {
					if prediction != 1 {
						incorrect++
					}

				} else {
					if prediction != 0 {
						incorrect++
					}

				}

				for _, val := range guess {
					assert.True(t, val < 1.1, "Probability (%v) should always be less than 1", val)
					assert.True(t, val > -0.1, "Probability (%v) should always be greater than 0", val)
				}

				assert.Nil(t, err, "Prediction error should be nil")
				count++
			}
		}
	}

	fmt.Printf("Predictions: %v\n\tIncorrect: %v\n\tAccuracy Rate: %v percent\n", count, incorrect, 100*(1.0-float64(incorrect)/float64(count)))
	assert.True(t, float64(incorrect)/float64(count) < 0.14, "Accuracy should be greater than 86%")
}

// test ( 10*i + j/20 + k ) > 0 but don't have enough iterations
func TestFourDimensionalSoftmaxShouldFail1(t *testing.T) {
	var err error

	model := NewSoftmax(base.BatchGA, 1e-7, 0, 3, 1, fdx, fdy)

	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64
	var incorrect int
	var count int

	for i := -1.0; i < 1.0; i += 0.3 {
		for j := -1.0; j < 1.0; j += 0.3 {
			for k := -1.0; k < 1.0; k += 0.3 {
				guess, err = model.Predict([]float64{i, j, k})
				assert.Len(t, guess, 3, "Length of Softmax hypothesis output should be 3")

				prediction := maxI(guess)

				if i/2+j+2*k > 0 && -1*i-j-0.5*k > 0 {
					if prediction != 2 {
						incorrect++
					}

				} else if i/2+j+2*k > 0 && -1*i-j-0.5*k < 0 {
					if prediction != 1 {
						incorrect++
					}

				} else {
					if prediction != 0 {
						incorrect++
					}

				}

				for _, val := range guess {
					assert.True(t, val < 1.1, "Probability (%v) should always be less than 1", val)
					assert.True(t, val > -0.1, "Probability (%v) should always be greater than 0", val)
				}

				assert.Nil(t, err, "Prediction error should be nil")
				count++
			}
		}
	}

	fmt.Printf("Predictions: %v\n\tIncorrect: %v\n\tAccuracy Rate: %v percent\n", count, incorrect, 100*(1.0-float64(incorrect)/float64(count)))
	assert.True(t, float64(incorrect)/float64(count) > 0.3, "Accuracy should be bad (error rate > 0.3)")
}

// same as above but with StochasticGA
func TestFourDimensionalSoftmaxShouldFail2(t *testing.T) {
	var err error

	model := NewSoftmax(base.StochasticGA, .000001, 0, 3, 1, fdx, fdy)

	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64
	var incorrect int
	var count int

	for i := -1.0; i < 1.0; i += 0.3 {
		for j := -1.0; j < 1.0; j += 0.3 {
			for k := -1.0; k < 1.0; k += 0.3 {
				guess, err = model.Predict([]float64{i, j, k})
				assert.Len(t, guess, 3, "Length of Softmax hypothesis output should be 3")

				prediction := maxI(guess)

				if i/2+j+2*k > 0 && -1*i-j-0.5*k > 0 {
					if prediction != 2 {
						incorrect++
					}

				} else if i/2+j+2*k > 0 && -1*i-j-0.5*k < 0 {
					if prediction != 1 {
						incorrect++
					}

				} else {
					if prediction != 0 {
						incorrect++
					}

				}

				for _, val := range guess {
					assert.True(t, val < 1.1, "Probability (%v) should always be less than 1", val)
					assert.True(t, val > -0.1, "Probability (%v) should always be greater than 0", val)
				}

				assert.Nil(t, err, "Prediction error should be nil")
				count++
			}
		}
	}

	fmt.Printf("Predictions: %v\n\tIncorrect: %v\n\tAccuracy Rate: %v percent\n", count, incorrect, 100*(1.0-float64(incorrect)/float64(count)))
	assert.True(t, float64(incorrect)/float64(count) > 0.3, "Accuracy should be bad (error rate > 0.3)")
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

func TestTwoDimensionalSoftmaxShouldPass1(t *testing.T) {
	var err error

	model := NewSoftmax(base.BatchGA, .0001, 0, 5, 10, bdx, bdy)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64
	var count int
	var incorrect int

	for i := -2.0; i < 2; i += 0.0372345432 {
		guess, err = model.Predict([]float64{float64(i)})

		prediction := maxI(guess)

		if i > 0.75 {
			if prediction != 4 {
				incorrect++
			}
		} else if i > 0.12 {
			if prediction != 3 {
				incorrect++
			}
		} else if i > -0.25 {
			if prediction != 2 {
				incorrect++
			}
		} else if i > -0.65 {
			if prediction != 1 {
				incorrect++
			}
		} else {
			if prediction != 0 {
				incorrect++
			}
		}

		for _, val := range guess {
			assert.True(t, val < 1.1, "Probability should always be less than 1")
			assert.True(t, val > -0.1, "Probability should always be greater than 0")
		}

		assert.Len(t, guess, 5, "Length of a Softmax model output from hypothesis should reflect the input dimensions")
		assert.Nil(t, err, "Prediction error should be nil")

		count++
	}

	fmt.Printf("Predictions: %v\n\tIncorrect: %v\n\tAccuracy Rate: %v percent\n", count, incorrect, 100*(1.0-float64(incorrect)/float64(count)))
	assert.True(t, float64(incorrect)/float64(count) < 0.35, "Accuracy should be greater than 65% (this is a more challenging model)")
}

// same as above but with StochasticGA
func TestTwoDimensionalSoftmaxShouldPass2(t *testing.T) {
	var err error

	model := NewSoftmax(base.StochasticGA, .0001, 0, 5, 10, bdx, bdy)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64
	var count int
	var incorrect int

	for i := -2.0; i < 2; i += 0.0372345432 {
		guess, err = model.Predict([]float64{float64(i)})

		prediction := maxI(guess)

		if i > 0.75 {
			if prediction != 4 {
				incorrect++
			}
		} else if i > 0.12 {
			if prediction != 3 {
				incorrect++
			}
		} else if i > -0.25 {
			if prediction != 2 {
				incorrect++
			}
		} else if i > -0.65 {
			if prediction != 1 {
				incorrect++
			}
		} else {
			if prediction != 0 {
				incorrect++
			}
		}

		for _, val := range guess {
			assert.True(t, val < 1.1, "Probability should always be less than 1")
			assert.True(t, val > -0.1, "Probability should always be greater than 0")
		}

		assert.Len(t, guess, 5, "Length of a Softmax model output from hypothesis should reflect the input dimensions")
		assert.Nil(t, err, "Prediction error should be nil")

		count++
	}

	fmt.Printf("Predictions: %v\n\tIncorrect: %v\n\tAccuracy Rate: %v percent\n", count, incorrect, 100*(1.0-float64(incorrect)/float64(count)))
	assert.True(t, float64(incorrect)/float64(count) < 0.35, "Accuracy should be greater than 65% (this is a more challenging model)")
}

// regularization term too large
func TestTwoDimensionalSoftmaxShouldFail1(t *testing.T) {
	var err error

	model := NewSoftmax(base.BatchGA, 1e-4, 20, 5, 10, bdx, bdy)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64
	var count int
	var incorrect int

	for i := -1.0; i < 1; i += 0.037 {
		guess, err = model.Predict([]float64{float64(i)})

		prediction := maxI(guess)

		if i > 0.75 {
			if prediction != 4 {
				incorrect++
			}
		} else if i > 0.12 {
			if prediction != 3 {
				incorrect++
			}
		} else if i > -0.25 {
			if prediction != 2 {
				incorrect++
			}
		} else if i > -0.65 {
			if prediction != 1 {
				incorrect++
			}
		} else {
			if prediction != 0 {
				incorrect++
			}
		}

		for _, val := range guess {
			assert.True(t, val < 1.1, "Probability should always be less than 1")
			assert.True(t, val > -0.1, "Probability should always be greater than 0")
		}

		assert.Len(t, guess, 5, "Length of a Softmax model output from hypothesis should reflect the input dimensions")
		assert.Nil(t, err, "Prediction error should be nil")

		count++
	}

	fmt.Printf("Predictions: %v\n\tIncorrect: %v\n\tAccuracy Rate: %v percent\n", count, incorrect, 100*(1.0-float64(incorrect)/float64(count)))
	assert.True(t, float64(incorrect)/float64(count) > 0.4, "Accuracy should be worse than 60%")
}

// same as above but with StochasticGA
func TestTwoDimensionalSoftmaxShouldFail2(t *testing.T) {
	var err error

	model := NewSoftmax(base.StochasticGA, 1e-5, 2, 5, 10, bdx, bdy)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64
	var count int
	var incorrect int

	for i := -1.0; i < 1; i += 0.037 {
		guess, err = model.Predict([]float64{float64(i)})

		prediction := maxI(guess)

		if i > 0.75 {
			if prediction != 4 {
				incorrect++
			}
		} else if i > 0.12 {
			if prediction != 3 {
				incorrect++
			}
		} else if i > -0.25 {
			if prediction != 2 {
				incorrect++
			}
		} else if i > -0.65 {
			if prediction != 1 {
				incorrect++
			}
		} else {
			if prediction != 0 {
				incorrect++
			}
		}

		for _, val := range guess {
			assert.True(t, val < 1.1, "Probability (%v) should always be less than 1", val)
			assert.True(t, val > -0.1, "Probability (%v) should always be greater than 0", val)
		}

		assert.Len(t, guess, 5, "Length of a Softmax model output from hypothesis should reflect the input dimensions")
		assert.Nil(t, err, "Prediction error should be nil")

		count++
	}

	fmt.Printf("Predictions: %v\n\tIncorrect: %v\n\tAccuracy Rate: %v percent\n", count, incorrect, 100*(1.0-float64(incorrect)/float64(count)))
	assert.True(t, float64(incorrect)/float64(count) > 0.4, "Accuracy should be worse than 60%")
}

// test i+j > 5
func TestThreeDimensionalSoftmaxShouldPass1(t *testing.T) {
	var err error

	model := NewSoftmax(base.BatchGA, 5e-5, 0, 3, 500, tdx, tdy)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64
	var count int
	var incorrect int

	for i := -1.0; i < 1.0; i += 0.112 {
		for j := -1.0; j < 1.0; j += 0.112 {
			guess, err = model.Predict([]float64{float64(i), float64(j)})

			prediction := maxI(guess)

			if -2*i+j/2-0.5 > 0 && -1*i-j < 0 {
				if prediction != 2 {
					incorrect++
				}

			} else if -2*i+j/2-0.5 > 0 && -1*i-j > 0 {
				if prediction != 1 {
					incorrect++
				}

			} else {
				if prediction != 0 {
					incorrect++
				}

			}

			assert.Len(t, guess, 3, "Length of a Softmax model output from hypothesis should reflect the input dimensions")
			assert.Nil(t, err, "Prediction error should be nil")

			count++
		}
	}

	fmt.Printf("Predictions: %v\n\tIncorrect: %v\n\tAccuracy Rate: %v percent\n", count, incorrect, 100*(1.0-float64(incorrect)/float64(count)))
	assert.True(t, float64(incorrect)/float64(count) < 0.14, "Accuracy should be greater than 86%")
}

// same as above but with StochasticGA
func TestThreeDimensionalSoftmaxShouldPass2(t *testing.T) {
	var err error

	model := NewSoftmax(base.StochasticGA, 5e-5, 0, 3, 500, tdx, tdy)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64
	var count int
	var incorrect int

	for i := -1.0; i < 1.0; i += 0.112 {
		for j := -1.0; j < 1.0; j += 0.112 {
			guess, err = model.Predict([]float64{float64(i), float64(j)})

			prediction := maxI(guess)

			if -2*i+j/2-0.5 > 0 && -1*i-j < 0 {
				if prediction != 2 {
					incorrect++
				}

			} else if -2*i+j/2-0.5 > 0 && -1*i-j > 0 {
				if prediction != 1 {
					incorrect++
				}

			} else {
				if prediction != 0 {
					incorrect++
				}

			}

			assert.Len(t, guess, 3, "Length of a Softmax model output from hypothesis should reflect the input dimensions")
			assert.Nil(t, err, "Prediction error should be nil")

			count++
		}
	}

	fmt.Printf("Predictions: %v\n\tIncorrect: %v\n\tAccuracy Rate: %v percent\n", count, incorrect, 100*(1.0-float64(incorrect)/float64(count)))
	assert.True(t, float64(incorrect)/float64(count) < 0.14, "Accuracy should be greater than 86%")
}

//* Test Online Learning through channels *//

func TestThreeDimensionalSoftmaxOnlineShouldPass2(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.Datapoint, 100)
	errors := make(chan error, 20)

	model := NewSoftmax(base.StochasticGA, 5e-5, 0, 3, 0, nil, nil, 2)

	go model.OnlineLearn(errors, stream, func(theta [][]float64) {})

	// start passing data to our datastream
	//
	// we could have data already in our channel
	// when we instantiated the Perceptron, though
	go func() {
		for iter := 0; iter < 3; iter++ {
			for i := -2.0; i < 2.0; i += 0.15 {
				for j := -2.0; j < 2.0; j += 0.15 {

					if -2*i+j/2-0.5 > 0 && -1*i-j < 0 {
						stream <- base.Datapoint{
							X: []float64{float64(i), float64(j)},
							Y: []float64{2.0},
						}
					} else if -2*i+j/2-0.5 > 0 && -1*i-j > 0 {
						stream <- base.Datapoint{
							X: []float64{float64(i), float64(j)},
							Y: []float64{1.0},
						}
					} else {
						stream <- base.Datapoint{
							X: []float64{float64(i), float64(j)},
							Y: []float64{0.0},
						}
					}
				}
			}
		}

		// close the dataset
		close(stream)
	}()

	for {
		err, more := <-errors

		assert.Nil(t, err, "There should not be any errors!")
		assert.False(t, more, "There should not be any errors!")
		if !more {
			break
		}
	}

	var guess []float64
	var count int
	var incorrect int
	var err error

	for i := -1.0; i < 1.0; i += 0.113 {
		for j := -1.0; j < 1.0; j += 0.113 {
			guess, err = model.Predict([]float64{float64(i), float64(j)})

			prediction := maxI(guess)

			if -2*i+j/2-0.5 > 0 && -1*i-j < 0 {
				if prediction != 2 {
					incorrect++
				}

			} else if -2*i+j/2-0.5 > 0 && -1*i-j > 0 {
				if prediction != 1 {
					incorrect++
				}

			} else {
				if prediction != 0 {
					incorrect++
				}

			}

			assert.Len(t, guess, 3, "Length of a Softmax model output from hypothesis should reflect the input dimensions")
			assert.Nil(t, err, "Prediction error should be nil")

			count++
		}
	}

	fmt.Printf("Predictions: %v\n\tIncorrect: %v\n\tAccuracy Rate: %v percent\n", count, incorrect, 100*(1.0-float64(incorrect)/float64(count)))
	assert.True(t, float64(incorrect)/float64(count) < 0.14, "Accuracy should be greater than 86%")
}

func TestThreeDimensionalSoftmaxOnlineNormalizedShouldPass2(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.Datapoint, 100)
	errors := make(chan error, 20)

	model := NewSoftmax(base.StochasticGA, 5e-5, 0, 3, 0, nil, nil, 2)

	go model.OnlineLearn(errors, stream, func(theta [][]float64) {}, true)

	// start passing data to our datastream
	//
	// we could have data already in our channel
	// when we instantiated the Perceptron, though
	go func() {
		for iter := 0; iter < 3; iter++ {
			for i := -1.25; i < 1.25; i += 0.15 {
				for j := -1.25; j < 1.25; j += 0.15 {

					if -2*i+j/2-0.5 > 0 && -1*i-j < 0 {
						stream <- base.Datapoint{
							X: []float64{float64(i), float64(j)},
							Y: []float64{2.0},
						}
					} else if -2*i+j/2-0.5 > 0 && -1*i-j > 0 {
						stream <- base.Datapoint{
							X: []float64{float64(i), float64(j)},
							Y: []float64{1.0},
						}
					} else {
						stream <- base.Datapoint{
							X: []float64{float64(i), float64(j)},
							Y: []float64{0.0},
						}
					}
				}
			}
		}

		// close the dataset
		close(stream)
	}()

	for {
		err, more := <-errors

		assert.Nil(t, err, "There should not be any errors!")
		assert.False(t, more, "There should not be any errors!")
		if !more {
			break
		}
	}

	var guess []float64
	var count int
	var incorrect int
	var err error

	for i := -1.0; i < 1.0; i += 0.113 {
		for j := -1.0; j < 1.0; j += 0.113 {
			guess, err = model.Predict([]float64{float64(i), float64(j)}, true)

			prediction := maxI(guess)

			if -2*i+j/2-0.5 > 0 && -1*i-j < 0 {
				if prediction != 2 {
					incorrect++
				}

			} else if -2*i+j/2-0.5 > 0 && -1*i-j > 0 {
				if prediction != 1 {
					incorrect++
				}

			} else {
				if prediction != 0 {
					incorrect++
				}

			}

			assert.Len(t, guess, 3, "Length of a Softmax model output from hypothesis should reflect the input dimensions")
			assert.Nil(t, err, "Prediction error should be nil")

			count++
		}
	}

	fmt.Printf("Predictions: %v\n\tIncorrect: %v\n\tAccuracy Rate: %v percent\n", count, incorrect, 100*(1.0-float64(incorrect)/float64(count)))
	assert.True(t, float64(incorrect)/float64(count) < 0.14, "Accuracy should be greater than 86%")
}

//* Test Model Persistance to File *//

// test persisting y=x to file
func TestPersistSoftmaxShouldPass1(t *testing.T) {
	var err error

	model := NewSoftmax(base.BatchGA, 5e-5, 0, 3, 500, tdx, tdy)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64
	var count int
	var incorrect int

	for i := -1.0; i < 1.0; i += 0.112 {
		for j := -1.0; j < 1.0; j += 0.112 {
			guess, err = model.Predict([]float64{float64(i), float64(j)})

			prediction := maxI(guess)

			if -2*i+j/2-0.5 > 0 && -1*i-j < 0 {
				if prediction != 2 {
					incorrect++
				}

			} else if -2*i+j/2-0.5 > 0 && -1*i-j > 0 {
				if prediction != 1 {
					incorrect++
				}

			} else {
				if prediction != 0 {
					incorrect++
				}

			}

			assert.Len(t, guess, 3, "Length of a Softmax model output from hypothesis should reflect the input dimensions")
			assert.Nil(t, err, "Prediction error should be nil")

			count++
		}
	}

	fmt.Printf("Predictions: %v\n\tIncorrect: %v\n\tAccuracy Rate: %v percent\n", count, incorrect, 100*(1.0-float64(incorrect)/float64(count)))
	assert.True(t, float64(incorrect)/float64(count) < 0.14, "Accuracy should be greater than 86%")

	// not that we know it works, try persisting to file,
	// then resetting the parameter vector theta, then
	// restoring it and testing that predictions are correct
	// again.

	err = model.PersistToFile("/tmp/.goml/Softmax.json")
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
		guess, err = model.Predict([]float64{float64(i), 0.2})
		assert.Len(t, guess, 3, "Length of a Softmax model output from hypothesis should reflect the input dimensions")
		assert.Nil(t, err, "Prediction error should be nil")

		for j := range guess {
			assert.InDelta(t, 0.333, guess[j], 1e-2, "Guess[%v] should all be equal", j)
		}
	}

	err = model.RestoreFromFile("/tmp/.goml/Softmax.json")
	assert.Nil(t, err, "Persistance error should be nil")

	for i := -1.0; i < 1.0; i += 0.112 {
		for j := -1.0; j < 1.0; j += 0.112 {
			guess, err = model.Predict([]float64{float64(i), float64(j)})

			prediction := maxI(guess)

			if -2*i+j/2-0.5 > 0 && -1*i-j < 0 {
				if prediction != 2 {
					incorrect++
				}

			} else if -2*i+j/2-0.5 > 0 && -1*i-j > 0 {
				if prediction != 1 {
					incorrect++
				}

			} else {
				if prediction != 0 {
					incorrect++
				}

			}

			assert.Len(t, guess, 3, "Length of a Softmax model output from hypothesis should reflect the input dimensions")
			assert.Nil(t, err, "Prediction error should be nil")

			count++
		}
	}

	fmt.Printf("Predictions: %v\n\tIncorrect: %v\n\tAccuracy Rate: %v percent\n", count, incorrect, 100*(1.0-float64(incorrect)/float64(count)))
	assert.True(t, float64(incorrect)/float64(count) < 0.14, "Accuracy should be greater than 86%")
}
