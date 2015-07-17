package linear

import (
	"fmt"
	"os"
	"testing"

	"github.com/cdipaolo/goml/base"

	"github.com/stretchr/testify/assert"
)

var fourDX [][]float64
var fourDY []float64

var twoDX [][]float64
var twoDY []float64

var threeDX [][]float64
var threeDY []float64

var nX [][]float64
var nY []float64

// tests basically make a bunch of planes where
// when the input is above the plane the resultant
// output is 1.0, else 0.0
func init() {

	// create the /tmp/.goml/ dir for persistance testing
	// if it doesn't already exist!
	err := os.MkdirAll("/tmp/.goml", os.ModePerm)
	if err != nil {
		panic(fmt.Sprintf("You should be able to create the directory for goml model persistance testing.\n\tError returned: %v\n", err.Error()))
	}

	// 1 when ( 10*i + j/20 + k ) > 0
	fourDX = [][]float64{}
	fourDY = []float64{}
	for i := -40; i < 40; i += 4 {
		for j := -40; j < 40; j += 4 {
			for k := -40; k < 40; k += 4 {
				fourDX = append(fourDX, []float64{float64(i), float64(j), float64(k)})
				if 10*i+j/20+k > 0 {
					fourDY = append(fourDY, 1.0)
				} else {
					fourDY = append(fourDY, 0.0)
				}
			}
		}
	}

	// 1 when i > 0
	twoDX = [][]float64{}
	twoDY = []float64{}
	for i := -40.0; i < 40.0; i += 0.15 {
		twoDX = append(twoDX, []float64{i})
		if i/2+10 > 0 {
			twoDY = append(twoDY, 1.0)
		} else {
			twoDY = append(twoDY, 0.0)
		}
	}

	threeDX = [][]float64{}
	threeDY = []float64{}

	nX = [][]float64{}
	nY = []float64{}
	// 1 when i+j > 5
	for i := -10; i < 10; i++ {
		for j := -10; j < 10; j++ {
			threeDX = append(threeDX, []float64{float64(i), float64(j)})
			nX = append(nX, []float64{float64(i), float64(j)})

			if i+j > 5 {
				threeDY = append(threeDY, 1.0)
			} else {
				threeDY = append(threeDY, 0.0)
			}
		}
	}

	base.Normalize(nX)

	for i := range nX {
		if nX[i][0]+nX[i][1] > 5 {
			threeDY = append(threeDY, 1.0)
			nY = append(nY, 1.0)
		} else {
			threeDY = append(threeDY, 0.0)
			nY = append(nY, 0.0)
		}
	}
}

// test ( 10*i + j/20 + k ) > 0
func TestFourDimensionalPlaneShouldPass1(t *testing.T) {
	var err error

	model := NewLogistic(base.BatchGA, .000001, 0, 800, fourDX, fourDY)

	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64

	for i := -20; i < 20; i += 10 {
		for j := -20; j < 20; j += 10 {
			for k := -20; k < 20; k += 10 {
				guess, err = model.Predict([]float64{float64(i), float64(j), float64(k)})
				assert.Len(t, guess, 1, "Length of a Logistic model output from the hypothesis should always be a 1 dimensional vector. Never multidimensional.")
				if 10*i+j/20+k > 0 {
					assert.True(t, guess[0] > 0.5, "Prediction should be more likely to be 1")
					assert.True(t, guess[0] < 1.001, "Prediction should never be greater than 1.0")
				} else if 10*i+j/20+k < 0 && guess[0] < 0.5 {
					assert.True(t, guess[0] < 0.5, "Prediction should be more likely to be 0")
					assert.True(t, guess[0] > 0.0, "Prediction should never be less than 0.0")
				}
				assert.Nil(t, err, "Prediction error should be nil")
			}
		}
	}
}

// same as above but with StochasticGA
func TestFourDimensionalPlaneShouldPass2(t *testing.T) {
	var err error

	model := NewLogistic(base.StochasticGA, .000001, 0, 800, fourDX, fourDY)

	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64

	for i := -20; i < 20; i += 10 {
		for j := -20; j < 20; j += 10 {
			for k := -20; k < 20; k += 10 {
				guess, err = model.Predict([]float64{float64(i), float64(j), float64(k)})
				assert.Len(t, guess, 1, "Length of a Logistic model output from the hypothesis should always be a 1 dimensional vector. Never multidimensional.")
				if 10*i+j/20+k > 0 {
					assert.True(t, guess[0] > 0.5, "Prediction should be more likely to be 1")
					assert.True(t, guess[0] < 1.001, "Prediction should never be greater than 1.0")
				} else if 10*i+j/20+k < 0 && guess[0] < 0.5 {
					assert.True(t, guess[0] < 0.5, "Prediction should be more likely to be 0")
					assert.True(t, guess[0] > 0.0, "Prediction should never be less than 0.0")
				}
				assert.Nil(t, err, "Prediction error should be nil")
			}
		}
	}
}

// test ( 10*i + j/20 + k ) > 0 but don't have enough iterations
func TestFourDimensionalPlaneShouldFail1(t *testing.T) {
	var err error

	model := NewLogistic(base.BatchGA, .000001, 0, 1, fourDX, fourDY)

	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64
	var faliures int

	for i := -20; i < 20; i += 7 {
		for j := -20; j < 20; j += 7 {
			for k := -20; k < 20; k += 7 {
				guess, err = model.Predict([]float64{float64(i), float64(j), float64(k)})
				assert.Len(t, guess, 1, "Length of a Logistic model output from the hypothesis should always be a 1 dimensional vector. Never multidimensional.")
				if 10*i+j/20+k > 0 && guess[0] > 0.5 {
					faliures++
				} else if 10*i+j/20+k < 0 && guess[0] < 0.5 {
					faliures++
				}
				assert.Nil(t, err, "Prediction error should be nil")
			}
		}
	}

	assert.True(t, faliures > 40, "There should be more faliures than half of the training set")
}

// same as above but with StochasticGA
func TestFourDimensionalPlaneShouldFail2(t *testing.T) {
	var err error

	model := NewLogistic(base.StochasticGA, .000001, 0, 1, fourDX, fourDY)

	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64
	var faliures int

	for i := -20; i < 20; i += 7 {
		for j := -20; j < 20; j += 7 {
			for k := -20; k < 20; k += 7 {
				guess, err = model.Predict([]float64{float64(i), float64(j), float64(k)})
				assert.Len(t, guess, 1, "Length of a Logistic model output from the hypothesis should always be a 1 dimensional vector. Never multidimensional.")
				if 10*i+j/20+k > 0 && guess[0] > 0.5 {
					faliures++
				} else if 10*i+j/20+k < 0 && guess[0] < 0.5 {
					faliures++
				}
				assert.Nil(t, err, "Prediction error should be nil")
			}
		}
	}

	assert.True(t, faliures > 40, "There should be more faliures than half of the training set")
}

// test ( 10*i + j/20 + k ) > 0 but include an invalid data set
func TestFourDimensionalPlaneShouldFail3(t *testing.T) {
	var err error

	model := NewLogistic(base.BatchGA, 1, 0, 800, [][]float64{}, fourDY)

	err = model.Learn()
	assert.NotNil(t, err, "Learning error should not be nil")

	model = NewLogistic(base.BatchGA, 1, 0, 800, [][]float64{[]float64{}, []float64{}}, fourDY)

	err = model.Learn()
	assert.NotNil(t, err, "Learning error should not be nil")

	model = NewLogistic(base.BatchGA, 1, 0, 800, nil, fourDY)

	err = model.Learn()
	assert.NotNil(t, err, "Learning error should not be nil")
}

// same as above but with StochasticGA
func TestFourDimensionalPlaneShouldFail4(t *testing.T) {
	var err error

	model := NewLogistic(base.StochasticGA, 1, 0, 800, [][]float64{}, fourDY)

	err = model.Learn()
	assert.NotNil(t, err, "Learning error should not be nil")

	model = NewLogistic(base.BatchGA, 1, 0, 800, [][]float64{[]float64{}, []float64{}}, fourDY)

	err = model.Learn()
	assert.NotNil(t, err, "Learning error should not be nil")

	model = NewLogistic(base.BatchGA, 1, 0, 800, nil, fourDY)

	err = model.Learn()
	assert.NotNil(t, err, "Learning error should not be nil")
}

// test ( 10*i + j/20 + k ) > 0 but include an invalid data set
func TestFourDimensionalPlaneShouldFail5(t *testing.T) {
	var err error

	model := NewLogistic(base.BatchGA, 1, 0, 800, fourDX, []float64{})

	err = model.Learn()
	assert.NotNil(t, err, "Learning error should not be nil")

	model = NewLogistic(base.BatchGA, 1, 0, 800, fourDX, nil)

	err = model.Learn()
	assert.NotNil(t, err, "Learning error should not be nil")
}

// same as above but with StochasticGA
func TestFourDimensionalPlaneShouldFail6(t *testing.T) {
	var err error

	model := NewLogistic(base.StochasticGA, 1, 0, 800, fourDX, []float64{})

	err = model.Learn()
	assert.NotNil(t, err, "Learning error should not be nil")

	model = NewLogistic(base.BatchGA, 1, 0, 800, fourDX, nil)

	err = model.Learn()
	assert.NotNil(t, err, "Learning error should not be nil")
}

// invalid method
func TestFourDimensionalPlaneShouldFail7(t *testing.T) {
	var err error

	model := NewLogistic(base.OptimizationMethod("Not A Method!!!"), 1, 0, 800, fourDX, fourDY)

	err = model.Learn()
	assert.NotNil(t, err, "Learning error should not be nil")
}

func TestTwoDimensionalPlaneShouldPass1(t *testing.T) {
	var err error

	model := NewLogistic(base.BatchGA, .0001, 0, 4000, twoDX, twoDY)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64

	for i := -40; i < 20; i++ {
		guess, err = model.Predict([]float64{float64(i)})

		if i/2+10 > 0 {
			assert.True(t, guess[0] > 0.5, "Guess should be more likely to be 1 when i=%v", i)
			assert.True(t, guess[0] < 1.001, "Guess should not exceed 1 ever when")
		} else {
			assert.True(t, guess[0] < 0.5, "Guess should be more likely to be 0 when i=%v", i)
			assert.True(t, guess[0] > 0.0, "Guess should not be below 0 even")
		}

		assert.Len(t, guess, 1, "Length of a Logistic model output from the hypothesis should always be a 1 dimensional vector. Never multidimensional.")
		assert.Nil(t, err, "Prediction error should be nil")
	}
}

// same as above but with StochasticGA
func TestTwoDimensionalPlaneShouldPass2(t *testing.T) {
	var err error

	model := NewLogistic(base.StochasticGA, .0001, 0, 3000, twoDX, twoDY)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64

	for i := -40; i < 20; i++ {
		guess, err = model.Predict([]float64{float64(i)})

		if i/2+10 > 0 {
			assert.True(t, guess[0] > 0.5, "Guess should be more likely to be 1 when i=%v", i)
			assert.True(t, guess[0] < 1.001, "Guess should not exceed 1 ever when")
		} else {
			assert.True(t, guess[0] < 0.5, "Guess should be more likely to be 0 when i=%v", i)
			assert.True(t, guess[0] > 0.0, "Guess should not be below 0 even")
		}

		assert.Len(t, guess, 1, "Length of a Logistic model output from the hypothesis should always be a 1 dimensional vector. Never multidimensional.")
		assert.Nil(t, err, "Prediction error should be nil")
	}
}

// regularization term too large
func TestTwoDimensionalPlaneShouldFail1(t *testing.T) {
	var err error

	model := NewLogistic(base.BatchGA, 1e-4, 1e4, 500, twoDX, twoDY)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64
	var faliures int

	for i := -200; i < 200; i += 15 {
		guess, err = model.Predict([]float64{float64(i)})
		if i/2+10 > 0 && guess[0] < 0.5 {
			faliures++
		} else if i/2+10 < 0 && guess[0] > 0.5 {
			faliures++
		}

		assert.Len(t, guess, 1, "Length of a Logistic model output from the hypothesis should always be a 1 dimensional vector. Never multidimensional.")
		assert.Nil(t, err, "Prediction error should be nil")
	}

	assert.True(t, faliures > 10, "There should be a strong majority of faliures of the training set")
}

// same as above but with StochasticGA
func TestTwoDimensionalPlaneShouldFail2(t *testing.T) {
	var err error

	model := NewLogistic(base.StochasticGA, 1e-4, 1e2, 100, twoDX, twoDY)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64
	var faliures int

	for i := -200; i < 200; i += 15 {
		guess, err = model.Predict([]float64{float64(i)})
		if i/2+10 > 0 && guess[0] < 0.5 {
			faliures++
		} else if i/2+10 < 0 && guess[0] > 0.5 {
			faliures++
		}

		assert.Len(t, guess, 1, "Length of a Logistic model output from the hypothesis should always be a 1 dimensional vector. Never multidimensional.")
		assert.Nil(t, err, "Prediction error should be nil")
	}

	assert.True(t, faliures > 10, "There should be a strong majority of faliures of the training set")
}

// test i+j > 5
func TestThreeDimensionalPlaneShouldPass1(t *testing.T) {
	var err error

	model := NewLogistic(base.BatchGA, .0001, 0, 3000, threeDX, threeDY)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64

	for i := -20; i < 20; i++ {
		for j := -20; j < 20; j++ {
			guess, err = model.Predict([]float64{float64(i), float64(j)})

			if i+j > 5 {
				assert.True(t, guess[0] > 0.5, "Guess should be more likely to be 1")
				assert.True(t, guess[0] < 1.001, "Guess should not exceed 1 ever")
			} else {
				assert.True(t, guess[0] < 0.5, "Guess should be more likely to be 0")
				assert.True(t, guess[0] > 0.0, "Guess should not be below 0 even")
			}

			assert.Len(t, guess, 1, "Length of a Logistic model output from the hypothesis should always be a 1 dimensional vector. Never multidimensional.")
			assert.Nil(t, err, "Prediction error should be nil")
		}
	}
}

// same as above but with StochasticGA
func TestThreeDimensionalPlaneShouldPass2(t *testing.T) {
	var err error

	model := NewLogistic(base.StochasticGA, .0001, 0, 3000, threeDX, threeDY)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64

	for i := -20; i < 20; i++ {
		for j := -20; j < 20; j++ {
			guess, err = model.Predict([]float64{float64(i), float64(j)})

			if i+j > 5 {
				assert.True(t, guess[0] > 0.5, "Guess should be more likely to be 1")
				assert.True(t, guess[0] < 1.001, "Guess should not exceed 1 ever")
			} else {
				assert.True(t, guess[0] < 0.5, "Guess should be more likely to be 0")
				assert.True(t, guess[0] > 0.0, "Guess should not be below 0 even")
			}

			assert.Len(t, guess, 1, "Length of a Logistic model output from the hypothesis should always be a 1 dimensional vector. Never multidimensional.")
			assert.Nil(t, err, "Prediction error should be nil")
		}
	}
}

func TestThreeDimensionalPlaneNormalizedShouldPass1(t *testing.T) {
	var err error

	model := NewLogistic(base.BatchGA, .0001, 0, 3000, nX, nY)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64

	for i := -20; i < 20; i++ {
		for j := -20; j < 20; j++ {
			x := []float64{float64(i), float64(j)}
			base.NormalizePoint(x)

			guess, err = model.Predict(x, true)

			if x[0]+x[1] > 5 {
				assert.True(t, guess[0] > 0.5, "Guess should be more likely to be 1")
				assert.True(t, guess[0] < 1.001, "Guess should not exceed 1 ever")
			} else {
				assert.True(t, guess[0] < 0.5, "Guess should be more likely to be 0")
				assert.True(t, guess[0] > 0.0, "Guess should not be below 0 even")
			}

			assert.Len(t, guess, 1, "Length of a Logistic model output from the hypothesis should always be a 1 dimensional vector. Never multidimensional.")
			assert.Nil(t, err, "Prediction error should be nil")
		}
	}
}

// same as above but with StochasticGA
func TestThreeDimensionalPlaneNormalizedShouldPass2(t *testing.T) {
	var err error

	model := NewLogistic(base.StochasticGA, .0001, 0, 3000, nX, nY)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64

	for i := -20; i < 20; i++ {
		for j := -20; j < 20; j++ {
			x := []float64{float64(i), float64(j)}
			base.NormalizePoint(x)

			guess, err = model.Predict(x, true)

			if x[0]+x[1] > 5 {
				assert.True(t, guess[0] > 0.5, "Guess should be more likely to be 1")
				assert.True(t, guess[0] < 1.001, "Guess should not exceed 1 ever")
			} else {
				assert.True(t, guess[0] < 0.5, "Guess should be more likely to be 0")
				assert.True(t, guess[0] > 0.0, "Guess should not be below 0 even")
			}

			assert.Len(t, guess, 1, "Length of a Logistic model output from the hypothesis should always be a 1 dimensional vector. Never multidimensional.")
			assert.Nil(t, err, "Prediction error should be nil")
		}
	}
}

//* Test Online Learning through channels *//

func TestOnlineOneDXShouldPass1(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.Datapoint, 100)
	errors := make(chan error)

	model := NewLogistic(base.StochasticGA, .0001, 0, 0, nil, nil, 1)

	go model.OnlineLearn(errors, stream, func(theta []float64) {})

	// start passing data to our datastream
	//
	// we could have data already in our channel
	// when we instantiated the Perceptron, though
	for iter := 0; iter < 3000; iter++ {
		for i := -40.0; i < 40; i += 0.15 {
			if 10+i/2 > 0 {
				stream <- base.Datapoint{
					X: []float64{i},
					Y: []float64{1.0},
				}
			} else {
				stream <- base.Datapoint{
					X: []float64{i},
					Y: []float64{0},
				}
			}
		}
	}

	// close the dataset
	close(stream)

	err, more := <-errors

	assert.Nil(t, err, "Learning error should be nil")
	assert.False(t, more, "There should be no errors returned")

	// test a larger dataset now
	iter := 0
	for i := -100.0; i < 100; i += 0.347 {
		guess, err := model.Predict([]float64{i})
		assert.Nil(t, err, "Prediction error should be nil")
		assert.Len(t, guess, 1, "Guess should have length 1")

		if i/2+10 > 0 {
			assert.InDelta(t, 1.0, guess[0], 0.499, "Guess should be 1 for i=%v", i)
		} else {
			assert.InDelta(t, 0.0, guess[0], 0.499, "Guess should be 0 for i=%v", i)
		}
		iter++
	}
	fmt.Printf("Iter: %v\n", iter)
}

func TestOnlineOneDXShouldFail1(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.Datapoint, 1000)
	errors := make(chan error)

	model := NewLogistic(base.StochasticGA, .0001, 0, 0, nil, nil, 1)

	go model.OnlineLearn(errors, stream, func(theta []float64) {})

	// give invalid data when it should be -1
	for i := -500.0; abs(i) > 1; i *= -0.90 {
		if (i-20)/2+10 > 0 {
			stream <- base.Datapoint{
				X: []float64{i - 20},
				Y: []float64{1.0},
			}
		} else {
			stream <- base.Datapoint{
				X: []float64{i - 20, 0.0, 0.0, 0.0},
				Y: []float64{0.0},
			}
		}
	}

	// close the dataset
	close(stream)

	err := <-errors
	assert.NotNil(t, err, "Learning error should not be nil")
}

func TestOnlineOneDXShouldFail2(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.Datapoint, 1000)
	errors := make(chan error)

	model := NewLogistic(base.StochasticGA, .0001, 0, 0, nil, nil, 1)

	go model.OnlineLearn(errors, stream, func(theta []float64) {})

	// give invalid data when it should be -1
	for i := -500.0; abs(i) > 1; i *= -0.90 {
		if i/10+20 > 0 {
			stream <- base.Datapoint{
				X: []float64{i},
				Y: []float64{1.0},
			}
		} else {
			stream <- base.Datapoint{
				X: []float64{i},
				Y: []float64{-1.0, 10, 10, 10},
			}
		}
	}

	// close the dataset
	close(stream)

	err := <-errors
	assert.NotNil(t, err, "Learning error should not be nil -> %v", err)
}

func TestOnlineOneDXShouldFail3(t *testing.T) {
	// create the channel of errors
	errors := make(chan error)

	model := NewLogistic(base.StochasticGA, .0001, 0, 0, nil, nil, 1)

	go model.OnlineLearn(errors, nil, func(theta []float64) {})

	err := <-errors
	assert.NotNil(t, err, "Learning error should not be nil")
}

func TestOnlineFourDXShouldPass1(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.Datapoint, 100)
	errors := make(chan error)

	var updates int

	model := NewLogistic(base.StochasticGA, .0001, 0, 0, nil, nil, 4)

	go model.OnlineLearn(errors, stream, func(theta []float64) {
		updates++
	})

	for iterations := 0; iterations < 25; iterations++ {
		for i := -200.0; abs(i) > 1; i *= -0.75 {
			for j := -200.0; abs(j) > 1; j *= -0.75 {
				for k := -200.0; abs(k) > 1; k *= -0.75 {
					for l := -200.0; abs(l) > 1; l *= -0.75 {
						if i/2+2*k-4*j+2*l+3 > 0 {
							stream <- base.Datapoint{
								X: []float64{i, j, k, l},
								Y: []float64{1.0},
							}
						} else {
							stream <- base.Datapoint{
								X: []float64{i, j, k, l},
								Y: []float64{0.0},
							}
						}
					}
				}
			}
		}
	}

	// close the dataset
	close(stream)

	err, more := <-errors
	assert.Nil(t, err, "Learning error should be nil")
	assert.False(t, more, "There should be no errors returned")

	assert.True(t, updates > 100, "There should be more than 100 updates of theta")

	for i := -200.0; i < 200; i += 100 {
		for j := -200.0; j < 200; j += 100 {
			for k := -200.0; k < 200; k += 100 {
				for l := -200.0; l < 200; l += 100 {
					guess, err := model.Predict([]float64{i, j, k, l})
					assert.Nil(t, err, "Prediction error should be nil")
					assert.Len(t, guess, 1, "Guess should have length 1")

					if i/2+2*k-4*j+2*l+3 > 0 {
						assert.InDelta(t, 1.0, guess[0], 0.48, "Guess should be 1")
					} else {
						assert.InDelta(t, 0.0, guess[0], 0.48, "Guess should be 0")
					}
				}
			}
		}
	}
}

//* Test Persistance to file *//

// test persisting y=x to file
func TestPersistLogisticShouldPass1(t *testing.T) {
	var err error

	model := NewLogistic(base.BatchGA, .0001, 0, 3500, twoDX, twoDY)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64

	for i := -40.0; i < 20; i++ {
		guess, err = model.Predict([]float64{float64(i)})

		if i/2+10 > 0 {
			assert.True(t, guess[0] > 0.5, "Guess should be more likely to be 1")
			assert.True(t, guess[0] < 1.001, "Guess should not exceed 1 ever")
		} else {
			assert.True(t, guess[0] < 0.5, "Guess should be more likely to be 0")
			assert.True(t, guess[0] > 0.0, "Guess should not be below 0 even")
		}

		assert.Len(t, guess, 1, "Length of a Logistic model output from the hypothesis should always be a 1 dimensional vector. Never multidimensional.")
		assert.Nil(t, err, "Prediction error should be nil")
	}

	// not that we know it works, try persisting to file,
	// then resetting the parameter vector theta, then
	// restoring it and testing that predictions are correct
	// again.

	err = model.PersistToFile("/tmp/.goml/Logistic.json")
	assert.Nil(t, err, "Persistance error should be nil")

	model.Parameters = make([]float64, len(model.Parameters))

	// make sure it WONT work now that we reset theta
	//
	// the result of Theta transpose * X should always
	// be 0.5 because the probablility is neutral
	for i := -20; i < 20; i++ {
		guess, err = model.Predict([]float64{float64(i)})
		assert.Len(t, guess, 1, "Length of a Logistic model output from the hypothesis should always be a 1 dimensional vector. Never multidimensional.")
		assert.Equal(t, 0.5, guess[0], "Guess should be 0 when theta is the zero vector")
		assert.Nil(t, err, "Prediction error should be nil")
	}

	err = model.RestoreFromFile("/tmp/.goml/Logistic.json")
	assert.Nil(t, err, "Persistance error should be nil")

	for i := -40.0; i < 20; i++ {
		guess, err = model.Predict([]float64{float64(i)})

		if i/2+10 > 0 {
			assert.True(t, guess[0] > 0.5, "Guess should be more likely to be 1")
			assert.True(t, guess[0] < 1.001, "Guess should not exceed 1 ever")
		} else {
			assert.True(t, guess[0] < 0.5, "Guess should be more likely to be 0")
			assert.True(t, guess[0] > 0.0, "Guess should not be below 0 even")
		}

		assert.Len(t, guess, 1, "Length of a Logistic model output from the hypothesis should always be a 1 dimensional vector. Never multidimensional.")
		assert.Nil(t, err, "Prediction error should be nil")
	}
}
