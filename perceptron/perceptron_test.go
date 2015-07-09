package perceptron

import (
	"fmt"
	"os"
	"testing"

	"github.com/cdipaolo/goml/base"

	"github.com/stretchr/testify/assert"
)

func init() {
	// create the /tmp/.goml/ dir for persistance testing
	// if it doesn't already exist!
	err := os.MkdirAll("/tmp/.goml", os.ModePerm)
	if err != nil {
		panic(fmt.Sprintf("You should be able to create the directory for goml model persistance testing.\n\tError returned: %v\n", err.Error()))
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -1 * x
	}

	return x
}

func TestOneDXShouldPass1(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.Datapoint, 100)
	errors := make(chan error)

	model := NewPerceptron(0.1, 1)

	go model.OnlineLearn(errors, stream, func(theta []float64) {})

	// start passing data to our datastream
	//
	// we could have data already in our channel
	// when we instantiated the Perceptron, though
	for i := -500.0; abs(i) > 1; i *= -0.997 {
		if 10+(i-20)/2 > 0 {
			stream <- base.Datapoint{
				X: []float64{i - 20},
				Y: []float64{1.0},
			}
		} else {
			stream <- base.Datapoint{
				X: []float64{i - 20},
				Y: []float64{0},
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
	for i := -500.0; i < 500; i++ {
		guess, err := model.Predict([]float64{i})
		assert.Nil(t, err, "Prediction error should be nil")
		assert.Len(t, guess, 1, "Guess should have length 1")

		if i/2+10 > 0 {
			assert.Equal(t, 1.0, guess[0], "Guess should be 1")
		} else {
			assert.Equal(t, -1.0, guess[0], "Guess should be -1")
		}
		iter++
	}
	fmt.Printf("Iter: %v", iter)
}

func TestOneDXShouldFail1(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.Datapoint, 1000)
	errors := make(chan error)

	model := NewPerceptron(0.1, 1)

	go model.OnlineLearn(errors, stream, func(theta []float64) {})

	// give invalid data when it should be -1
	for i := -500.0; abs(i) > 1; i *= -0.99 {
		if (i-20)/2+10 > 0 {
			stream <- base.Datapoint{
				X: []float64{i - 20},
				Y: []float64{1.0},
			}
		} else {
			stream <- base.Datapoint{
				X: []float64{i - 20, 0.0, 0.0, 0.0},
				Y: []float64{-1.0},
			}
		}
	}

	// close the dataset
	close(stream)

	err := <-errors
	assert.NotNil(t, err, "Learning error should not be nil")
}

func TestOneDXShouldFail2(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.Datapoint, 1000)
	errors := make(chan error)

	model := NewPerceptron(0.1, 1)

	go model.OnlineLearn(errors, stream, func(theta []float64) {})

	// give invalid data when it should be -1
	for i := -500.0; abs(i) > 1; i *= -0.99 {
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
	assert.NotNil(t, err, "Learning error should not be nil")
}

func TestOneDXShouldFail3(t *testing.T) {
	// create the channel of errors
	errors := make(chan error)

	model := NewPerceptron(0.1, 1)

	go model.OnlineLearn(errors, nil, func(theta []float64) {})

	err := <-errors
	assert.NotNil(t, err, "Learning error should not be nil")
}

func TestFourDXShouldPass1(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.Datapoint, 100)
	errors := make(chan error)

	var updates int

	model := NewPerceptron(0.1, 4)

	go model.OnlineLearn(errors, stream, func(theta []float64) {
		updates++
	})

	var iter int
	for i := -200.0; abs(i) > 1; i *= -0.82 {
		for j := -200.0; abs(j) > 1; j *= -0.82 {
			for k := -200.0; abs(k) > 1; k *= -0.82 {
				for l := -200.0; abs(l) > 1; l *= -0.82 {
					if i/2+2*k-4*j+2*l+3 > 0 {
						stream <- base.Datapoint{
							X: []float64{i, j, k, l},
							Y: []float64{1.0},
						}
					} else {
						stream <- base.Datapoint{
							X: []float64{i, j, k, l},
							Y: []float64{-1.0},
						}
					}

					iter++
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
						assert.Equal(t, 1.0, guess[0], "Guess should be 1")
					} else {
						assert.Equal(t, -1.0, guess[0], "Guess should be -1")
					}
				}
			}
		}
	}
}

func TestPersistPerceptronShouldPass1(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.Datapoint, 100)
	errors := make(chan error)

	model := NewPerceptron(0.1, 1)

	go model.OnlineLearn(errors, stream, func(theta []float64) {})

	// start passing data to our datastream
	//
	// we could have data already in our channel
	// when we instantiated the Perceptron, though
	for i := -500.0; abs(i) > 1; i *= -0.997 {
		if 10+(i-20)/2 > 0 {
			stream <- base.Datapoint{
				X: []float64{i - 20},
				Y: []float64{1.0},
			}
		} else {
			stream <- base.Datapoint{
				X: []float64{i - 20},
				Y: []float64{0},
			}
		}
	}

	// close the dataset
	close(stream)

	err, more := <-errors
	assert.Nil(t, err, "Learning error should be nil")
	assert.False(t, more, "There should be no errors returned")

	// test a larger dataset now
	for i := -500.0; i < 500; i++ {
		guess, err := model.Predict([]float64{i})
		assert.Nil(t, err, "Prediction error should be nil")
		assert.Len(t, guess, 1, "Guess should have length 1")

		if i/2+10 > 0 {
			assert.Equal(t, 1.0, guess[0], "Guess should be 1")
		} else {
			assert.Equal(t, -1.0, guess[0], "Guess should be -1")
		}
	}

	// now persist to file
	err = model.PersistToFile("/tmp/.goml/Perceptron.json")
	assert.Nil(t, err, "Persistance error should be nil")

	model.Parameters = make([]float64, len(model.Parameters))

	// make sure it WONT work now that we reset theta
	//
	// the result of Theta transpose * X should always
	// be 0 because theta is the zero vector right now.
	for i := -40; i < 40; i++ {
		guess, err := model.Predict([]float64{float64(i)})
		assert.Len(t, guess, 1, "Length of a Perceptron model output from the hypothesis should always be a 1 dimensional vector. Never multidimensional.")
		assert.Equal(t, -1.0, guess[0], "Guess should be 0 when theta is the zero vector")
		assert.Nil(t, err, "Prediction error should be nil")
	}

	// restore from file
	err = model.RestoreFromFile("/tmp/.goml/Perceptron.json")
	assert.Nil(t, err, "Persistance error should be nil")

	// test with original data
	// test a larger dataset now
	for i := -500.0; i < 500; i++ {
		guess, err := model.Predict([]float64{i})
		assert.Nil(t, err, "Prediction error should be nil")
		assert.Len(t, guess, 1, "Guess should have length 1")

		if i/2+10 > 0 {
			assert.Equal(t, 1.0, guess[0], "Guess should be 1")
		} else {
			assert.Equal(t, -1.0, guess[0], "Guess should be -1")
		}
	}
}
