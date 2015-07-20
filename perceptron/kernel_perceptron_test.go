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

func TestLinearKernelOneDXShouldPass1(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.Datapoint, 100)
	errors := make(chan error)

	model := NewKernelPerceptron(base.LinearKernel())

	go model.OnlineLearn(errors, stream, func(supportVector [][]float64) {})

	// start passing data to our datastream
	//
	// we could have data already in our channel
	// when we instantiated the Perceptron, though
	var points int
	for i := -20.1; abs(i) > 1; i *= -0.997 {
		points++
		if (i-20)/2 > 0 {
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

	fmt.Printf("%v Data Points Pushed\n", points)

	// close the dataset
	close(stream)

	err, more := <-errors

	assert.Nil(t, err, "Learning error should be nil")
	assert.False(t, more, "There should be no errors returned")

	// test a larger dataset now
	count := 0
	wrong := 0
	for i := -500.0; i < 500; i++ {
		guess, err := model.Predict([]float64{i})
		assert.Nil(t, err, "Prediction error should be nil")
		assert.Len(t, guess, 1, "Guess should have length 1")

		if i/2 > 0 {
			if guess[0] != 1.0 {
				wrong++
			}
			assert.Equal(t, 1.0, guess[0], "Guess should be 1")
		} else {
			if guess[0] != -1.0 {
				wrong++
			}
			assert.Equal(t, -1.0, guess[0], "Guess should be -1")
		}
		count++
	}
	fmt.Printf("Accuracy: %v\n\tPoints Tested: %v\n\tMisclassifications: %v\n", 100*(1-float64(wrong)/float64(count)), count, wrong)
}

func TestLinearKernelOneDXShouldFail1(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.Datapoint, 1000)
	errors := make(chan error)

	model := NewKernelPerceptron(base.LinearKernel())

	go model.OnlineLearn(errors, stream, func(supportVector [][]float64) {})

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

	for {
		err, more := <-errors

		if more {
			assert.NotNil(t, err, "Learning error should not be nil")
		} else {
			break
		}
	}
}

func TestLinearKernelOneDXShouldFail2(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.Datapoint, 1000)
	errors := make(chan error)

	model := NewKernelPerceptron(base.LinearKernel())

	go model.OnlineLearn(errors, stream, func(supportVector [][]float64) {})

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

	for {
		err, more := <-errors

		if more {
			assert.NotNil(t, err, "Learning error should not be nil")
		} else {
			break
		}
	}
}

func TestLinearKernelOneDXShouldFail3(t *testing.T) {
	// create the channel of errors
	errors := make(chan error)

	model := NewKernelPerceptron(base.LinearKernel())

	go model.OnlineLearn(errors, nil, func(supportVector [][]float64) {})

	for {
		err, more := <-errors

		if more {
			assert.NotNil(t, err, "Learning error should not be nil")
		} else {
			break
		}
	}
}

func TestLinearKernelFourDXShouldPass1(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.Datapoint, 100)
	errors := make(chan error)

	var updates int

	model := NewKernelPerceptron(base.LinearKernel())

	go model.OnlineLearn(errors, stream, func(theta [][]float64) {
		updates++
	})

	var count int
	for i := -200.0; abs(i) > 1; i *= -0.7 {
		for j := -200.0; abs(j) > 1; j *= -0.7 {
			for k := -200.0; abs(k) > 1; k *= -0.7 {
				for l := -200.0; abs(l) > 1; l *= -0.7 {
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

					count++
				}
			}
		}
	}

	fmt.Printf("%v Training Examples Pushed\n", count)

	// close the dataset
	close(stream)

	err, more := <-errors
	assert.Nil(t, err, "Learning error should be nil")
	assert.False(t, more, "There should be no errors returned")

	assert.True(t, updates > 100, "There should be more than 100 updates of theta")

	count = 0
	wrong := 0

	for i := -200.0; i < 200; i += 100 {
		for j := -200.0; j < 200; j += 100 {
			for k := -200.0; k < 200; k += 100 {
				for l := -200.0; l < 200; l += 100 {
					guess, err := model.Predict([]float64{i, j, k, l})
					assert.Nil(t, err, "Prediction error should be nil")
					assert.Len(t, guess, 1, "Guess should have length 1")

					count++

					if i/2+2*k-4*j+2*l+3 > 0 {
						if guess[0] != 1.0 {
							wrong++
						}
					} else {
						if guess[0] != -1.0 {
							wrong++
						}
					}
				}
			}
		}
	}

	accuracy := 100 * (1 - float64(wrong)/float64(count))

	assert.True(t, accuracy > 97, "There should be greater than 97 percent accuracy (currently %v)", accuracy)
	fmt.Printf("Accuracy: %v\n\tPoints Tested: %v\n\tMisclassifications: %v\n", accuracy, count, wrong)
}

func TestGaussianKernelFourDXShouldPass1(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.Datapoint, 100)
	errors := make(chan error)

	var updates int

	model := NewKernelPerceptron(base.GaussianKernel(50))

	go model.OnlineLearn(errors, stream, func(theta [][]float64) {
		updates++
	})

	var count int
	go func() {
		for i := -200.0; abs(i) > 1; i *= -0.7 {
			for j := -200.0; abs(j) > 1; j *= -0.7 {
				for k := -200.0; abs(k) > 1; k *= -0.7 {
					for l := -200.0; abs(l) > 1; l *= -0.7 {
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

						count++
					}
				}
			}
		}

		// close the dataset
		close(stream)
	}()

	fmt.Printf("%v Training Examples Pushed\n", count)

	err, more := <-errors
	assert.Nil(t, err, "Learning error should be nil")
	assert.False(t, more, "There should be no errors returned")

	assert.True(t, updates > 100, "There should be more than 100 updates of theta")

	count = 0
	wrong := 0

	for i := -200.0; i < 200; i += 100 {
		for j := -200.0; j < 200; j += 100 {
			for k := -200.0; k < 200; k += 100 {
				for l := -200.0; l < 200; l += 100 {
					guess, err := model.Predict([]float64{i, j, k, l})
					assert.Nil(t, err, "Prediction error should be nil")
					assert.Len(t, guess, 1, "Guess should have length 1")

					count++

					if i/2+2*k-4*j+2*l+3 > 0 {
						if guess[0] != 1.0 {
							wrong++
						}
					} else {
						if guess[0] != -1.0 {
							wrong++
						}
					}
				}
			}
		}
	}

	accuracy := 100 * (1 - float64(wrong)/float64(count))

	assert.True(t, accuracy > 95, "There should be greater than 95 percent accuracy (currently %v)", accuracy)
	fmt.Printf("Accuracy: %v\n\tPoints Tested: %v\n\tMisclassifications: %v\n", accuracy, count, wrong)
}

func TestPolynomialKernelFourDXShouldPass1(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.Datapoint, 100)
	errors := make(chan error)

	var updates int

	model := NewKernelPerceptron(base.PolynomialKernel(3))

	go model.OnlineLearn(errors, stream, func(theta [][]float64) {
		updates++
	})

	var count int
	for i := -200.0; abs(i) > 1; i *= -0.65 {
		for j := -200.0; abs(j) > 1; j *= -0.65 {
			for k := -200.0; abs(k) > 1; k *= -0.65 {
				for l := -200.0; abs(l) > 1; l *= -0.65 {
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

					count++
				}
			}
		}
	}

	fmt.Printf("%v Training Examples Pushed\n", count)

	// close the dataset
	close(stream)

	err, more := <-errors
	assert.Nil(t, err, "Learning error should be nil")
	assert.False(t, more, "There should be no errors returned")

	assert.True(t, updates > 50, "There should be more than 50 updates of theta")

	count = 0
	wrong := 0

	for i := -200.0; i < 200; i += 100 {
		for j := -200.0; j < 200; j += 100 {
			for k := -200.0; k < 200; k += 100 {
				for l := -200.0; l < 200; l += 100 {
					guess, err := model.Predict([]float64{i, j, k, l})
					assert.Nil(t, err, "Prediction error should be nil")
					assert.Len(t, guess, 1, "Guess should have length 1")

					count++

					if i/2+2*k-4*j+2*l+3 > 0 {
						if guess[0] != 1.0 {
							wrong++
						}
					} else {
						if guess[0] != -1.0 {
							wrong++
						}
					}
				}
			}
		}
	}

	accuracy := 100 * (1 - float64(wrong)/float64(count))

	assert.True(t, accuracy > 95, "There should be greater than 95 percent accuracy (currently %v)", accuracy)
	fmt.Printf("Accuracy: %v\n\tPoints Tested: %v\n\tMisclassifications: %v\n", accuracy, count, wrong)
}

func TestTanhKernelFourDXShouldPass1(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.Datapoint, 100)
	errors := make(chan error)

	var updates int

	model := NewKernelPerceptron(base.TanhKernel(5))

	go model.OnlineLearn(errors, stream, func(theta [][]float64) {
		updates++
	})

	var count int
	for i := -200.0; abs(i) > 1; i *= -0.45 {
		for j := -200.0; abs(j) > 1; j *= -0.45 {
			for k := -200.0; abs(k) > 1; k *= -0.45 {
				for l := -200.0; abs(l) > 1; l *= -0.45 {
					if i/2+2*k-4*j+2*l > 0 {
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

					count++
				}
			}
		}
	}

	fmt.Printf("%v Training Examples Pushed\n", count)

	// close the dataset
	close(stream)

	err, more := <-errors
	assert.Nil(t, err, "Learning error should be nil")
	assert.False(t, more, "There should be no errors returned")

	assert.True(t, updates > 100, "There should be more than 100 updates of theta")

	count = 0
	wrong := 0

	for i := -200.0; i < 200; i += 100 {
		for j := -200.0; j < 200; j += 100 {
			for k := -200.0; k < 200; k += 100 {
				for l := -200.0; l < 200; l += 100 {
					guess, err := model.Predict([]float64{i, j, k, l})
					assert.Nil(t, err, "Prediction error should be nil")
					assert.Len(t, guess, 1, "Guess should have length 1")

					count++

					if i/2+2*k-4*j+2*l > 0 {
						if guess[0] != 1.0 {
							wrong++
						}
					} else {
						if guess[0] != -1.0 {
							wrong++
						}
					}
				}
			}
		}
	}

	accuracy := 100 * (1 - float64(wrong)/float64(count))

	assert.True(t, accuracy > 75, "There should be greater than 75 percent accuracy (currently %v)", accuracy)
	fmt.Printf("Accuracy: %v\n\tPoints Tested: %v\n\tMisclassifications: %v\n", accuracy, count, wrong)
}

func TestLinearKernelTwoDXNormalizedShouldPass1(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.Datapoint, 100)
	errors := make(chan error)

	var updates int

	model := NewKernelPerceptron(base.LinearKernel())

	go model.OnlineLearn(errors, stream, func(theta [][]float64) {
		updates++
	}, true)

	for i := -200.0; abs(i) > 1; i *= -0.981 {
		for j := -200.0; abs(j) > 1; j *= -0.981 {
			x := []float64{i, j}
			base.NormalizePoint(x)

			if 5*x[0]+10*x[1]-4 > 0 {
				stream <- base.Datapoint{
					X: x,
					Y: []float64{1.0},
				}
			} else {
				stream <- base.Datapoint{
					X: x,
					Y: []float64{-1.0},
				}
			}
		}
	}

	// close the dataset
	close(stream)

	for {
		err, more := <-errors
		assert.False(t, more, "There should not be any errors!")

		if more {
			assert.Nil(t, err, "Learning error should be nil")
		} else {
			break
		}
	}

	assert.True(t, updates > 50, "There should be more than 50 updates of theta (%v updates recorded)", updates)

	var count int
	var incorrect int

	for i := -200.0; abs(i) > 1; i *= -0.85 {
		for j := -200.0; abs(j) > 1; j *= -0.85 {
			x := []float64{i, j}
			base.NormalizePoint(x)

			guess, err := model.Predict([]float64{i, j}, true)
			assert.Nil(t, err, "Prediction error should be nil")
			assert.Len(t, guess, 1, "Guess should have length 1")

			if 5*x[0]+10*x[1]-4 > 0 && guess[0] != 1.0 {
				incorrect++
			} else if 5*x[0]+10*x[1]-4 <= 0 && guess[0] != -1.0 {
				incorrect++
			}

			count++
		}
	}

	fmt.Printf("Predictions: %v\n\tIncorrect: %v\n\tAccuracy Rate: %v percent\n", count, incorrect, 100*(1.0-float64(incorrect)/float64(count)))
	assert.True(t, float64(incorrect)/float64(count) < 0.14, "Accuracy should be greater than 86%")
}

func TestGaussianKernelPersistPerceptronShouldPass1(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.Datapoint, 100)
	errors := make(chan error)

	model := NewKernelPerceptron(base.GaussianKernel(50))

	go model.OnlineLearn(errors, stream, func(supportVector [][]float64) {})

	// start passing data to our datastream
	//
	// we could have data already in our channel
	// when we instantiated the Perceptron, though
	go func() {
		var count int
		for i := -200.0; abs(i) > 1; i *= -0.7 {
			for j := -200.0; abs(j) > 1; j *= -0.7 {
				for k := -200.0; abs(k) > 1; k *= -0.7 {
					for l := -200.0; abs(l) > 1; l *= -0.7 {
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

						count++
					}
				}
			}
		}

		fmt.Printf("%v Training Examples Pushed\n", count)

		// close the dataset
		close(stream)
	}()

	err, more := <-errors
	assert.Nil(t, err, "Learning error should be nil")
	assert.False(t, more, "There should be no errors returned")

	// test a larger dataset now
	var count int
	var wrong int
	for i := -200.0; i < 200; i += 100 {
		for j := -200.0; j < 200; j += 100 {
			for k := -200.0; k < 200; k += 100 {
				for l := -200.0; l < 200; l += 100 {
					guess, err := model.Predict([]float64{i, j, k, l})
					assert.Nil(t, err, "Prediction error should be nil")
					assert.Len(t, guess, 1, "Guess should have length 1")

					count++

					if i/2+2*k-4*j+2*l+3 > 0 {
						if guess[0] != 1.0 {
							wrong++
						}
					} else {
						if guess[0] != -1.0 {
							wrong++
						}
					}
				}
			}
		}
	}

	accuracy := 100 * (1 - float64(wrong)/float64(count))

	assert.True(t, accuracy > 95, "There should be greater than 95 percent accuracy (currently %v)", accuracy)
	fmt.Printf("Accuracy: %v\n\tPoints Tested: %v\n\tMisclassifications: %v\n", accuracy, count, wrong)

	// now persist to file
	err = model.PersistToFile("/tmp/.goml/KernelPerceptron.json")
	assert.Nil(t, err, "Persistance error should be nil")

	model.SV = []base.Datapoint{}

	// make sure it WONT work now that we reset theta
	//
	// the result of Theta transpose * X should always
	// be 0 because theta is the zero vector right now.
	wrong = 0
	count = 0
	for i := -200.0; i < 200; i += 100 {
		for j := -200.0; j < 200; j += 100 {
			for k := -200.0; k < 200; k += 100 {
				for l := -200.0; l < 200; l += 100 {
					guess, err := model.Predict([]float64{i, j, k, l})
					assert.Nil(t, err, "Prediction error should be nil")
					assert.Len(t, guess, 1, "Guess should have length 1")

					count++

					if i/2+2*k-4*j+2*l+3 > 0 {
						if guess[0] != 1.0 {
							wrong++
						}
					} else {
						if guess[0] != -1.0 {
							wrong++
						}
					}
				}
			}
		}
	}

	accuracy = 100 * (1 - float64(wrong)/float64(count))

	assert.True(t, accuracy < 51, "There should be less than 50 percent accuracy (currently %v)", accuracy)

	// restore from file
	err = model.RestoreFromFile("/tmp/.goml/KernelPerceptron.json")
	assert.Nil(t, err, "Persistance error should be nil")

	// test with original data
	// test a larger dataset now
	wrong = 0
	count = 0
	for i := -200.0; i < 200; i += 100 {
		for j := -200.0; j < 200; j += 100 {
			for k := -200.0; k < 200; k += 100 {
				for l := -200.0; l < 200; l += 100 {
					guess, err := model.Predict([]float64{i, j, k, l})
					assert.Nil(t, err, "Prediction error should be nil")
					assert.Len(t, guess, 1, "Guess should have length 1")

					count++

					if i/2+2*k-4*j+2*l+3 > 0 {
						if guess[0] != 1.0 {
							wrong++
						}
					} else {
						if guess[0] != -1.0 {
							wrong++
						}
					}
				}
			}
		}
	}

	accuracy = 100 * (1 - float64(wrong)/float64(count))

	assert.True(t, accuracy > 95, "There should be greater than 95 percent accuracy (currently %v)", accuracy)
	fmt.Printf("Accuracy: %v\n\tPoints Tested: %v\n\tMisclassifications: %v\n", accuracy, count, wrong)
}
