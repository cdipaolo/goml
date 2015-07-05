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
	for i := -10; i < 10; i += 1 {
		twoDX = append(twoDX, []float64{float64(i)})
		if i > 0 {
			twoDY = append(twoDY, 1.0)
		} else {
			twoDY = append(twoDY, 0.0)
		}
	}

	threeDX = [][]float64{}
	threeDY = []float64{}
	// 1 when i+j > 5
	for i := -10; i < 10; i++ {
		for j := -10; j < 10; j++ {
			threeDX = append(threeDX, []float64{float64(i), float64(j)})

			if i+j > 5 {
				threeDY = append(threeDY, 1.0)
			} else {
				threeDY = append(threeDY, 0.0)
			}
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
				if 10*i + j/20 + k > 0 {
					assert.True(t, guess[0] > 0.5, "Prediction should be more likely to be 1")
                    assert.True(t, guess[0] < 1.001, "Prediction should never be greater than 1.0")
                } else if 10*i + j/20 + k < 0 && guess[0] < 0.5 {
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
				if 10*i + j/20 + k > 0 {
					assert.True(t, guess[0] > 0.5, "Prediction should be more likely to be 1")
                    assert.True(t, guess[0] < 1.001, "Prediction should never be greater than 1.0")
                } else if 10*i + j/20 + k < 0 && guess[0] < 0.5 {
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
				if 10*i + j/20 + k > 0 && guess[0] > 0.5 {
					faliures++
                } else if 10*i + j/20 + k < 0 && guess[0] < 0.5 {
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
				if 10*i + j/20 + k > 0 && guess[0] > 0.5 {
					faliures++
                } else if 10*i + j/20 + k < 0 && guess[0] < 0.5 {
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

// test i+j > 5
func TestTwoDimensionalPlaneShouldPass1(t *testing.T) {
	var err error

	model := NewLogistic(base.BatchGA, .0001, 0, 500, twoDX, twoDY)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64

	for i := -20; i < 20; i += 3 {
        guess, err = model.Predict([]float64{float64(i)})

        if i > 0 {
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

// same as above but with StochasticGA
func TestTwoDimensionalPlaneShouldPass2(t *testing.T) {
	var err error

	model := NewLogistic(base.StochasticGA, .0001, 0, 500, twoDX, twoDY)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64

	for i := -20; i < 20; i += 3 {
        guess, err = model.Predict([]float64{float64(i)})

        if i > 0 {
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

// regularization term too large
func TestTwoDimensionalPlaneShouldFail1(t *testing.T) {
	var err error

	model := NewLogistic(base.BatchGA, 1e-4, 1e3, 500, twoDX, twoDY)
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

        assert.Len(t, guess, 1, "Length of a Logistic model output from the hypothesis should always be a 1 dimensional vector. Never multidimensional.")
        assert.Nil(t, err, "Prediction error should be nil")
	}

	assert.True(t, faliures > 10, "There should be a strong majority of faliures of the training set")
}

// same as above but with StochasticGA
func TestTwoDimensionalPlaneShouldFail2(t *testing.T) {
	var err error

	model := NewLogistic(base.StochasticGA, 1e-4, 1e3, 300, twoDX, twoDY)
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


// test persisting y=x to file
func TestPersistLogisticShouldPass1(t *testing.T) {
	var err error

	model := NewLogistic(base.BatchGA, .0001, 0, 500, twoDX, twoDY)
	err = model.Learn()
	assert.Nil(t, err, "Learning error should be nil")

	var guess []float64

	for i := -20; i < 20; i++ {
		guess, err = model.Predict([]float64{float64(i)})
        
        if i > 0 {
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

    for i := -20; i < 20; i++ {
		guess, err = model.Predict([]float64{float64(i)})
        
        if i > 0 {
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
