package cluster

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/cdipaolo/goml/base"

	"github.com/stretchr/testify/assert"
)

var (
	circles [][]float64
	double  [][]float64
)

func init() {
	circles = [][]float64{}
	for i := -12.0; i < -8; i += 0.2 {
		for j := -12.0; j < -8; j += 0.2 {
			circles = append(circles, []float64{i, j})
		}

		for j := 8.0; j < 12; j += 0.2 {
			circles = append(circles, []float64{i, j})
		}
	}

	for i := 8.0; i < 12; i += 0.2 {
		for j := -12.0; j < -8; j += 0.2 {
			circles = append(circles, []float64{i, j})
		}

		for j := 8.0; j < 12; j += 0.2 {
			circles = append(circles, []float64{i, j})
		}
	}

	double = [][]float64{}
	for i := -10.0; i < -3; i += 0.1 {
		for j := -10.0; j < 10; j += 0.1 {
			double = append(double, []float64{i, j})
		}
	}

	for i := 3.0; i < 10; i += 0.1 {
		for j := -10.0; j < 10; j += 0.1 {
			double = append(double, []float64{i, j})
		}
	}
}

// Note that these tests usually have
// 100% accuracy, but I'm thresholding faliure
// at 87% because maybe <10% of the time the
// randomization of the clusters leaves two
// areas with the same classification
func TestKMeansShouldPass1(t *testing.T) {
	model := NewKMeans(4, 30, circles)

	assert.Nil(t, model.Learn(), "Learning error should be nil")

	// now predict with the same training set and
	// make sure the classes are the same within
	// each block
	c1, err := model.Predict([]float64{-10, -10})
	assert.Nil(t, err, "Prediction error should be nil")

	c2, err := model.Predict([]float64{-10, 10})
	assert.Nil(t, err, "Prediction error should be nil")

	c3, err := model.Predict([]float64{10, -10})
	assert.Nil(t, err, "Prediction error should be nil")

	c4, err := model.Predict([]float64{10, 10})
	assert.Nil(t, err, "Prediction error should be nil")

	var count int
	var wrong int

	for i := -12.0; i < -8; i += 0.2 {
		for j := -12.0; j < -8; j += 0.2 {
			guess, err := model.Predict([]float64{i, j})
			assert.Nil(t, err, "Prediction error should be nil")

			if c1[0] != guess[0] {
				wrong++
			}
			count++
		}

		for j := 8.0; j < 12; j += 0.2 {
			guess, err := model.Predict([]float64{i, j})
			assert.Nil(t, err, "Prediction error should be nil")

			if c2[0] != guess[0] {
				wrong++
			}
			count++
		}
	}

	for i := 8.0; i < 12; i += 0.2 {
		for j := -12.0; j < -8; j += 0.2 {
			guess, err := model.Predict([]float64{i, j})
			assert.Nil(t, err, "Prediction error should be nil")

			if c3[0] != guess[0] {
				wrong++
			}
			count++
		}

		for j := 8.0; j < 12; j += 0.2 {
			guess, err := model.Predict([]float64{i, j})
			assert.Nil(t, err, "Prediction error should be nil")

			if c4[0] != guess[0] {
				wrong++
			}
			count++
		}
	}

	accuracy := 100 * (1 - float64(wrong)/float64(count))
	assert.True(t, accuracy > 87, "Accuracy (%v) should be greater than 87 percent", accuracy)
	fmt.Printf("Accuracy: %v percent\n\tPoints Tested: %v\n\tMisclassifications: %v\n\tClasses: %v\n", accuracy, count, wrong, []float64{c1[0], c2[0], c3[0], c4[0]})
}

// use normalized data
func TestKMeansShouldPass2(t *testing.T) {
	norm := append([][]float64{}, circles...)
	base.Normalize(norm)
	model := NewKMeans(4, 30, norm)

	assert.Nil(t, model.Learn(), "Learning error should be nil")

	// now predict with the same training set and
	// make sure the classes are the same within
	// each block
	c1, err := model.Predict([]float64{-10, -10}, true)
	assert.Nil(t, err, "Prediction error should be nil")

	c2, err := model.Predict([]float64{-10, 10}, true)
	assert.Nil(t, err, "Prediction error should be nil")

	c3, err := model.Predict([]float64{10, -10}, true)
	assert.Nil(t, err, "Prediction error should be nil")

	c4, err := model.Predict([]float64{10, 10}, true)
	assert.Nil(t, err, "Prediction error should be nil")

	var count int
	var wrong int

	for i := -12.0; i < -8; i += 0.2 {
		for j := -12.0; j < -8; j += 0.2 {
			guess, err := model.Predict([]float64{i, j}, true)
			assert.Nil(t, err, "Prediction error should be nil")

			if c1[0] != guess[0] {
				wrong++
			}
			count++
		}

		for j := 8.0; j < 12; j += 0.2 {
			guess, err := model.Predict([]float64{i, j}, true)
			assert.Nil(t, err, "Prediction error should be nil")

			if c2[0] != guess[0] {
				wrong++
			}
			count++
		}
	}

	for i := 8.0; i < 12; i += 0.2 {
		for j := -12.0; j < -8; j += 0.2 {
			guess, err := model.Predict([]float64{i, j}, true)
			assert.Nil(t, err, "Prediction error should be nil")

			if c3[0] != guess[0] {
				wrong++
			}
			count++
		}

		for j := 8.0; j < 12; j += 0.2 {
			guess, err := model.Predict([]float64{i, j}, true)
			assert.Nil(t, err, "Prediction error should be nil")

			if c4[0] != guess[0] {
				wrong++
			}
			count++
		}
	}

	accuracy := 100 * (1 - float64(wrong)/float64(count))
	assert.True(t, accuracy > 87, "Accuracy (%v) should be greater than 87 percent", accuracy)
	fmt.Printf("Accuracy: %v percent\n\tPoints Tested: %v\n\tMisclassifications: %v\n\tClasses: %v\n", accuracy, count, wrong, []float64{c1[0], c2[0], c3[0], c4[0]})
}

func TestKMeansShouldPass3(t *testing.T) {

	// test multiple times because of some
	// issues with randomization
	var wrong int
	var count int
	var c1, c2 []float64
	var err error

	for iter := 0; iter < 30; iter++ {
		model := NewKMeans(2, 30, double)

		assert.Nil(t, model.Learn(), "Learning error should be nil")

		// now predict with the same training set and
		// make sure the classes are the same within
		// each block
		c1, err = model.Predict([]float64{-7.5, 0})
		assert.Nil(t, err, "Prediction error should be nil")

		c2, err = model.Predict([]float64{7.5, 0})
		assert.Nil(t, err, "Prediction error should be nil")

		for i := -10.0; i < -3; i++ {
			for j := -10.0; j < 10; j++ {
				guess, err := model.Predict([]float64{i, j})
				assert.Nil(t, err, "Prediction error should be nil")

				if c1[0] != guess[0] {
					wrong++
				}
				count++
			}
		}

		for i := 3.0; i < 10; i += 0.7 {
			for j := -10.0; j < 10; j += 0.7 {
				guess, err := model.Predict([]float64{i, j})
				assert.Nil(t, err, "Prediction error should be nil")

				if c2[0] != guess[0] {
					wrong++
				}
				count++
			}
		}
	}

	accuracy := 100 * (1 - float64(wrong)/float64(count))
	assert.True(t, accuracy > 80, "Accuracy (%v) should be greater than 80 percent", accuracy)
	fmt.Printf("Accuracy: %v percent\n\tPoints Tested: %v\n\tMisclassifications: %v\n\tClasses: %v\n", accuracy, count, wrong, []float64{c1[0], c2[0]})
}

//* Test Online KMeans *//

func TestOnlineKMeansShouldPass1(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.Datapoint, 100)
	errors := make(chan error, 20)

	model := NewKMeans(4, 0, nil, OnlineParams{
		alpha:    0.5,
		features: 4,
	})

	go model.OnlineLearn(errors, stream, func(theta [][]float64) {})

	go func() {
		// start passing data to our datastream
		//
		// we could have data already in our channel
		// when we instantiated the model, though
		for i := -40.0; i < -30; i += 4.99 {
			for j := -40.0; j < -30; j += 4.99 {
				for k := -40.0; k < -30; k += 4.99 {
					for l := -40.0; l < -30; l += 4.99 {
						stream <- base.Datapoint{
							X: []float64{i, j, k, l},
						}
					}
				}
			}
		}
		for i := -40.0; i < -30; i += 4.99 {
			for j := 30.0; j < 40; j += 4.99 {
				for k := -40.0; k < -30; k += 4.99 {
					for l := 30.0; l < 40; l += 4.99 {
						stream <- base.Datapoint{
							X: []float64{i, j, k, l},
						}
					}
				}
			}
		}
		for i := 30.0; i < 40; i += 4.99 {
			for j := -40.0; j < -30; j += 4.99 {
				for k := 30.0; k < 40; k += 4.99 {
					for l := -40.0; l < -30; l += 4.99 {
						stream <- base.Datapoint{
							X: []float64{i, j, k, l},
						}
					}
				}
			}
		}
		for i := 30.0; i < 40; i += 4.99 {
			for j := -40.0; j < -30; j += 4.99 {
				for k := -40.0; k < -30; k += 4.99 {
					for l := 30.0; l < 40; l += 4.99 {
						stream <- base.Datapoint{
							X: []float64{i, j, k, l},
						}
					}
				}
			}
		}

		// close the dataset
		close(stream)
	}()

	err, more := <-errors

	assert.Nil(t, err, "Learning error should be nil")
	assert.False(t, more, "There should be no errors returned")

	c1, err := model.Predict([]float64{-35, -35, -35, -35})
	assert.Nil(t, err, "Prediction error should be nil")

	c2, err := model.Predict([]float64{-35, 35, -35, 35})
	assert.Nil(t, err, "Prediction error should be nil")

	c3, err := model.Predict([]float64{35, -35, 35, -35})
	assert.Nil(t, err, "Prediction error should be nil")

	c4, err := model.Predict([]float64{35, -35, -35, 35})
	assert.Nil(t, err, "Prediction error should be nil")

	var count int
	var wrong int

	// test a larger dataset now
	for i := -40.0; i < -30; i += 1.99 {
		for j := -40.0; j < -30; j += 1.99 {
			for k := -40.0; k < -30; k += 1.99 {
				for l := -40.0; l < -30; l += 1.99 {
					guess, err := model.Predict([]float64{i, j, k, l})
					assert.Nil(t, err, "Prediction error should be nil")

					if guess[0] != c1[0] {
						wrong++
					}
					count++
				}
			}
		}
	}
	for i := -40.0; i < -30; i += 1.99 {
		for j := 30.0; j < 40; j += 1.99 {
			for k := -40.0; k < -30; k += 1.99 {
				for l := 30.0; l < 40; l += 1.99 {
					guess, err := model.Predict([]float64{i, j, k, l})
					assert.Nil(t, err, "Prediction error should be nil")

					if guess[0] != c2[0] {
						wrong++
					}
					count++
				}
			}
		}
	}
	for i := 30.0; i < 40; i += 1.99 {
		for j := -40.0; j < -30; j += 1.99 {
			for k := 30.0; k < 40; k += 1.99 {
				for l := -40.0; l < -30; l += 1.99 {
					guess, err := model.Predict([]float64{i, j, k, l})
					assert.Nil(t, err, "Prediction error should be nil")

					if guess[0] != c3[0] {
						wrong++
					}
					count++
				}
			}
		}
	}
	for i := 30.0; i < 40; i += 1.99 {
		for j := -40.0; j < -30; j += 1.99 {
			for k := -40.0; k < -30; k += 1.99 {
				for l := 30.0; l < 40; l += 1.99 {
					guess, err := model.Predict([]float64{i, j, k, l})
					assert.Nil(t, err, "Prediction error should be nil")

					if guess[0] != c4[0] {
						wrong++
					}
					count++
				}
			}
		}
	}

	accuracy := 100 * (1 - float64(wrong)/float64(count))
	assert.True(t, accuracy > 95, "Accuracy (%v) should be greater than 95 percent", accuracy)
	fmt.Printf("Accuracy: %v percent\n\tPoints Tested: %v\n\tMisclassifications: %v\n\tClasses: %v\n", accuracy, count, wrong, []float64{c1[0], c2[0], c3[0], c4[0]})
}

//* Test Persistance *//

func TestKMeansPersistToFileShouldPass1(t *testing.T) {
	var wrong int
	var count int
	var c1, c2 []float64
	var err error

	model := NewKMeans(2, 30, double)

	assert.Nil(t, model.Learn(), "Learning error should be nil")

	// now predict with the same training set and
	// make sure the classes are the same within
	// each block
	c1, err = model.Predict([]float64{-7.5, 0})
	assert.Nil(t, err, "Prediction error should be nil")

	c2, err = model.Predict([]float64{7.5, 0})
	assert.Nil(t, err, "Prediction error should be nil")

	for i := -10.0; i < -3; i += 0.7 {
		for j := -10.0; j < 10; j += 0.7 {
			guess, err := model.Predict([]float64{i, j})
			assert.Nil(t, err, "Prediction error should be nil")

			if c1[0] != guess[0] {
				wrong++
			}
			count++
		}
	}

	for i := 3.0; i < 10; i += 0.7 {
		for j := -10.0; j < 10; j += 0.7 {
			guess, err := model.Predict([]float64{i, j})
			assert.Nil(t, err, "Prediction error should be nil")

			if c2[0] != guess[0] {
				wrong++
			}
			count++
		}
	}

	accuracy := 100 * (1 - float64(wrong)/float64(count))
	fmt.Printf("Accuracy: %v percent\n\tPoints Tested: %v\n\tMisclassifications: %v\n\tClasses: %v\n", accuracy, count, wrong, []float64{c1[0], c2[0]})

	// persist to file!
	assert.Nil(t, model.PersistToFile("/tmp/.goml/KMeans.csv"), "Persist error should be nil")

	rand.Seed(time.Now().UTC().Unix())

	features := len(model.Centroids[0])
	model.Centroids = make([][]float64, len(model.Centroids))
	for i := range model.Centroids {
		model.Centroids[i] = make([]float64, features)
	}

	wrong = 0
	count = 0
	for i := -10.0; i < -3; i += 0.1 {
		for j := -10.0; j < 10; j += 0.1 {
			guess, err := model.Predict([]float64{i, j})
			assert.Nil(t, err, "Prediction error should be nil")

			if c1[0] != guess[0] {
				wrong++
			}
			count++
		}
	}

	for i := 3.0; i < 10; i += 0.1 {
		for j := -10.0; j < 10; j += 0.1 {
			guess, err := model.Predict([]float64{i, j})
			assert.Nil(t, err, "Prediction error should be nil")

			if c2[0] != guess[0] {
				wrong++
			}
			count++
		}
	}
	assert.NotEqual(t, 100*(1-float64(wrong)/float64(count)), accuracy, "Reset accuracy should not be equal to trained accuracy")

	// restore from file!
	assert.Nil(t, model.RestoreFromFile("/tmp/.goml/KMeans.csv"), "Restore error should be nil")

	wrong = 0
	count = 0
	for i := -10.0; i < -3; i += 0.1 {
		for j := -10.0; j < 10; j += 0.1 {
			guess, err := model.Predict([]float64{i, j})
			assert.Nil(t, err, "Prediction error should be nil")

			if c1[0] != guess[0] {
				wrong++
			}
			count++
		}
	}

	for i := 3.0; i < 10; i += 0.1 {
		for j := -10.0; j < 10; j += 0.1 {
			guess, err := model.Predict([]float64{i, j})
			assert.Nil(t, err, "Prediction error should be nil")

			if c2[0] != guess[0] {
				wrong++
			}
			count++
		}
	}

	assert.InDelta(t, 100*(1-float64(wrong)/float64(count)), accuracy, 1, "Accuracy Should be Equal")

	// save results to disk
	assert.Nil(t, model.SaveClusteredData("/tmp/.goml/KMeansResults.csv"), "Save results error should be nil")
}
