package cluster

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/cdipaolo/goml/base"

	"github.com/stretchr/testify/assert"
)

func TestComputeCentroidDistanceMatrix1(t *testing.T) {
	model := NewTriangleKMeans(4, 2, circles)

	// now assign centroids to certain values
	// so we can test distance computation
	model.Centroids = [][]float64{
		[]float64{0, 0},
		[]float64{-6, 6},
		[]float64{100, 0},
		[]float64{10, 10},
	}

	// should is the distances such that
	// should[i][j] is the correct
	// distances from centroid[i] to
	// centroid[j]
	//
	// also note that these distances are
	// the *SQUARED* Euclidean distances
	// because it's faster to compute and
	// relative comparison is the same
	should := [][]float64{
		[]float64{0, 72, 10000, 200},
		[]float64{72, 0, 11272, 272},
		[]float64{10000, 11272, 0, 8200},
		[]float64{200, 272, 8200, 0},
	}

	mins := []float64{
		36,
		36,
		4100,
		100,
	}

	model.computeCentroidDistanceMatrix()

	// test matrix similarities from expected
	for i := range should {
		assert.InDeltaSlice(t, should[i], model.centroidDist[i], 1e-6, "Centroid distances should match")
	}

	// now test min similarities
	assert.InDeltaSlice(t, mins, model.minCentroidDist, 1e-6, "Differences in min centroid dist from expected should be small")
}

func TestTriangleKMeansShouldPass1(t *testing.T) {
	model := NewTriangleKMeans(4, 2, circles)

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
func TestTriangleKMeansShouldPass2(t *testing.T) {
	norm := append([][]float64{}, circles...)
	base.Normalize(norm)
	model := NewTriangleKMeans(4, 2, norm)

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

func TestTriangleKMeansShouldPass3(t *testing.T) {

	// test multiple times because of some
	// issues with randomization
	var wrong int
	var count int
	var c1, c2 []float64
	var err error

	for iter := 0; iter < 30; iter++ {
		model := NewTriangleKMeans(2, 2, double)

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

//* Test Persistance *//

func TestTriangleKMeansPersistToFileShouldPass1(t *testing.T) {
	var wrong int
	var count int
	var c1, c2 []float64
	var err error

	model := NewTriangleKMeans(2, 2, double)

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
	assert.True(t, 100*(1-float64(wrong)/float64(count)) <= accuracy, "Reset accuracy should not be greater than the trained accuracy")

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
