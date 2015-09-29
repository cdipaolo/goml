package cluster

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/cdipaolo/goml/base"

	"github.com/stretchr/testify/assert"
)

var (
	fourClusters  [][]float64
	fourClustersY []float64

	twoClusters  [][]float64
	twoClustersY []float64
)

func init() {
	fourClusters = [][]float64{}
	fourClustersY = []float64{}
	for i := -12.0; i < -8; i += 0.1 {
		for j := -12.0; j < -8; j += 0.1 {
			fourClusters = append(fourClusters, []float64{i, j})
			fourClustersY = append(fourClustersY, 0.0)
		}

		for j := 8.0; j < 12; j += 0.1 {
			fourClusters = append(fourClusters, []float64{i, j})
			fourClustersY = append(fourClustersY, 1.0)
		}
	}

	for i := 8.0; i < 12; i += 0.1 {
		for j := -12.0; j < -8; j += 0.1 {
			fourClusters = append(fourClusters, []float64{i, j})
			fourClustersY = append(fourClustersY, 2.0)
		}

		for j := 8.0; j < 12; j += 0.1 {
			fourClusters = append(fourClusters, []float64{i, j})
			fourClustersY = append(fourClustersY, 3.0)
		}
	}

	twoClusters = [][]float64{}
	twoClustersY = []float64{}
	for i := -10.0; i < -3; i += 0.1 {
		for j := -10.0; j < 10; j += 0.1 {
			twoClusters = append(twoClusters, []float64{i, j})
			twoClustersY = append(twoClustersY, 0.0)
		}
	}

	for i := 3.0; i < 10; i += 0.1 {
		for j := -10.0; j < 10; j += 0.1 {
			twoClusters = append(twoClusters, []float64{i, j})
			twoClustersY = append(twoClustersY, 1.0)
		}
	}

	// create the /tmp/.goml/ dir for persistance testing
	// if it doesn't already exist!
	err := os.MkdirAll("/tmp/.goml", os.ModePerm)
	if err != nil {
		panic(fmt.Sprintf("You should be able to create the directory for goml model persistance testing.\n\tError returned: %v\n", err.Error()))
	}
}

func TestInsertSortedShouldPass1(t *testing.T) {
	sorted := insertSorted(nn{
		Distance: 0.0,
	}, []nn{nn{
		Distance: -13.0,
	}, nn{
		Distance: -12.0,
	}, nn{
		Distance: -12.0,
	}, nn{
		Distance: -10.0,
	}}, 4)

	fmt.Printf("Sorted List: %v\n", sorted)

	assert.Equal(t, 4, len(sorted), "Length of sorted array should not change")
	assert.Equal(t, -13.0, sorted[0].Distance, "Sorted[0] should be the lowest distance given")
	assert.Equal(t, -12.0, sorted[1].Distance, "Sorted[1] should match initial")
	assert.Equal(t, -12.0, sorted[2].Distance, "Sorted[2] should match initial")
	assert.Equal(t, -10.0, sorted[3].Distance, "Sorted[3] should match initial")
}

func TestInsertSortedShouldPass2(t *testing.T) {
	sorted := insertSorted(nn{
		Distance: 0.0,
	}, []nn{nn{
		Distance: 12.0,
	}, nn{
		Distance: 13.0,
	}, nn{
		Distance: 14.0,
	}, nn{
		Distance: 15.0,
	}}, 4)

	fmt.Printf("Sorted List: %v\n", sorted)

	assert.Equal(t, 4, len(sorted), "Length of sorted array should not change")
	assert.Equal(t, 0.0, sorted[0].Distance, "Sorted[0] should be the lowest distance given")
	assert.Equal(t, 12.0, sorted[1].Distance, "Sorted[1] should match initial")
	assert.Equal(t, 13.0, sorted[2].Distance, "Sorted[2] should match initial")
	assert.Equal(t, 14.0, sorted[3].Distance, "Sorted[3] should match initial")
}

func TestInsertSortedShouldPass3(t *testing.T) {
	sorted := insertSorted(nn{
		Distance: 14.0,
	}, []nn{nn{
		Distance: 12.0,
	}, nn{
		Distance: 13.0,
	}, nn{
		Distance: 15.0,
	}, nn{
		Distance: 16.0,
	}}, 4)

	fmt.Printf("Sorted List: %v\n", sorted)

	assert.Equal(t, 4, len(sorted), "Length of sorted array should not change")
	assert.Equal(t, 12.0, sorted[0].Distance, "Sorted[0] should be the lowest distance given")
	assert.Equal(t, 13.0, sorted[1].Distance, "Sorted[1] should match initial")
	assert.Equal(t, 14.0, sorted[2].Distance, "Sorted[2] should match initial")
	assert.Equal(t, 15.0, sorted[3].Distance, "Sorted[3] should match initial")
}

func TestInsertSortedShouldPass4(t *testing.T) {
	sorted := insertSorted(nn{
		Distance: 13.0,
	}, []nn{nn{
		Distance: 12.0,
	}, nn{
		Distance: 13.0,
	}, nn{
		Distance: 15.0,
	}, nn{
		Distance: 16.0,
	}}, 4)

	fmt.Printf("Sorted List: %v\n", sorted)

	assert.Equal(t, 4, len(sorted), "Length of sorted array should not change")
	assert.Equal(t, 12.0, sorted[0].Distance, "Sorted[0] should be the lowest distance given")
	assert.Equal(t, 13.0, sorted[1].Distance, "Sorted[1] should match initial")
	assert.Equal(t, 13.0, sorted[2].Distance, "Sorted[2] should match initial")
	assert.Equal(t, 15.0, sorted[3].Distance, "Sorted[3] should match initial")
}

func TestInsertSortedShouldPass5(t *testing.T) {
	sorted := insertSorted(nn{
		Distance: 13.0,
	}, []nn{nn{
		Distance: 12.0,
	}, nn{
		Distance: 13.0,
	}, nn{
		Distance: 15.0,
	}, nn{
		Distance: 16.0,
	}}, 6)

	fmt.Printf("Sorted List: %v\n", sorted)

	assert.Equal(t, 5, len(sorted), "Length of sorted array should allow entry")
	assert.Equal(t, 12.0, sorted[0].Distance, "Sorted[0] should be the lowest distance given")
	assert.Equal(t, 13.0, sorted[1].Distance, "Sorted[1] should match initial")
	assert.Equal(t, 13.0, sorted[2].Distance, "Sorted[2] should match initial")
	assert.Equal(t, 15.0, sorted[3].Distance, "Sorted[3] should match initial")
	assert.Equal(t, 16.0, sorted[4].Distance, "Sorted[4] should match initial")
}

func TestKNNShouldPass1(t *testing.T) {
	model := NewKNN(3, fourClusters, fourClustersY, base.EuclideanDistance)

	var count int
	var wrong int

	duration := time.Duration(0)
	for i := -12.0; i < -8; i += 0.5 {
		for j := -12.0; j < -8; j += 0.5 {
			now := time.Now()
			guess, err := model.Predict([]float64{i, j})
			duration += time.Now().Sub(now)
			assert.Nil(t, err, "Prediction error should be nil")

			if 0.0 != guess[0] {
				wrong++
			}
			count++
		}

		for j := 8.0; j < 12; j += 0.5 {
			now := time.Now()
			guess, err := model.Predict([]float64{i, j})
			duration += time.Now().Sub(now)
			assert.Nil(t, err, "Prediction error should be nil")

			if 1.0 != guess[0] {
				wrong++
			}
			count++
		}
	}

	for i := 8.0; i < 12; i += 0.5 {
		for j := -12.0; j < -8; j += 0.5 {
			now := time.Now()
			guess, err := model.Predict([]float64{i, j})
			duration += time.Now().Sub(now)
			assert.Nil(t, err, "Prediction error should be nil")

			if 2.0 != guess[0] {
				wrong++
			}
			count++
		}

		for j := 8.0; j < 12; j += 0.5 {
			now := time.Now()
			guess, err := model.Predict([]float64{i, j})
			duration += time.Now().Sub(now)
			assert.Nil(t, err, "Prediction error should be nil")

			if 3.0 != guess[0] {
				wrong++
			}
			count++
		}
	}

	accuracy := 100 * (1 - float64(wrong)/float64(count))
	assert.True(t, accuracy > 95, "Accuracy (%v) should be greater than 95 percent", accuracy)
	fmt.Printf("Accuracy: %v percent\n\tPoints Tested: %v\n\tMisclassifications: %v\n\tAverage Prediction Time: %v\n", accuracy, count, wrong, duration/time.Duration(count))
}

// use normalized data
func TestKNNShouldPass2(t *testing.T) {
	norm := append([][]float64{}, fourClusters...)
	base.Normalize(norm)
	model := NewKNN(3, norm, fourClustersY, base.EuclideanDistance)

	var count int
	var wrong int

	duration := time.Duration(0)
	for i := -12.0; i < -8; i += 0.5 {
		for j := -12.0; j < -8; j += 0.5 {
			now := time.Now()
			guess, err := model.Predict([]float64{i, j}, true)
			duration += time.Now().Sub(now)
			assert.Nil(t, err, "Prediction error should be nil")

			if 0.0 != guess[0] {
				wrong++
			}
			count++
		}

		for j := 8.0; j < 12; j += 0.5 {
			now := time.Now()
			guess, err := model.Predict([]float64{i, j}, true)
			duration += time.Now().Sub(now)
			assert.Nil(t, err, "Prediction error should be nil")

			if 1.0 != guess[0] {
				wrong++
			}
			count++
		}
	}

	for i := 8.0; i < 12; i += 0.5 {
		for j := -12.0; j < -8; j += 0.5 {
			now := time.Now()
			guess, err := model.Predict([]float64{i, j}, true)
			duration += time.Now().Sub(now)
			assert.Nil(t, err, "Prediction error should be nil")

			if 2.0 != guess[0] {
				wrong++
			}
			count++
		}

		for j := 8.0; j < 12; j += 0.5 {
			now := time.Now()
			guess, err := model.Predict([]float64{i, j}, true)
			duration += time.Now().Sub(now)
			assert.Nil(t, err, "Prediction error should be nil")

			if 3.0 != guess[0] {
				wrong++
			}
			count++
		}
	}

	accuracy := 100 * (1 - float64(wrong)/float64(count))
	assert.True(t, accuracy > 95, "Accuracy (%v) should be greater than 95 percent", accuracy)
	fmt.Printf("Accuracy: %v percent\n\tPoints Tested: %v\n\tMisclassifications: %v\n\tAverage Prediction Time: %v\n", accuracy, count, wrong, duration/time.Duration(count))
}

func TestKNNShouldPass3(t *testing.T) {
	model := NewKNN(3, twoClusters, twoClustersY, base.EuclideanDistance)

	var count int
	var wrong int

	duration := time.Duration(0)

	for i := -15.0; i < 0; i += 0.5 {
		for j := -20.0; j < 20; j += 0.5 {
			now := time.Now()
			guess, err := model.Predict([]float64{i, j})
			duration += time.Now().Sub(now)
			assert.Nil(t, err, "Prediction error should be nil")

			if 0.0 != guess[0] {
				wrong++
			}
			count++
		}
	}

	for i := 0.0; i < 15; i += 0.5 {
		for j := -20.0; j < 20; j += 0.5 {
			now := time.Now()
			guess, err := model.Predict([]float64{i, j})
			duration += time.Now().Sub(now)
			assert.Nil(t, err, "Prediction error should be nil")

			if 1.0 != guess[0] {
				wrong++
			}
			count++
		}
	}

	accuracy := 100 * (1 - float64(wrong)/float64(count))
	assert.True(t, accuracy > 95, "Accuracy (%v) should be greater than 95 percent", accuracy)
	fmt.Printf("Accuracy: %v percent\n\tPoints Tested: %v\n\tMisclassifications: %v\n\tAverage Prediction Time: %v\n", accuracy, count, wrong, duration/time.Duration(count))
}
