package linear

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/cdipaolo/goml/base"

	"github.com/stretchr/testify/assert"
)

func init() {
	rand.Seed(42)
}

func TestLocalLinearShouldPass1(t *testing.T) {
	x := [][]float64{}
	y := []float64{}

	// throw in some junk points which
	// should be more-or-less ignored
	// by the weighting
	for i := -70.0; i < -65; i += 2 {
		for j := -70.0; j < -65; j += 2 {
			x = append(x, []float64{i, j})
			y = append(y, 20*(rand.Float64()-0.5))
		}
	}
	for i := 65.0; i < 70; i += 2 {
		for j := 65.0; j < 70; j += 2 {
			x = append(x, []float64{i, j})
			y = append(y, 20*(rand.Float64()-0.5))
		}
	}

	// put in some linear points
	for i := -20.0; i < 20; i++ {
		for j := -20.0; j < 20; j++ {
			x = append(x, []float64{i, j})
			y = append(y, 5*i-5*j-10)
		}
	}

	model := NewLocalLinear(base.BatchGA, 1e-4, 0, 0.75, 1500, x, y)

	var count int
	var err float64
	for i := -15.0; i < 15; i += 4.98 {
		for j := -15.0; j < 15; j += 4.98 {
			guess, predErr := model.Predict([]float64{i, j})
			assert.Nil(t, predErr, "learning/prediction error should be nil")
			count++

			err += abs(guess[0] - (5*i - 5*j - 10))
		}
	}

	avgError := err / float64(count)

	assert.True(t, avgError < 0.4, "Average error should be less than 0.4 (currently %v)", avgError)
	fmt.Printf("Average Error: %v\n\tPoints Tested: %v\n\tTotal Error: %v\n", avgError, count, err)
}

// same as above but with
func TestLocalLinearShouldPass2(t *testing.T) {
	x := [][]float64{}
	y := []float64{}

	// throw in some junk points which
	// should be more-or-less ignored
	// by the weighting
	for i := -70.0; i < -65; i += 2 {
		for j := -70.0; j < -65; j += 2 {
			x = append(x, []float64{i, j})
			y = append(y, 20*(rand.Float64()-0.5))
		}
	}
	for i := 65.0; i < 70; i += 2 {
		for j := 65.0; j < 70; j += 2 {
			x = append(x, []float64{i, j})
			y = append(y, 20*(rand.Float64()-0.5))
		}
	}

	// put in some linear points
	for i := -20.0; i < 20; i++ {
		for j := -20.0; j < 20; j++ {
			x = append(x, []float64{i, j})
			y = append(y, 5*i-5*j-10)
		}
	}

	model := NewLocalLinear(base.StochasticGA, 1e-4, 0, 0.75, 1500, x, y)

	var count int
	var err float64
	for i := -15.0; i < 15; i += 4.98 {
		for j := -15.0; j < 15; j += 4.98 {
			guess, predErr := model.Predict([]float64{i, j})
			assert.Nil(t, predErr, "learning/prediction error should be nil")
			count++

			err += abs(guess[0] - (5*i - 5*j - 10))
		}
	}

	avgError := err / float64(count)

	assert.True(t, avgError < 0.4, "Average error should be less than 0.4 (currently %v)", avgError)
	fmt.Printf("Average Error: %v\n\tPoints Tested: %v\n\tTotal Error: %v\n", avgError, count, err)
}
