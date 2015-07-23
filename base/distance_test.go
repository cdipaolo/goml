package base

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDistanceEuclideanShouldPass1(t *testing.T) {
	u := []float64{1, 2.5, 1.23, 0.105}
	v := []float64{3, 3.1, 2, 1.2}

	assert.InDelta(t, 2.480307, EuclideanDistance(u, v), 1e-3, "Distance should match")
}

func TestDistanceEuclideanShouldPass2(t *testing.T) {
	u := []float64{0, 2.5, 1.23, -10.013}
	v := []float64{3, 3.1, 2, 1.2}

	assert.InDelta(t, 11.648359, EuclideanDistance(u, v), 1e-3, "Distance should match")
}

func TestDistanceManhattanShouldPass1(t *testing.T) {
	u := []float64{0, 2.5, 1.23, -10.013, 1.3}
	v := []float64{3, 3.1, 2, 1.2, 1.2}

	assert.InDelta(t, 15.683, ManhattanDistance(u, v), 1e-3, "Distance should match")
}

func TestDistanceManhattanShouldPass2(t *testing.T) {
	u := []float64{-100.123, 2.5, 100.0162, -10.013, 1.3}
	v := []float64{3, 3.1, 2, 1.2, 1.2}

	assert.InDelta(t, 213.0522, ManhattanDistance(u, v), 1e-3, "Distance should match")
}
