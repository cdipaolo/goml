package base

import (
	"math"
)

// Normalize takes in an array of arrays of
// inputs as well as the corresponding array
// of solutions and normalizes each 'row' of
// data to unit vector length.
//
// That is:
// x[i][j] := x[i][j] / |x[i]|
func Normalize(x [][]float64) {
	for i := range x {
		NormalizePoint(x[i])
	}
}

// NormalizePoint is the same as Normalize,
// but it only operates on one singular datapoint,
// normalizing it's value to unit length.
func NormalizePoint(x []float64) {

	var sum float64
	for i := range x {
		sum += x[i] * x[i]
	}

	mag := math.Sqrt(sum)

	for i := range x {
		if math.IsInf(x[i]/mag, 0) || math.IsNaN(x[i]/mag) {
			// fallback to zero when dividing by 0
			x[i] = 0
			continue
		}

		x[i] /= mag
	}
}
