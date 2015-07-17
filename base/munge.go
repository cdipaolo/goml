package base

import "math"

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

	if mag == 0 {
		return
	}

	for i := range x {
		x[i] /= mag
	}
}
