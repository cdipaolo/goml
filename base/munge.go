package base

import (
	"fmt"
	"math"
)

// Normalize takes in an array of arrays of
// inputs as well as the corresponding array
// of solutions and normalizes each 'row' of
// data to unit vector length.
//
// That is:
// x[i][j] := x[i][j] / |x[i]|
// y[i][j] := y[i][j] / |x[i]|
//
// Note that if you don't want to modify the
// solution values (for instance if you are
// trying to classify within the Bernoulli set
// and the solution values are all in {0,1})
// then you can pass 'true' to the optional
// 'ignoreY' parameter
func Normalize(x [][]float64, y [][]float64, ignoreY ...bool) error {
	if len(x) != len(y) {
		return fmt.Errorf("ERROR: Length of x (%v) does not equal length of y (%v)", len(x), len(y))
	}

	ignore := len(ignoreY) != 0 && ignoreY[0]

	for i := range x {
		NormalizePoint(x[i], y[i], ignore)
	}

	return nil
}

// NormalizePoint is the same as Normalize,
// but it only operates on one singular datapoint,
// normalizing it's value to unit length.
//
// If 'true' is passed for 'ignoreY', then
// the y value is not modified. Note that
// ignoreY is optional.
func NormalizePoint(x []float64, y []float64, ignoreY ...bool) {

	var sum float64
	for i := range x {
		sum += x[i] * x[i]
	}

	mag := math.Sqrt(sum)

	for i := range x {
		x[i] /= mag
	}

	ignore := len(ignoreY) != 0 && ignoreY[0]

	if !ignore {
		for i := range y {
			y[i] /= mag
		}
	}
}
