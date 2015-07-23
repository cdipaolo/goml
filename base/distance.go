package base

import "math"

// DistanceMeasure is any function that
// maps two vectors of float64s to a
// float64. Used for vector distance
// calculations
type DistanceMeasure func([]float64, []float64) float64

// EuclideanDistance returns the distance
// betweek two float64 vectors. NOTE that
// this function does not check that the
// vectors are different lengths (to improve
// computation speed in, say, KNN.) Make
// sure you pass in same-length vectors.
func EuclideanDistance(u []float64, v []float64) float64 {
	var sum float64
	for i := range u {
		sum += (u[i] - v[i]) * (u[i] - v[i])
	}
	return math.Sqrt(sum)
}

// ManhattanDistance returns the manhattan
// distance between teo float64 vectors.
// This is the sum of the differences between
// each value
//
// Example Points:
//     .
//     |
//    2|
//     |______.
//        2
//
// Note that the Euclidean distance between
// these 2 points is 2*sqrt(2)=2.828. The Manhattan
// distance is 4.
//
// NOTE that this function does not check that
// the vectors are different lengths (to improve
// computation speed in, say, KNN.) Make
// sure you pass in same-length vectors.
func ManhattanDistance(u []float64, v []float64) float64 {
	var sum float64
	for i := range u {
		sum += math.Abs(u[i] - v[i])
	}
	return sum
}

// LNorm returns a DistanceMeasure of the
// l-p norm. L norms are a generalized family
// of the Euclidean and Manhattan distance.
//
// https://en.wikipedia.org/wiki/Norm_(mathematics)
//
// (p = 1) -> Manhattan Distance
// (p = 2) -> Euclidean Distance
//
// NOTE that this function does not check that
// the vectors are different lengths (to improve
// computation speed in, say, KNN.) Make
// sure you pass in same-length vectors.
func LNorm(p int) DistanceMeasure {
	return func(u []float64, v []float64) float64 {
		var sum float64
		for i := range u {
			sum += math.Pow(u[i]-v[i], float64(p))
		}
		return math.Pow(sum, 1/float64(p))
	}
}
