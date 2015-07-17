package base

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNormalizeShouldPass1(t *testing.T) {
	x := [][]float64{}
	y := [][]float64{}

	for i := -200; i < 200; i++ {
		row := []float64{}
		y = append(y, []float64{float64(i)})

		for j := -200; j < 200; j++ {
			row = append(row, float64(j))
		}

		x = append(x, row)
	}

	err := Normalize(x, y)
	assert.Nil(t, err, "Normalization error should be nil")

	for i := range x {
		for j := range x[i] {
			assert.True(t, x[i][j] > -1.0001, "X[i][j] must be greater than -1")
			assert.True(t, x[i][j] < 1.0001, "X[i][j] must be less than 1")
		}

		for j := range y[i] {
			assert.True(t, y[i][j] > -1.0001, "Y[i][j] = %v - should be greater than -1", y[i][j])
			assert.True(t, y[i][j] < 1.0001, "Y[i][j] = %v - should be less than 1", y[i][j])
		}
	}
}

func TestNormalizeShouldPass2(t *testing.T) {
	x := [][]float64{}
	y := [][]float64{}

	for i := -200; i < 200; i++ {
		row := []float64{}
		y = append(y, []float64{float64(i)})

		for j := -200; j < 200; j++ {
			row = append(row, float64(j))
		}

		x = append(x, row)
	}

	err := Normalize(x, y, true)
	assert.Nil(t, err, "Normalization error should be nil")

	for i := range x {
		for j := range x[i] {
			assert.True(t, x[i][j] > -1.0001, "X[i][j] must be greater than -1")
			assert.True(t, x[i][j] < 1.0001, "X[i][j] must be less than 1")
		}

		for j := range y[i] {
			assert.Equal(t, float64(i-200), y[i][j], "Y[i][j] should not have changed")
			assert.Equal(t, float64(i-200), y[i][j], "Y[i][j] should not have changed")
		}
	}
}

func TestNormalizeShouldFail1(t *testing.T) {
	err := Normalize([][]float64{
		[]float64{1, 2, 3, 4, 5},
		[]float64{1, 2, 3, 2, 1},
	}, [][]float64{
		[]float64{1, 2, 3, 4, 5},
	})
	assert.NotNil(t, err, "Normalization error should not be nil")
}

func TestNormalizePointShouldPass1(t *testing.T) {
	x := []float64{}
	y := []float64{}

	for i := -200; i < 200; i++ {
		x = append(x, float64(i))
	}

	NormalizePoint(x, []float64{100.0, -100.0})

	for i := range x {
		assert.True(t, x[i] > -1.0001, "X[i] must be greater than -1")
		assert.True(t, x[i] < 1.0001, "X[i] must be less than 1")
	}

	for i := range y {
		assert.True(t, y[i] > -1.0001, "Y[i] should be greater than -1")
		assert.True(t, y[i] < 1.0001, "Y[i] should be less than 1")
	}
}

func TestNormalizePointShouldPass2(t *testing.T) {
	x := []float64{}
	y := []float64{100.0, -100.0}

	for i := -200; i < 200; i++ {
		x = append(x, float64(i))
	}

	NormalizePoint(x, y, true)

	for i := range x {
		assert.True(t, x[i] > -1.0001, "X[i] must be greater than -1")
		assert.True(t, x[i] < 1.0001, "X[i] must be less than 1")
	}

	assert.Equal(t, 100.0, y[0], "Y[0] should not have changed")
	assert.Equal(t, -100.0, y[1], "Y[1] should not have changed")
}

func TestNormalizePointShouldPass3(t *testing.T) {
	x := []float64{}
	y := []float64{}

	for i := -200; i < 200; i++ {
		x = append(x, float64(i))
	}

	NormalizePoint(x, []float64{100.0, -100.0}, false)

	for i := range x {
		assert.True(t, x[i] > -1.0001, "X[i] must be greater than -1")
		assert.True(t, x[i] < 1.0001, "X[i] must be less than 1")
	}

	for i := range y {
		assert.True(t, y[i] > -1.0001, "Y[i] should be greater than -1")
		assert.True(t, y[i] < 1.0001, "Y[i] should be less than 1")
	}
}

/* Benchmarks */

func BenchmarkNormalizePoint1Input(b *testing.B) {
	x := []float64{100.0}
	y := []float64{100.0}

	for i := 0; i < b.N; i++ {
		NormalizePoint(x, y)
	}
}

func BenchmarkNormalizePoint5Inputs(b *testing.B) {
	x := []float64{100.0, -12345.5, 10.091238491375, 32.0, 78743.1}
	y := []float64{100.0}

	for i := 0; i < b.N; i++ {
		NormalizePoint(x, y)
	}
}

func BenchmarkNormalizePoint50Inputs(b *testing.B) {
	x := []float64{}
	y := []float64{100.0}

	for i := 0; i < 50; i++ {
		x = append(x, float64(i))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NormalizePoint(x, y)
	}
}

func BenchmarkNormalizePoint100Inputs(b *testing.B) {
	x := []float64{}
	y := []float64{100.0}

	for i := 0; i < 100; i++ {
		x = append(x, float64(i))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NormalizePoint(x, y)
	}
}

func BenchmarkNormalizePoint200Inputs(b *testing.B) {
	x := []float64{}
	y := []float64{100.0}

	for i := 0; i < 200; i++ {
		x = append(x, float64(i))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NormalizePoint(x, y)
	}
}

func BenchmarkNormalizePoint300Inputs(b *testing.B) {
	x := []float64{}
	y := []float64{100.0}

	for i := 0; i < 300; i++ {
		x = append(x, float64(i))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NormalizePoint(x, y)
	}
}

func BenchmarkNormalizePoint400Inputs(b *testing.B) {
	x := []float64{}
	y := []float64{100.0}

	for i := 0; i < 400; i++ {
		x = append(x, float64(i))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NormalizePoint(x, y)
	}
}
