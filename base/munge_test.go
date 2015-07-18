package base

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNormalizeShouldPass1(t *testing.T) {
	x := [][]float64{}

	for i := -200; i < 200; i++ {
		row := []float64{}

		for j := -200; j < 200; j++ {
			row = append(row, float64(j))
		}

		x = append(x, row)
	}

	Normalize(x)

	for i := range x {
		for j := range x[i] {
			assert.True(t, x[i][j] > -1.0001, "X[i][j] must be greater than -1")
			assert.True(t, x[i][j] < 1.0001, "X[i][j] must be less than 1")
		}
	}
}

func TestNormalizeShouldPass2(t *testing.T) {
	x := [][]float64{}

	for i := -200; i < 200; i++ {
		row := []float64{}

		for j := -200; j < 200; j++ {
			row = append(row, float64(j))
		}

		x = append(x, row)
	}

	Normalize(x)

	for i := range x {
		for j := range x[i] {
			assert.True(t, x[i][j] > -1.0001, "X[i][j] must be greater than -1")
			assert.True(t, x[i][j] < 1.0001, "X[i][j] must be less than 1")
		}

	}
}

func TestNormalizePointShouldPass1(t *testing.T) {
	x := []float64{}

	for i := -200; i < 200; i++ {
		x = append(x, float64(i))
	}

	NormalizePoint(x)

	for i := range x {
		assert.True(t, x[i] > -1.0001, "X[i] must be greater than -1")
		assert.True(t, x[i] < 1.0001, "X[i] must be less than 1")
	}
}

func TestNormalizePointShouldPass2(t *testing.T) {
	x := []float64{}

	for i := -200; i < 200; i++ {
		x = append(x, float64(i))
	}

	NormalizePoint(x)

	for i := range x {
		assert.True(t, x[i] > -1.0001, "X[i] must be greater than -1")
		assert.True(t, x[i] < 1.0001, "X[i] must be less than 1")
	}

}

func TestNormalizePointShouldPass3(t *testing.T) {
	x := []float64{}

	for i := -200; i < 200; i++ {
		x = append(x, float64(i))
	}

	NormalizePoint(x)

	for i := range x {
		assert.True(t, x[i] > -1.0001, "X[i] must be greater than -1")
		assert.True(t, x[i] < 1.0001, "X[i] must be less than 1")
	}
}

/* Benchmarks */

func BenchmarkNormalizePoint1Input(b *testing.B) {
	x := []float64{100.0}

	for i := 0; i < b.N; i++ {
		NormalizePoint(x)
	}
}

func BenchmarkNormalizePoint5Inputs(b *testing.B) {
	x := []float64{100.0, -12345.5, 10.091238491375, 32.0, 78743.1}

	for i := 0; i < b.N; i++ {
		NormalizePoint(x)
	}
}

func BenchmarkNormalizePoint50Inputs(b *testing.B) {
	x := []float64{}

	for i := 0; i < 50; i++ {
		x = append(x, float64(i))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NormalizePoint(x)
	}
}

func BenchmarkNormalizePoint100Inputs(b *testing.B) {
	x := []float64{}

	for i := 0; i < 100; i++ {
		x = append(x, float64(i))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NormalizePoint(x)
	}
}

func BenchmarkNormalizePoint200Inputs(b *testing.B) {
	x := []float64{}

	for i := 0; i < 200; i++ {
		x = append(x, float64(i))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NormalizePoint(x)
	}
}

func BenchmarkNormalizePoint300Inputs(b *testing.B) {
	x := []float64{}

	for i := 0; i < 300; i++ {
		x = append(x, float64(i))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NormalizePoint(x)
	}
}

func BenchmarkNormalizePoint400Inputs(b *testing.B) {
	x := []float64{}

	for i := 0; i < 400; i++ {
		x = append(x, float64(i))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NormalizePoint(x)
	}
}
