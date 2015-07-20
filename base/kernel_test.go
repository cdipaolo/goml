package base

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGaussianKernelShouldPass1(t *testing.T) {
	k := GaussianKernel(1.0)

	// test different dot products which
	// should be valid

	assert.InDelta(t, math.Exp(-1*1.0 / 2, k([]float64{
		0.0, 1.0, 1.0, 0.0,
	}, []float64{
		0.0, 1.0, 0.0, 0.0,
	}), 5e-4, "Dot product should be valid")

	assert.InDelta(t, math.Exp(-1*3.0 / 2, k([]float64{
		15.0, 1.0, -1.0, 0.0,
	}, []float64{
		1.0, 1.0, 10.0, 0.0,
	}), 5e-4, "Dot product should be valid")

	assert.InDelta(t, math.Exp(-1*-87.0 / 2, k([]float64{
		15.0, 1.0, -1.0, 0.0,
	}, []float64{
		1.0, 1.0, 100.0, 0.0,
	}), 5e-4, "Dot product should be valid")
}

func TestGaussianKernelShouldPass2(t *testing.T) {
	k := GaussianKernel(4.0)

	// test different dot products which
	// should be valid

	assert.InDelta(t, math.Exp(-1*1.0 / 32, k([]float64{
		0.0, 1.0, 1.0, 0.0,
	}, []float64{
		0.0, 1.0, 0.0, 0.0,
	}), 5e-4, "Dot product should be valid")

	assert.InDelta(t, math.Exp(-1*3.0 / 32, k([]float64{
		15.0, 1.0, -1.0, 0.0,
	}, []float64{
		1.0, 1.0, 10.0, 0.0,
	}), 5e-4, "Dot product should be valid")

	assert.InDelta(t, math.Exp(-1*-87.0 / 32, k([]float64{
		15.0, 1.0, -1.0, 0.0,
	}, []float64{
		1.0, 1.0, 100.0, 0.0,
	}), 5e-4, "Dot product should be valid")
}

func TestLinearKernelShouldPass1(t *testing.T) {
	k := LinearKernel()

	// test different dot products which
	// should be valid

	assert.InDelta(t, 1.0, k([]float64{
		0.0, 1.0, 1.0, 0.0,
	}, []float64{
		0.0, 1.0, 0.0, 0.0,
	}), 5e-4, "Dot product should be valid")

	assert.InDelta(t, 6.0, k([]float64{
		15.0, 1.0, -1.0, 0.0,
	}, []float64{
		1.0, 1.0, 10.0, 0.0,
	}), 5e-4, "Dot product should be valid")

	assert.InDelta(t, -84.0, k([]float64{
		15.0, 1.0, -1.0, 0.0,
	}, []float64{
		1.0, 1.0, 100.0, 0.0,
	}), 5e-4, "Dot product should be valid")
}

func TestPolynomialKernelShouldPass1(t *testing.T) {
	k := PolynomialKernel(1)

	// test different dot products which
	// should be valid

	assert.InDelta(t, 1.0, k([]float64{
		0.0, 1.0, 1.0, 0.0,
	}, []float64{
		0.0, 1.0, 0.0, 0.0,
	}), 5e-4, "Dot product should be valid")

	assert.InDelta(t, 6.0, k([]float64{
		15.0, 1.0, -1.0, 0.0,
	}, []float64{
		1.0, 1.0, 10.0, 0.0,
	}), 5e-4, "Dot product should be valid")

	assert.InDelta(t, -84.0, k([]float64{
		15.0, 1.0, -1.0, 0.0,
	}, []float64{
		1.0, 1.0, 100.0, 0.0,
	}), 5e-4, "Dot product should be valid")
}

func TestPolynomialKernelShouldPass2(t *testing.T) {
	k := PolynomialKernel(1, 10, 6)

	// test different dot products which
	// should be valid

	assert.InDelta(t, 1.0+16.0, k([]float64{
		0.0, 1.0, 1.0, 0.0,
	}, []float64{
		0.0, 1.0, 0.0, 0.0,
	}), 5e-4, "Dot product should be valid")

	assert.InDelta(t, 6.0+16.0, k([]float64{
		15.0, 1.0, -1.0, 0.0,
	}, []float64{
		1.0, 1.0, 10.0, 0.0,
	}), 5e-4, "Dot product should be valid")

	assert.InDelta(t, -84.0+16.0, k([]float64{
		15.0, 1.0, -1.0, 0.0,
	}, []float64{
		1.0, 1.0, 100.0, 0.0,
	}), 5e-4, "Dot product should be valid")
}

func TestPolynomialKernelShouldPass3(t *testing.T) {
	k := PolynomialKernel(3, 16)

	// test different dot products which
	// should be valid

	assert.InDelta(t, math.Pow(1.0+16.0, 3.0), k([]float64{
		0.0, 1.0, 1.0, 0.0,
	}, []float64{
		0.0, 1.0, 0.0, 0.0,
	}), 5e-4, "Dot product should be valid")

	assert.InDelta(t, math.Pow(6.0+16.0, 3.0), k([]float64{
		15.0, 1.0, -1.0, 0.0,
	}, []float64{
		1.0, 1.0, 10.0, 0.0,
	}), 5e-4, "Dot product should be valid")

	assert.InDelta(t, math.Pow(-84.0+16.0, 3.0), k([]float64{
		15.0, 1.0, -1.0, 0.0,
	}, []float64{
		1.0, 1.0, 100.0, 0.0,
	}), 5e-4, "Dot product should be valid")
}

func TestTanhKernelShouldPass1(t *testing.T) {
	k := TanhKernel(1)

	// test different dot products which
	// should be valid

	// when constant is 0, default to -1.0

	assert.InDelta(t, math.Tanh(1.0-1.0), k([]float64{
		0.0, 1.0, 1.0, 0.0,
	}, []float64{
		0.0, 1.0, 0.0, 0.0,
	}), 5e-4, "Dot product should be valid")

	assert.InDelta(t, math.Tanh(6.0-1.0), k([]float64{
		15.0, 1.0, -1.0, 0.0,
	}, []float64{
		1.0, 1.0, 10.0, 0.0,
	}), 5e-4, "Dot product should be valid")

	assert.InDelta(t, math.Tanh(-84.0-1.0), k([]float64{
		15.0, 1.0, -1.0, 0.0,
	}, []float64{
		1.0, 1.0, 100.0, 0.0,
	}), 5e-4, "Dot product should be valid")
}

func TestTanhKernelShouldPass2(t *testing.T) {
	k := TanhKernel(1, -10.0, -6.0)

	// test different dot products which
	// should be valid

	assert.InDelta(t, math.Tanh(1.0-16.0), k([]float64{
		0.0, 1.0, 1.0, 0.0,
	}, []float64{
		0.0, 1.0, 0.0, 0.0,
	}), 5e-4, "Dot product should be valid")

	assert.InDelta(t, math.Tanh(6.0-16.0), k([]float64{
		15.0, 1.0, -1.0, 0.0,
	}, []float64{
		1.0, 1.0, 10.0, 0.0,
	}), 5e-4, "Dot product should be valid")

	assert.InDelta(t, math.Tanh(-84.0-16.0), k([]float64{
		15.0, 1.0, -1.0, 0.0,
	}, []float64{
		1.0, 1.0, 100.0, 0.0,
	}), 5e-4, "Dot product should be valid")
}

func TestTanhKernelShouldPass3(t *testing.T) {
	k := TanhKernel(3, -16.0)

	// test different dot products which
	// should be valid

	assert.InDelta(t, math.Tanh(3.0-16.0), k([]float64{
		0.0, 1.0, 1.0, 0.0,
	}, []float64{
		0.0, 1.0, 0.0, 0.0,
	}), 5e-4, "Dot product should be valid")

	assert.InDelta(t, math.Tanh(18.0-16.0), k([]float64{
		15.0, 1.0, -1.0, 0.0,
	}, []float64{
		1.0, 1.0, 10.0, 0.0,
	}), 5e-4, "Dot product should be valid")

	assert.InDelta(t, math.Tanh(-252.0-16.0), k([]float64{
		15.0, 1.0, -1.0, 0.0,
	}, []float64{
		1.0, 1.0, 100.0, 0.0,
	}), 5e-4, "Dot product should be valid")
}
