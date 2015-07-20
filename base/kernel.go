package base

import "math"

// GaussianKernel takes in a parameter for sigma (σ)
// and returns a valid (Gaussian) Radial Basis Function
// Kernel. If the input dimensions aren't valid, the
// kernel will return 0.0 (as if the vectors are orthogonal)
//
//     K(x, x`) = exp( -1 * |x - x`|^2 / 2σ^2)
//
// https://en.wikipedia.org/wiki/Radial_basis_function_kernel
//
// This can be used within any models that can use Kernels.
//
// Sigma (σ) will default to 1 if given 0.0
func GaussianKernel(sigma float64) func([]float64, []float64) float64 {
	if sigma == 0 {
		sigma = 1.0
	}

	denom := 2 * sigma * sigma

	return func(X []float64, x []float64) float64 {

		// don't throw error but fail peacefully
		//
		// returning "not at all similar", basically
		if len(X) != len(x) {
			return 0.0
		}

		var diff float64

		for i := range X {
			diff += (X[i] - x[i]) * (X[i] - x[i])
		}

		return math.Exp(-1 * diff / denom)
	}
}

// LinearKernel is the base kernel function. It
// will return a valid kernel for use within models
// that can use the Kernel Trick. The resultant
// kernel just takes the dot/inner product of it's
// argument vectors.
//
//     K(x, x`) = x*x`
//
// This is also a subset of the Homogeneous Polynomial
// kernel family (where the degree is 1 in this case):
// https://en.wikipedia.org/wiki/Homogeneous_polynomial
//
// Using this kernel is effectively the same as
// not using a kernel at all (for SVM and Kernel
// perceptron, at least.)
func LinearKernel() func([]float64, []float64) float64 {
	return func(X []float64, x []float64) float64 {
		// don't throw error but fail peacefully
		//
		// returning "not at all similar", basically
		if len(X) != len(x) {
			return 0.0
		}

		var dot float64

		for i := range X {
			dot += X[i] * x[i]
		}

		return dot
	}
}

// PolynomialKernel takes in an optional constant (where
// any extra args passed will be added and count as the
// constant,) and a main arg of the degree of the polynomial
// and returns a valid kernel in the Polynomial Function
// Kernel family. This kernel can be used with all models
// that take kernels.
//
//     K(x, x`) = (x*x` + c)^d
//
// https://en.wikipedia.org/wiki/Polynomial_kernel
//
// Note that if no extra argument is passed (no constant)
// then the kernel is a Homogeneous Polynomial Kernel (as
// opposed to Inhomogeneous!) Also if there is no constant
// and d=1, then the returned kernel is the same (though
// less efficient) as just LinearKernel().
//
// https://en.wikipedia.org/wiki/Homogeneous_polynomial
//
// `d` will default to 1 if 0 is given.
func PolynomialKernel(d int, constants ...float64) func([]float64, []float64) float64 {
	if d == 0 {
		d = 1
	}

	var c float64
	if len(constants) != 0 {
		for _, val := range constants {
			c += val
		}
	}

	return func(X []float64, x []float64) float64 {
		// don't throw error but fail peacefully
		//
		// returning "not at all similar", basically
		if len(X) != len(x) {
			return 0.0
		}

		var dot float64

		for i := range X {
			dot += X[i] * x[i]
		}

		return math.Pow(dot+c, float64(d))
	}
}

// TanhKernel takes in a required Kappa modifier
// parameter (defaults to 1.0 if 0.0 given,) and
// optional float64 args afterwords which will be
// added together to create a constant term (general
// reccomended use is to just pass one arg as the
// constant if you need it.)
//
//     K(x, x`) = tanh(κx*x` + c)
//
// https://en.wikipedia.org/wiki/Hyperbolic_function
// https://en.wikipedia.org/wiki/Support_vector_machine#Nonlinear_classification
//
// Note that c must be less than 0 (if >= 0 default
// to -1.0) and κ (for most cases, but not all -
// hence no default) must be greater than 0
func TanhKernel(k float64, constants ...float64) func([]float64, []float64) float64 {
	if k == 0.0 {
		k = 1.0
	}

	var c float64
	if len(constants) != 0 {
		for _, val := range constants {
			c += val
		}
	}

	if c >= 0.0 {
		c = -1.0
	}

	return func(X []float64, x []float64) float64 {
		// don't throw error but fail peacefully
		//
		// returning "not at all similar", basically
		if len(X) != len(x) {
			return 0.0
		}

		var dot float64

		for i := range X {
			dot += k * X[i] * x[i]
		}

		return math.Tanh(dot + c)
	}
}
