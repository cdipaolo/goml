package base

import (
	"fmt"
	"math"
)

// GradientAscent operates on a Ascendable model and
// further optimizes the parameter vector Theta of the
// model, which is then used within the Predict function.
//
// Gradient Ascent follows the following algorithm:
// θ[j] := θ[j] + α·∇J(θ)
//
// where J(θ) is the cost function, α is the learning
// rate, and θ[j] is the j-th value in the parameter
// vector
func GradientAscent(d Ascendable) error {
	Theta := d.Theta()
	Alpha := d.LearningRate()
	MaxIterations := d.MaxIterations()

	// if the iterations given is 0, set it to be
	// 250 (seems reasonable base value)
	if MaxIterations == 0 {
		MaxIterations = 250
	}

	var iter int
	features := len(Theta)

	// Stop iterating if the number of iterations exceeds
	// the limit
	for ; iter < MaxIterations; iter++ {
		newTheta := make([]float64, features)
		for j := range Theta {
			dj, err := d.Dj(j)
			if err != nil {
				return err
			}

			newTheta[j] = Theta[j] + Alpha*dj
		}

		// now simultaneously update Theta
		for j := range Theta {
			newθ := newTheta[j]
			if math.IsInf(newθ, 0) || math.IsNaN(newθ) {
				return fmt.Errorf("Sorry! Learning diverged. Some value of the parameter vector theta is ±Inf or NaN")
			}
			Theta[j] = newθ
		}
	}

	return nil
}

// StochasticGradientAscent operates on a StochasticAscendable
// model and further optimizes the parameter vector Theta of the
// model, which is then used within the Predict function.
// Stochastic gradient descent updates the parameter vector
// after looking at each individual training example, which
// can result in never converging to the absolute minimum; even
// raising the cost function potentially, but it will typically
// converge faster than batch gradient descent (implemented as
// func GradientAscent(d Ascendable) error) because of that very
// difference.
//
// Gradient Ascent follows the following algorithm:
// θ[j] := θ[j] + α·∇J(θ)
//
// where J(θ) is the cost function, α is the learning
// rate, and θ[j] is the j-th value in the parameter
// vector
func StochasticGradientAscent(d StochasticAscendable) error {
	Theta := d.Theta()
	Alpha := d.LearningRate()
	MaxIterations := d.MaxIterations()
	Examples := d.Examples()

	// if the iterations given is 0, set it to be
	// 250 (seems reasonable base value)
	if MaxIterations == 0 {
		MaxIterations = 250
	}

	var iter int
	features := len(Theta)

	// Stop iterating if the number of iterations exceeds
	// the limit
	for ; iter < MaxIterations; iter++ {
		newTheta := make([]float64, features)
		for i := 0; i < Examples; i++ {
			for j := range Theta {
				dj, err := d.Dij(i, j)
				if err != nil {
					return err
				}

				newTheta[j] = Theta[j] + Alpha*dj
			}

			// now simultaneously update Theta
			for j := range Theta {
				newθ := newTheta[j]
				if math.IsInf(newθ, 0) || math.IsNaN(newθ) {
					return fmt.Errorf("Sorry! Learning diverged. Some value of the parameter vector theta is ±Inf or NaN")
				}
				Theta[j] = newθ
			}
		}
	}

	return nil
}
