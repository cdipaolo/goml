package base

import (
	"fmt"
)

// GradientAscent operates on a Descendable model and
// further optimizes the parameter vector Theta of the
// model, which is then used within the Predict function
func GradientAscent(d Ascendable) error {
	Theta := d.Theta()
	Alpha := d.LearningRate()
	MaxIterations := d.MaxIterations()

	// if the iterations given is 0, set it to be
	// 5000 (seems reasonable base value)
	if MaxIterations == 0 {
		MaxIterations = 5000
	}
    
	var iter int
	features := len(Theta)

	// Stop iterating if the number of iterations exceeds
	// the limit, or if the cost function is infinite.
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
			Theta[j] = newTheta[j]
		}
	}

	fmt.Printf("Went through %v iterations.\n", iter+1)

	return nil
}
