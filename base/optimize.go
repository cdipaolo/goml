package base

import (
	"fmt"
	"math"
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

	J, err := d.J()
	if err != nil {
		return err
	}
	var iter int
	features := len(Theta)

	// Stop iterating if the number of iterations exceeds
	// the limit, or if the cost function is infinite.
	for ; iter < MaxIterations && !math.IsInf(J, 0); iter++ {
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

		J, err = d.J()
		if err != nil {
			return err
		}
	}
    
    if math.IsInf(J, 0) || math.IsNaN(J) {
        return fmt.Errorf("ERROR: Learning diverged. Try picking a smaller value for the learning rate alpha! :)")
    }

	fmt.Printf("Went through %v iterations.\n", iter+1)

	return nil
}
