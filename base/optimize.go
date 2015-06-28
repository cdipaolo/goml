package base
import (
    "fmt"
    "math"
    )


// abs = |x|
func abs(x float64) float64 {
    if x < 0 {
        return -1 * x
    }
    
    return x
}

// GradientAscent operates on a Descendable model and
// further optimizes the parameter vector theta of the
// model, which is then used within the Predict function
func GradientAscent(d Ascendable) error {
	theta := d.Theta()
	rate := d.LearningRate()
    fmt.Printf("Theta: %v\nAlpha: %v\n\n", theta, rate)

	J, err := d.J()
	if err != nil {
		return err
	}
    costHistory := []float64{J}
    features := len(theta)
    
	for iter := 0; iter < 1000 && !math.IsInf(J, 0); iter++ {
		newTheta := make([]float64, features)
		for j := range theta {
			dj, err := d.Dj(j)
			if err != nil {
				return err
			}
            //fmt.Printf("Dj: %v\trate*Dj: %v\ttheta[j]: %v\n", dj, rate*dj, theta[j])

			newTheta[j] = theta[j] + rate*dj
		}
        
        // now simultaneously update theta
        for j := range theta {
            theta[j] = newTheta[j]
        }

		J, err = d.J()
		if err != nil {
			return err
		}
        
        costHistory = append(costHistory, J)
	}
    fmt.Printf("Cost History: \n%v\n\n", costHistory)

	return nil
}
