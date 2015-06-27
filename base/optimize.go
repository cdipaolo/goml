package base

// GradientDescent operates on a Descendable model and
// further optimizes the parameter vector theta of the
// model, which is then used within the Predict function
func GradientDescent(d Descendable) error {
	theta := d.Theta()
	rate := d.LearningRate()

	JOld, err := d.J()
	if err != nil {
		return err
	}

	var J float64 = JOld - 1
	for iter := 0; iter < 1000 && JOld-J < 1e-3; iter++ {
		newTheta := []float64{}
		for j := range *theta {
			dj, err := d.Dj(j)
			if err != nil {
				return err
			}

			newTheta[j] = (*theta)[j] - rate*dj
		}

		theta = &newTheta

		JOld = J
		J, err = d.J()
		if err != nil {
			return err
		}
	}

	return nil
}
