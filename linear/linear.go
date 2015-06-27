package linear

import (
	"bytes"
	"fmt"

	"github.com/cdipaolo/goml/base"
)

// LeastSquares implements a standard linear regression model
// with a Least Squares cost function.
//
// https://en.wikipedia.org/wiki/Least_squares
//
// The model uses gradient descent, NOT regular equations.
type LeastSquares struct {
	alpha float64

	// trainingSet and expectedResults are the
	// 'x', and 'y' of the data, expressed as
	// vectors, that the model can optimize from
	trainingSet     *[][]float64
	expectedResults *[]float64

	Parameters []float64 `json:"theta"`
}

// NewLeastSquares returns a pointer to the linear model
// initialized with the learning rate alpha, the training
// set trainingSet, and the expected results upon which to
// use the dataset to train, expectedResults.
func NewLeastSquares(alpha float64, trainingSet *[][]float64, expectedResults *[]float64) (*LeastSquares, error) {
	if len(*trainingSet) == 0 {
		return nil, fmt.Errorf("Error: length of given training set is 0! Need data!")
	}

	return &LeastSquares{
		alpha:           alpha,
		trainingSet:     trainingSet,
		expectedResults: expectedResults,

		// initialize θ as the zero vector (that is,
		// the vector of all zeros)
		Parameters: make([]float64, len((*trainingSet)[0])+1),
	}, nil
}

// LearningRate returns the learning rate α for gradient
// descent to optimize the model. Could vary as a function
// of something else later, potentially.
func (l *LeastSquares) LearningRate() float64 {
	return l.alpha
}

// Predict takes in a variable x (an array of floats,) and
// finds the value of the hypothesis function given the
// current parameter vector θ
func (l *LeastSquares) Predict(x []float64) (float64, error) {
	if len(x)+1 != len(l.Parameters) {
		return 0, fmt.Errorf("Error: Parameter vector should be 1 longer than input vector!\n\tLength of x given: %v\n\tLength of parameters: %v", len(x), len(l.Parameters))
	}

	// include constant term in sum
	sum := l.Parameters[0]

	for i := range x {
		sum += x[i] * l.Parameters[i+1]
	}

	return sum, nil
}

// Learn takes the struct's dataset and expected results and runs
// batch gradient descent on them, optimizing theta so you can
// predict based on those results
func (l *LeastSquares) Learn() {
	examples := len(*l.trainingSet)
	if examples == 0 {
		fmt.Printf("\nERROR: Attempting to learn with no training examples!\n\n")
		return
	}

	fmt.Printf("Training:\n\tModel: Linear Least Squares\n\tOptimization Method: Batch Gradient Descent\n\tTraining Examples: %v\n\tFeatures: %v\n...\n\n", examples, len((*l.trainingSet)[0]))

	err := base.GradientDescent(l)
	if err != nil {
		fmt.Printf("\nERROR: Error while learing –\n\t%v\n\n", err)
	}

	fmt.Printf("Training Completed.\n%v\n\n", l)
}

// String implements the fmt interface for clean printing. Here
// we're using it to print the model as the equation h(θ)=...
// where h is the linear hypothesis model
func (l *LeastSquares) String() string {
	features := len(l.Parameters) - 1
	if len(l.Parameters) == 0 {
		fmt.Printf("ERROR: Attempting to print model with the 0 vector as it's parameter vector! Train first!")
	}
	var buffer bytes.Buffer

	buffer.WriteString(fmt.Sprintf("h(θ) = %.3f + ", l.Parameters[0]))

	for i := 1; i < features+1; i++ {
		buffer.WriteString(fmt.Sprintf("%.3f(x[%d])", l.Parameters[i], i))

		if i != features {
			buffer.WriteString(fmt.Sprintf(" + "))
		}
	}

	return buffer.String()
}

// Dj returns the partial derivative of the cost function J(θ)
// with respect to theta[j] where theta is the parameter vector
// associated with our hypothesis function Predict (upon which
// we are optimizing
func (l *LeastSquares) Dj(j int) (float64, error) {
	// use batch method
	var sum float64

	for i := range *l.trainingSet {
		prediction, err := l.Predict((*l.trainingSet)[i])
		if err != nil {
			return 0, err
		}

		sum += ((*l.expectedResults)[i] - prediction) * (*l.trainingSet)[i][j]
	}

	return sum, nil
}

// J returns the Least Squares cost function of the given linear
// model, for use in gradient descent.
func (l *LeastSquares) J() (float64, error) {
	var sum float64

	for i := range *l.trainingSet {
		prediction, err := l.Predict((*l.trainingSet)[i])
		if err != nil {
			return 0, err
		}

		sum += ((*l.expectedResults)[i] - prediction) * ((*l.expectedResults)[i] - prediction)
	}

	return sum / 2, nil
}

// Theta returns the parameter vector θ for use in persisting
// the model, and optimizing the model through gradient descent
// ( or other methods like Newton's Method)
func (l *LeastSquares) Theta() *[]float64 {
	return &l.Parameters
}
