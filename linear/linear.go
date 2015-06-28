package linear

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/cdipaolo/goml/base"
)

// LeastSquares implements a standard linear regression model
// with a Least Squares cost function.
//
// https://en.wikipedia.org/wiki/Least_squares
//
// The model uses gradient descent, NOT regular equations.
type LeastSquares struct {
	// alpha and maxIterations are used only for
	// GradientAscent during learning. If maxIterations
	// is 0, then GradientAscent will run until the
	// algorithm detects convergance
	alpha         float64
	maxIterations int

	// trainingSet and expectedResults are the
	// 'x', and 'y' of the data, expressed as
	// vectors, that the model can optimize from
	trainingSet     [][]float64
	expectedResults []float64

	Parameters []float64 `json:"theta"`
}

// NewLeastSquares returns a pointer to the linear model
// initialized with the learning rate alpha, the training
// set trainingSet, and the expected results upon which to
// use the dataset to train, expectedResults.
func NewLeastSquares(alpha float64, maxIterations int, trainingSet [][]float64, expectedResults []float64) *LeastSquares {
	var params []float64
	if trainingSet == nil || len(trainingSet) == 0 {
		params = []float64{}
	} else {
		params = make([]float64, len((trainingSet)[0])+1)
	}

	return &LeastSquares{
		alpha:         alpha,
		maxIterations: maxIterations,

		trainingSet:     trainingSet,
		expectedResults: expectedResults,

		// initialize θ as the zero vector (that is,
		// the vector of all zeros)
		Parameters: params,
	}
}

// UpdateTrainingSet takes in a new training set (variable x)
// as well as a new result set (y). This could be useful if
// you want to retrain a model starting with the parameter
// vector of a previous training session, but most of the time
// wouldn't be used.
func (l *LeastSquares) UpdateTrainingSet(trainingSet [][]float64, expectedResults []float64) error {
	if len(trainingSet) == 0 {
		return fmt.Errorf("Error: length of given training set is 0! Need data!")
	}
	if len(expectedResults) == 0 {
		return fmt.Errorf("Error: length of given result data set is 0! Need expected results!")
	}

	l.trainingSet = trainingSet
	l.expectedResults = expectedResults

	return nil
}

// UpdateLearningRate set's the learning rate of the model
// to the given float64.
func (l *LeastSquares) UpdateLearningRate(a float64) {
	l.alpha = a
}

// LearningRate returns the learning rate α for gradient
// descent to optimize the model. Could vary as a function
// of something else later, potentially.
func (l *LeastSquares) LearningRate() float64 {
	return l.alpha
}

// MaxIterations returns the number of maximum iterations
// the model will go through in GradientAscent, in the
// worst case
func (l *LeastSquares) MaxIterations() int {
	return l.maxIterations
}

// Predict takes in a variable x (an array of floats,) and
// finds the value of the hypothesis function given the
// current parameter vector θ
func (l *LeastSquares) Predict(x []float64) ([]float64, error) {
	if len(x)+1 != len(l.Parameters) {
		return nil, fmt.Errorf("Error: Parameter vector should be 1 longer than input vector!\n\tLength of x given: %v\n\tLength of parameters: %v\n", len(x), len(l.Parameters))
	}

	// include constant term in sum
	sum := l.Parameters[0]

	for i := range x {
		sum += x[i] * l.Parameters[i+1]
	}

	return []float64{sum}, nil
}

// Learn takes the struct's dataset and expected results and runs
// batch gradient descent on them, optimizing theta so you can
// predict based on those results
func (l *LeastSquares) Learn() error {
	if l.trainingSet == nil || l.expectedResults == nil {
		err := fmt.Errorf("ERROR: Attempting to learn with no training examples!\n")
		fmt.Printf(err.Error())
		return err
	}

	examples := len(l.trainingSet)
	if examples == 0 || len(l.trainingSet[0]) == 0 {
		err := fmt.Errorf("ERROR: Attempting to learn with no training examples!\n")
		fmt.Printf(err.Error())
		return err
	}
	if len(l.expectedResults) == 0 {
		err := fmt.Errorf("ERROR: Attempting to learn with no expected results! This isn't an unsupervised model!! You'll need to include data before you learn :)\n")
		fmt.Printf(err.Error())
		return err
	}

	fmt.Printf("Training:\n\tModel: Linear Least Squares\n\tOptimization Method: Batch Gradient Descent\n\tTraining Examples: %v\n\tFeatures: %v\n...\n\n", examples, len(l.trainingSet[0]))

	err := base.GradientAscent(l)
	if err != nil {
		fmt.Printf("\nERROR: Error while learning –\n\t%v\n\n", err)
		return err
	}

	fmt.Printf("Training Completed.\n%v\n\n", l)
	return nil
}

// String implements the fmt interface for clean printing. Here
// we're using it to print the model as the equation h(θ)=...
// where h is the linear hypothesis model
func (l *LeastSquares) String() string {
	features := len(l.Parameters) - 1
	if len(l.Parameters) == 0 {
		fmt.Printf("ERROR: Attempting to print model with the 0 vector as it's parameter vector! Train first!\n")
	}
	var buffer bytes.Buffer

	buffer.WriteString(fmt.Sprintf("h(θ) = %.3f + ", l.Parameters[0]))

	length := features + 1
	for i := 1; i < length; i++ {
		buffer.WriteString(fmt.Sprintf("%.5f(x[%d])", l.Parameters[i], i))

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
	if j > len(l.Parameters)-1 {
		return 0, fmt.Errorf("J (%v) would index out of the bounds of the training set data (len: %v)", j, len(l.Parameters))
	}

	var sum float64

	for i := range l.trainingSet {
		prediction, err := l.Predict(l.trainingSet[i])
		if err != nil {
			return 0, err
		}

		// account for constant term
		// x is x[i][j] via Andrew Ng's terminology
		var x float64
		if j == 0 {
			x = 1
		} else {
			x = l.trainingSet[i][j-1]
		}

		sum += (l.expectedResults[i] - prediction[0]) * x
	}

	return sum, nil
}

// J returns the Least Squares cost function of the given linear
// model, for use in gradient descent.
func (l *LeastSquares) J() (float64, error) {
	var sum float64

	for i := range l.trainingSet {
		prediction, err := l.Predict(l.trainingSet[i])
		if err != nil {
			return 0, err
		}

		sum += (l.expectedResults[i] - prediction[0]) * (l.expectedResults[i] - prediction[0])
	}

	return sum / 2, nil
}

// Theta returns the parameter vector θ for use in persisting
// the model, and optimizing the model through gradient descent
// ( or other methods like Newton's Method)
func (l *LeastSquares) Theta() []float64 {
	return l.Parameters
}

// PersistToFile takes in an absolute filepath and saves the
// parameter vector θ to the file, which can be restored later.
// The function will take paths from the current directory, but
// functions
//
// The data is stored as JSON because it's one of the most
// efficient storage method (you only need one comma extra
// per feature + two brackets, total!) And it's extendable.
func (l *LeastSquares) PersistToFile(path string) error {
	if path == "" {
		return fmt.Errorf("ERROR: you just tried to persist your model to a file with no path!! That's a no-no. Try it with a valid filepath!")
	}

	bytes, err := json.Marshal(l.Parameters)
	if err != nil {
		return err
	}

	err = ioutil.WriteFile(path, bytes, os.ModePerm)
	if err != nil {
		return err
	}

	return nil
}

// RestoreFromFile takes in a path to a parameter vector theta
// and assigns the model it's operating on's parameter vector
// to that.
//
// The path must ba an absolute path or a path from the current
// directory
//
// This would be useful in persisting data between running
// a model on data, or for graphing a dataset with a fit in
// another framework like Julia/Gadfly.
func (l *LeastSquares) RestoreFromFile(path string) error {
	if path == "" {
		return fmt.Errorf("ERROR: you just tried to restore your model from a file with no path! That's a no-no. Try it with a valid filepath!")
	}

	bytes, err := ioutil.ReadFile(path)
	if err != nil {
		return err
	}

	err = json.Unmarshal(bytes, &l.Parameters)
	if err != nil {
		return err
	}

	return nil
}
