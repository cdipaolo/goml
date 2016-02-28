package linear

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/cdipaolo/goml/base"
)

// LocalLinear implements a locally weighted
// linear least squares regression.
//
// https://en.wikipedia.org/wiki/Least_squares
// http://cs229.stanford.edu/notes/cs229-notes1.pdf
//
// Note that this is not modeled to work in an
// online setting, so the model does not implement
// that interface. Also, because a new hypothesis
// is needed for every point you try to predict,
// there is no 'Learn' function. Instead, when
// calling predict the model first learns from the
// data set with weights set with respect to the
// given input, then returns the trained hypothesis
// when evaluated at the given input.
//
// While that may sound really inefficient, sometimes
// this model can perform well with little tweaking,
// especially if you're working with a smaller data
// set to train off of. Andrew Ng said in one of his
// Stanford CS229 lectures that Locally Weighted
// Linear Regression is one of his mentor's 'favourite'
// 'off-the-shelf' learning algorithm. Obviously
// model selection plays a large role in this.
//
// NOTE that there is no file persistance of this
// model because you need to retrain at the time
// of every prediction anyway.
//
// Example Locally Weighted Linear Regression Usage:
//
//     x := [][]float64{}
//     y := []float64{}
//
//     // throw in some junk points which
//     // should be more-or-less ignored
//     // by the weighting
//     for i := -70.0; i < -65; i += 2 {
//         for j := -70.0; j < -65; j += 2 {
//             x = append(x, []float64{i, j})
//             y = append(y, 20*(rand.Float64()-0.5))
//         }
//     }
//     for i := 65.0; i < 70; i += 2 {
//         for j := 65.0; j < 70; j += 2 {
//             x = append(x, []float64{i, j})
//             y = append(y, 20*(rand.Float64()-0.5))
//         }
//     }
//
//     // put in some linear points
//     for i := -20.0; i < 20; i++ {
//         for j := -20.0; j < 20; j++ {
//             x = append(x, []float64{i, j})
//             y = append(y, 5*i-5*j-10)
//         }
//     }
//
//     model := NewLocalLinear(base.StochasticGA, 1e-4, 0, 0.75, 1500, x, y)
//
//     // now when you predict it'll train off the
//     // dataset, weighting points closer to the
//     // targer evaluation more, then return
//     // the prediction.
//     guess, err := model.Predict([]float64{10.0, -13.666})
type LocalLinear struct {
	// alpha and maxIterations are used only for
	// GradientAscent during learning. If maxIterations
	// is 0, then GradientAscent will run until the
	// algorithm detects convergance.
	//
	// regularization is used as the regularization
	// term to avoid overfitting within regression.
	// Having a regularization term of 0 is like having
	// _no_ data regularization. The higher the term,
	// the greater the bias on the regression
	alpha          float64
	regularization float64
	bandwidth      float64
	maxIterations  int

	// method is the optimization method used when training
	// the model
	method base.OptimizationMethod

	// trainingSet and expectedResults are the
	// 'x', and 'y' of the data, expressed as
	// vectors, that the model can optimize from
	trainingSet     [][]float64
	expectedResults []float64

	Parameters []float64 `json:"theta"`

	// Output is the io.Writer used for logging
	// and printing. Defaults to os.Stdout.
	Output io.Writer
}

// NewLocalLinear returns a pointer to the linear model
// initialized with the learning rate alpha, the training
// set trainingSet, and the expected results upon which to
// use the dataset to train, expectedResults.
//
// if you're passing in no training set directly because you want
// to learn using the online method then just declare the number of
// features (it's an integer) as an extra arg after the rest
// of the arguments
//
// Example Least Squares (Stochastic GA):
//
//     // optimization method: Stochastic Gradient Ascent
//     // Learning rate: 1e-4
//     // Regulatization term: 6
//     // Weight Bandwidth: 1.0
//     // Max Iterations: 800
//     // Dataset to learn fron: testX
//     // Expected results dataset: testY
//     model := NewLocalLinear(base.StochasticGA, 1e-4, 6, 1.0, 800, testX, testY)
//
//     err := model.Learn()
//     if err != nil {
//         panic("SOME ERROR!! RUN!")
//     }
//
//     // now I want to predict off of this
//     // Ordinary Least Squares model!
//     guess, err = model.Predict([]float64{10000,6})
//     if err != nil {
//         panic("AAAARGGGH! SHIVER ME TIMBERS! THESE ROTTEN SCOUNDRELS FOUND AN ERROR!!!")
//     }
func NewLocalLinear(method base.OptimizationMethod, alpha, regularization, bandwidth float64, maxIterations int, trainingSet [][]float64, expectedResults []float64) *LocalLinear {
	var params []float64
	if trainingSet == nil || len(trainingSet) == 0 {
		params = []float64{}
	} else {
		params = make([]float64, len(trainingSet[0])+1)
	}

	return &LocalLinear{
		alpha:          alpha,
		regularization: regularization,
		bandwidth:      bandwidth,
		maxIterations:  maxIterations,

		method: method,

		trainingSet:     trainingSet,
		expectedResults: expectedResults,

		// initialize θ as the zero vector (that is,
		// the vector of all zeros)
		Parameters: params,

		Output: os.Stdout,
	}
}

// UpdateTrainingSet takes in a new training set (variable x)
// as well as a new result set (y). This could be useful if
// you want to retrain a model starting with the parameter
// vector of a previous training session, but most of the time
// wouldn't be used.
func (l *LocalLinear) UpdateTrainingSet(trainingSet [][]float64, expectedResults []float64) error {
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
func (l *LocalLinear) UpdateLearningRate(a float64) {
	l.alpha = a
}

// LearningRate returns the learning rate α for gradient
// descent to optimize the model. Could vary as a function
// of something else later, potentially.
func (l *LocalLinear) LearningRate() float64 {
	return l.alpha
}

// Examples returns the number of training examples (m)
// that the model currently is training from.
func (l *LocalLinear) Examples() int {
	return len(l.trainingSet)
}

// MaxIterations returns the number of maximum iterations
// the model will go through in GradientAscent, in the
// worst case
func (l *LocalLinear) MaxIterations() int {
	return l.maxIterations
}

// Predict takes in a variable x (an array of floats,) and
// finds the value of the hypothesis function given the
// current parameter vector θ
//
// if normalize is given as true, then the input will
// first be normalized to unit length. Only use this if
// you trained off of normalized inputs and are feeding
// an un-normalized input
func (l *LocalLinear) Predict(x []float64, normalize ...bool) ([]float64, error) {
	if len(x)+1 != len(l.Parameters) {
		err := fmt.Errorf("ERROR: Parameter vector should be 1 longer than input vector!\n\tLength of x given: %v\n\tLength of parameters: %v\n", len(x), len(l.Parameters))
		print(err.Error())
		return nil, err
	}

	norm := len(normalize) != 0 && normalize[0]
	if norm {
		base.NormalizePoint(x)
	}

	if l.trainingSet == nil || l.expectedResults == nil {
		err := fmt.Errorf("ERROR: Attempting to learn with no training examples!\n")
		print(err.Error())
		return nil, err
	}

	examples := len(l.trainingSet)
	if examples == 0 || len(l.trainingSet[0]) == 0 {
		err := fmt.Errorf("ERROR: Attempting to learn with no training examples!\n")
		print(err.Error())
		return nil, err
	}
	if len(l.expectedResults) == 0 {
		err := fmt.Errorf("ERROR: Attempting to learn with no expected results! This isn't an unsupervised model!! You'll need to include data before you learn :)\n")
		print(err.Error())
		return nil, err
	}

	fmt.Fprintf(l.Output, "Training:\n\tModel: Locally Weighted Linear Regression\n\tOptimization Method: %v\n\tCenter Point: %v\n\tTraining Examples: %v\n\tFeatures: %v\n\tLearning Rate α: %v\n\tRegularization Parameter λ: %v\n...\n\n", l.method, x, examples, len(l.trainingSet[0]), l.alpha, l.regularization)

	var iter int
	features := len(l.Parameters)

	if l.method == base.BatchGA {
		for ; iter < l.maxIterations; iter++ {
			newTheta := make([]float64, features)
			for j := range l.Parameters {
				dj, err := l.Dj(x, j)
				if err != nil {
					return nil, err
				}

				newTheta[j] = l.Parameters[j] + l.alpha*dj
			}

			// now simultaneously update Theta
			for j := range l.Parameters {
				newθ := newTheta[j]
				if math.IsInf(newθ, 0) || math.IsNaN(newθ) {
					return nil, fmt.Errorf("Sorry! Learning diverged. Some value of the parameter vector theta is ±Inf or NaN")
				}
				l.Parameters[j] = newθ
			}
		}
	} else if l.method == base.StochasticGA {
		for ; iter < l.maxIterations; iter++ {
			newTheta := make([]float64, features)
			for i := 0; i < examples; i++ {
				for j := range l.Parameters {
					dj, err := l.Dij(x, i, j)
					if err != nil {
						return nil, err
					}

					newTheta[j] = l.Parameters[j] + l.alpha*dj
				}

				// now simultaneously update Theta
				for j := range l.Parameters {
					newθ := newTheta[j]
					if math.IsInf(newθ, 0) || math.IsNaN(newθ) {
						return nil, fmt.Errorf("Sorry! Learning diverged. Some value of the parameter vector theta is ±Inf or NaN")
					}
					l.Parameters[j] = newθ
				}
			}
		}
	} else {
		return nil, fmt.Errorf("Chose a training method not implemented for LocalLinear regression")
	}

	fmt.Fprintf(l.Output, "Training Completed. Went through %v iterations.\n%v\n\n", iter, l)

	// include constant term in sum
	sum := l.Parameters[0]

	for i := range x {
		sum += x[i] * l.Parameters[i+1]
	}

	return []float64{sum}, nil
}

// String implements the fmt interface for clean printing. Here
// we're using it to print the model as the equation h(θ)=...
// where h is the linear hypothesis model
func (l *LocalLinear) String() string {
	features := len(l.Parameters) - 1
	if len(l.Parameters) == 0 {
		fmt.Fprintf(l.Output, "ERROR: Attempting to print model with the 0 vector as it's parameter vector! Train first!\n")
	}
	var buffer bytes.Buffer

	buffer.WriteString(fmt.Sprintf("h(θ,x) = %.3f + ", l.Parameters[0]))

	length := features + 1
	for i := 1; i < length; i++ {
		buffer.WriteString(fmt.Sprintf("%.5f(x[%d])", l.Parameters[i], i))

		if i != features {
			buffer.WriteString(fmt.Sprintf(" + "))
		}
	}

	return buffer.String()
}

// weight corresponds to the weight given between
// two datapoints (based on how 'far apart' they
// are.)
//
// w[i] = exp(-1 * |x[i] - x|^2 / 2σ^2)
func (l *LocalLinear) weight(X []float64, x []float64) float64 {
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

	return math.Exp(-1 * diff / (2 * l.bandwidth * l.bandwidth))
}

// Dj returns the partial derivative of the cost function J(θ)
// with respect to theta[j] where theta is the parameter vector
// associated with our hypothesis function Predict (upon which
// we are optimizing
func (l *LocalLinear) Dj(input []float64, j int) (float64, error) {
	if j > len(l.Parameters)-1 {
		return 0, fmt.Errorf("J (%v) would index out of the bounds of the training set data (len: %v)", j, len(l.Parameters))
	}
	if len(input) != len(l.Parameters)-1 {
		return 0, fmt.Errorf("Length of input x (%v) should be one less than the length of the parameter vector (len: %v)", len(input), len(l.Parameters))
	}

	var sum float64

	for i := range l.trainingSet {
		prediction := l.Parameters[0]
		for k := 1; k < len(l.Parameters); k++ {
			prediction += l.Parameters[k] * input[k-1]
		}

		// account for constant term
		// x is x[i][j] via Andrew Ng's terminology
		var x float64
		if j == 0 {
			x = 1
		} else {
			x = l.trainingSet[i][j-1]
		}

		sum += l.weight(l.trainingSet[i], input) * (l.expectedResults[i] - prediction) * x
	}

	// add in the regularization term
	// λ*θ[j]
	//
	// notice that we don't count the
	// constant term
	if j != 0 {
		sum += l.regularization * l.Parameters[j]
	}

	return sum, nil
}

// Dij returns the derivative of the cost function
// J(θ) with respect to the j-th parameter of
// the hypothesis, θ[j], for the training example
// x[i]. Used in Stochastic Gradient Descent.
//
// assumes that i,j is within the bounds of the
// data they are looking up! (because this is getting
// called so much, it needs to be efficient with
// comparisons)
func (l *LocalLinear) Dij(input []float64, i, j int) (float64, error) {
	if j > len(l.Parameters)-1 || i > len(l.trainingSet)-1 {
		return 0, fmt.Errorf("j (%v) or i (%v) would index out of the bounds of the training set data (len: %v)", j, i, len(l.Parameters))
	}
	if len(input) != len(l.Parameters)-1 {
		return 0, fmt.Errorf("Length of input x (%v) should be one less than the length of the parameter vector (len: %v)", len(input), len(l.Parameters))
	}

	prediction := l.Parameters[0]
	for k := 1; k < len(l.Parameters); k++ {
		prediction += l.Parameters[k] * input[k-1]
	}

	// account for constant term
	// x is x[i][j] via Andrew Ng's terminology
	var x float64
	if j == 0 {
		x = 1
	} else {
		x = l.trainingSet[i][j-1]
	}

	var gradient float64
	gradient = l.weight(l.trainingSet[i], input) * (l.expectedResults[i] - prediction) * x

	// add in the regularization term
	// λ*θ[j]
	//
	// notice that we don't count the
	// constant term
	if j != 0 {
		gradient += l.regularization * l.Parameters[j]
	}

	return gradient, nil
}

// J returns the Least Squares cost function of the given linear
// model. Could be usefull in testing convergance
func (l *LocalLinear) J() (float64, error) {
	var sum float64

	for i := range l.trainingSet {
		prediction, err := l.Predict(l.trainingSet[i])
		if err != nil {
			return 0, err
		}

		sum += (l.expectedResults[i] - prediction[0]) * (l.expectedResults[i] - prediction[0])
	}

	// add regularization term!
	//
	// notice that the constant term doesn't matter
	for i := 1; i < len(l.Parameters); i++ {
		sum += l.regularization * l.Parameters[i] * l.Parameters[i]
	}

	return sum / float64(2*len(l.trainingSet)), nil
}
