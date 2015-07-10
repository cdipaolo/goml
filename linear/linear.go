// Package linear implements commonly used
// General Linear Models.
//
// https://en.wikipedia.org/wiki/General_linear_model
//
// Models implemented as of yet include:
//     - Ordinary Least Squares
//     - Logistic Regression
//
// General Usage:
// Find the model you want to use. Then
// find the 'NewXXXXXX' function, such as
//     func NewLeastSquares(method base.OptimizationMethod, alpha, regularization float64, maxIterations int, trainingSet [][]float64, expectedResults []float64) *LeastSquares
//
// load in the given parameters, then run
//     func Learn() error
//
// Now you can predict off of the model!
//     func Predict([]float64) ([]float64, error)
//
// Full example assuming testX is of type
// [][]float64 and testY is of type []float64
// where there are 2 features being inputted
// (ie. the size of a house and the number of
// bedrooms being given as x[0] and x[1],
// respectively.) textY[i] should be the observed
// result of the inputs testX[i].:
//
// Example Model Usage (Batch Ordinary Least Squares):
//
//     // optimization method: Batch Gradient Ascent
//     // Learning rate: 1e-4
//     // Regulatization term: 6
//     // Max Iterations: 800
//     // Dataset to learn fron: testX
//     // Expected results dataset: testY
//     model := NewLeastSquares(base.BatchGA, 1e-4, 6, 800, testX, testY)
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
package linear

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
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
	// algorithm detects convergance.
	//
	// regularization is used as the regularization
	// term to avoid overfitting within regression.
	// Having a regularization term of 0 is like having
	// _no_ data regularization. The higher the term,
	// the greater the bias on the regression
	alpha          float64
	regularization float64
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
}

// NewLeastSquares returns a pointer to the linear model
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
//     // Max Iterations: 800
//     // Dataset to learn fron: testX
//     // Expected results dataset: testY
//     model := NewLeastSquares(base.StochasticGA, 1e-4, 6, 800, testX, testY)
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
func NewLeastSquares(method base.OptimizationMethod, alpha, regularization float64, maxIterations int, trainingSet [][]float64, expectedResults []float64, features ...int) *LeastSquares {
	var params []float64
	if len(features) != 0 {
		params = make([]float64, features[0]+1)
	} else if trainingSet == nil || len(trainingSet) == 0 {
		params = []float64{}
	} else {
		params = make([]float64, len(trainingSet[0])+1)
	}

	return &LeastSquares{
		alpha:          alpha,
		regularization: regularization,
		maxIterations:  maxIterations,

		method: method,

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

// Examples returns the number of training examples (m)
// that the model currently is training from.
func (l *LeastSquares) Examples() int {
	return len(l.trainingSet)
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

	fmt.Printf("Training:\n\tModel: Logistic (Binary) Classification\n\tOptimization Method: %v\n\tTraining Examples: %v\n\tFeatures: %v\n\tLearning Rate α: %v\n\tRegularization Parameter λ: %v\n...\n\n", l.method, examples, len(l.trainingSet[0]), l.alpha, l.regularization)

	var err error
	if l.method == base.BatchGA {
		err = base.GradientAscent(l)
	} else if l.method == base.StochasticGA {
		err = base.StochasticGradientAscent(l)
	} else {
		err = fmt.Errorf("Chose a training method not implemented for LeastSquares regression")
	}

	if err != nil {
		fmt.Printf("\nERROR: Error while learning –\n\t%v\n\n", err)
		return err
	}

	fmt.Printf("Training Completed.\n%v\n\n", l)
	return nil
}

// OnlineLearn runs similar to using a fixed dataset with
// Stochastic Gradient Descent, but it handles data by
// passing it as a channal, and returns errors through
// a channel, which lets it run responsive to inputted data
// from outside the model itself (like using data from the
// stock market at timed intervals or using realtime data
// about the weather.)
//
// The onUpdate callback is called whenever the parameter
// vector theta is changed, so you are able to persist the
// model with the most up to date vector at all times (you
// could persist to a database within the callback, for
// example.) Don't worry about it taking too long and blocking,
// because the callback is spawned into another goroutine.
//
// NOTE that this function is suggested to run in it's own
// goroutine, or at least is designed as such.
//
// NOTE part 2: You can pass in an empty dataset, so long
// as it's not nil, and start pushing after.
//
// NOTE part 3: each example is only looked at as it goes
// through the channel, so if you want to have each example
// looked at more than once you must manually pass the data
// yourself.
//
// Example Online Linear Least Squares:
//
// 	   // create the channel of data and errors
//     stream := make(chan base.Datapoint, 100)
//     errors := make(chan error)
//
//     // notice how we are adding another integer
//     // to the end of the NewLogistic call. This
//     // tells the model to use that number of features
//     // (4) in leu of finding that from the dataset
//     // like you would with batch/stochastic GD
//     //
//     // Also – the 'base.StochasticGA' doesn't affect
//     // anything. You could put batch.
//     model := NewLeastSquares(base.StochasticGA, .0001, 0, 0, nil, nil, 4)
//
//     go model.OnlineLearn(errors, stream, func(theta []float64) {
//         // do something with the new theta (persist
//         // to database?) in here.
//     })
//
//     go func() {
//         for iterations := 0; iterations < 20; iterations++ {
//             for i := -200.0; abs(i) > 1; i *= -0.75 {
//                 for j := -200.0; abs(j) > 1; j *= -0.75 {
//                     for k := -200.0; abs(k) > 1; k *= -0.75 {
//                         for l := -200.0; abs(l) > 1; l *= -0.75 {
//                              stream <- base.Datapoint{
//                                  X: []float64{i, j, k, l},
//                                  Y: []float64{i/2 + 2*k - 4*j + 2*l + 3},
//                              }
//                         }
//                     }
//                 }
//             }
//         }
//
//       // close the dataset to tell the model
//       // to stop learning when it finishes reading
//       // what's left in the channel
//       close(stream)
//     }()
//
//     // this will block until the error
//     // channel is closed in the learning
//     // function (it will, don't worry!)
//     for {
//         err, more := <-errors
//         if err != nil {
//             panic("THERE WAS AN ERROR!!! RUN!!!!")
//       }
//         if !more {
//             break
//         }
//     }
//
//     // Below here all the learning is completed
//
//     // predict like usual
//     guess, err = model.Predict([]float64{42,6,10,-32})
//     if err != nil {
//         panic("AAAARGGGH! SHIVER ME TIMBERS! THESE ROTTEN SCOUNDRELS FOUND AN ERROR!!!")
//     }
func (l *LeastSquares) OnlineLearn(errors chan error, dataset chan base.Datapoint, onUpdate func([]float64)) {
	if dataset == nil {
		err := fmt.Errorf("ERROR: Attempting to learn with a nil data stream!\n")
		fmt.Printf(err.Error())
		errors <- err
		close(errors)
		return
	}

	fmt.Printf("Training:\n\tModel: Ordinary Least Squares Regression\n\tOptimization Method: Online Stochastic Gradient Descent\n\tFeatures: %v\n\tLearning Rate α: %v\n...\n\n", len(l.Parameters), l.alpha)

	for {
		point, more := <-dataset
		if more {
			if len(point.Y) != 1 {
				errors <- fmt.Errorf("ERROR: point.Y must have a length of 1. Point: %v", point)
			}

			newTheta := make([]float64, len(l.Parameters))
			for j := range l.Parameters {

				// find the gradient using the point
				// from the channel (different than
				// calling from the dataset so we need
				// to have a new function instead of calling
				// Dij(i, j))
				dj, err := func(point base.Datapoint, j int) (float64, error) {
					prediction, err := l.Predict(point.X)
					if err != nil {
						return 0, err
					}

					// account for constant term
					// x is x[i][j] via Andrew Ng's terminology
					var x float64
					if j == 0 {
						x = 1
					} else {
						x = point.X[j-1]
					}

					var gradient float64
					gradient = (point.Y[0] - prediction[0]) * x

					// add in the regularization term
					// λ*θ[j]
					//
					// notice that we don't count the
					// constant term
					if j != 0 {
						gradient += l.regularization * l.Parameters[j]
					}

					return gradient, nil
				}(point, j)
				if err != nil {
					errors <- err
					continue
				}

				newTheta[j] = l.Parameters[j] + l.alpha*dj
			}

			// now simultaneously update Theta
			for j := range l.Parameters {
				newθ := newTheta[j]
				if math.IsInf(newθ, 0) || math.IsNaN(newθ) {
					errors <- fmt.Errorf("Sorry dude! Learning diverged. Some value of the parameter vector theta is ±Inf or NaN")
					continue
				}
				l.Parameters[j] = newθ
			}

			go onUpdate(l.Parameters)

		} else {
			fmt.Printf("Training Completed.\n%v\n\n", l)
			close(errors)
			return
		}
	}
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
func (l *LeastSquares) Dij(i int, j int) (float64, error) {
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

	var gradient float64
	gradient = (l.expectedResults[i] - prediction[0]) * x

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
func (l *LeastSquares) J() (float64, error) {
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
		return fmt.Errorf("ERROR: you just tried to persist your model to a file with no path!! That's a no-no. Try it with a valid filepath")
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
		return fmt.Errorf("ERROR: you just tried to restore your model from a file with no path! That's a no-no. Try it with a valid filepath")
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
