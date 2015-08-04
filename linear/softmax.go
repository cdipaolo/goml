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

// Softmax represents a softmax classification model
// in 'k' demensions. It is generally thought of as
// a generalization of the Logistic Regression model.
// Prediction will return a vector ([]float64) that
// corresponds to the probabilty (where i is the index)
// that the inputted features is 'i'. Softmax classification
// operates assuming the Multinomial probablility
// distribution of data.
//
// TODO: add wikipedia link
//
// Expected results expects an 'integer' (it's still
// passed as a float64) between 0 and k-1. K must be
// passed when creating the model.
type Softmax struct {
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

	// k is the dimension of classification (the number
	// of possible outcomes)
	k int

	// method is the optimization method used when training
	// the model
	method base.OptimizationMethod

	// trainingSet and expectedResults are the
	// 'x', and 'y' of the data, expressed as
	// vectors, that the model can optimize from
	trainingSet     [][]float64
	expectedResults []float64

	Parameters [][]float64 `json:"theta"`
}

func abs(x float64) float64 {
	if x < 0 {
		return -1 * x
	}

	return x
}

// NewSoftmax takes in a learning rate alpha, a regularization
// parameter value (0 means no regularization, higher value
// means higher bias on the model,) the maximum number of
// iterations the data can go through in gradient descent,
// as well as a training set and expected results for that
// training set.
func NewSoftmax(method base.OptimizationMethod, alpha, regularization float64, k, maxIterations int, trainingSet [][]float64, expectedResults []float64, features ...int) *Softmax {
	params := make([][]float64, k)

	if len(features) != 0 {
		for i := range params {
			params[i] = make([]float64, features[0]+1)
		}
	} else if trainingSet == nil || len(trainingSet) == 0 {
		for i := range params {
			params[i] = []float64{}
		}
	} else {
		for i := range params {
			params[i] = make([]float64, len((trainingSet)[0])+1)
		}
	}

	return &Softmax{
		alpha:          alpha,
		regularization: regularization,
		maxIterations:  maxIterations,

		k: k,

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
func (s *Softmax) UpdateTrainingSet(trainingSet [][]float64, expectedResults []float64) error {
	if len(trainingSet) == 0 {
		return fmt.Errorf("Error: length of given training set is 0! Need data!")
	}
	if len(expectedResults) == 0 {
		return fmt.Errorf("Error: length of given result data set is 0! Need expected results!")
	}

	s.trainingSet = trainingSet
	s.expectedResults = expectedResults

	return nil
}

// UpdateLearningRate set's the learning rate of the model
// to the given float64.
func (s *Softmax) UpdateLearningRate(a float64) {
	s.alpha = a
}

// LearningRate returns the learning rate α for gradient
// descent to optimize the model. Could vary as a function
// of something else later, potentially.
func (s *Softmax) LearningRate() float64 {
	return s.alpha
}

// Examples returns the number of training examples (m)
// that the model currently is training from.
func (s *Softmax) Examples() int {
	return len(s.trainingSet)
}

// MaxIterations returns the number of maximum iterations
// the model will go through in GradientAscent, in the
// worst case
func (s *Softmax) MaxIterations() int {
	return s.maxIterations
}

// Predict takes in a variable x (an array of floats,) and
// finds the value of the hypothesis function given the
// current parameter vector θ
func (s *Softmax) Predict(x []float64, normalize ...bool) ([]float64, error) {
	if len(s.Parameters) != 0 && len(x)+1 != len(s.Parameters[0]) {
		return nil, fmt.Errorf("Error: Parameter vector should be 1 longer than input vector!\n\tLength of x given: %v\n\tLength of parameters: %v (len(theta[0]) = %v)\n", len(x), len(s.Parameters), len(s.Parameters[0]))
	}

	if len(normalize) != 0 && normalize[0] {
		base.NormalizePoint(x)
	}

	result := make([]float64, s.k)
	var denom float64

	for i := 0; i < s.k; i++ {
		// include constant term in sum
		sum := s.Parameters[i][0]

		for j := range x {
			sum += x[j] * s.Parameters[i][j+1]
		}

		result[i] = math.Exp(sum)
		denom += result[i]
	}

	for i := range result {
		result[i] /= denom
	}

	return result, nil
}

// Learn takes the struct's dataset and expected results and runs
// gradient descent on them, optimizing theta so you can
// predict accurately based on those results
func (s *Softmax) Learn() error {
	if s.trainingSet == nil || s.expectedResults == nil {
		err := fmt.Errorf("ERROR: Attempting to learn with no training examples!\n")
		fmt.Printf(err.Error())
		return err
	}

	examples := len(s.trainingSet)
	if examples == 0 || len(s.trainingSet[0]) == 0 {
		err := fmt.Errorf("ERROR: Attempting to learn with no training examples!\n")
		fmt.Printf(err.Error())
		return err
	}
	if len(s.expectedResults) == 0 {
		err := fmt.Errorf("ERROR: Attempting to learn with no expected results! This isn't an unsupervised model!! You'll need to include data before you learn :)\n")
		fmt.Printf(err.Error())
		return err
	}

	fmt.Printf("Training:\n\tModel: Softmax Classification\n\tOptimization Method: %v\n\tTraining Examples: %v\n\t Classification Dimensions: %v\n\tFeatures: %v\n\tLearning Rate α: %v\n\tRegularization Parameter λ: %v\n...\n\n", s.method, examples, s.k, len(s.trainingSet[0]), s.alpha, s.regularization)

	var err error
	if s.method == base.BatchGA {
		err = func() error {
			// if the iterations given is 0, set it to be
			// 5000 (seems reasonable base value)
			if s.maxIterations == 0 {
				s.maxIterations = 5000
			}

			iter := 0

			// Stop iterating if the number of iterations exceeds
			// the limit
			for ; iter < s.maxIterations; iter++ {

				// go over each parameter vector for each
				// classification value
				newTheta := make([][]float64, len(s.Parameters))
				for k, theta := range s.Parameters {
					newTheta[k] = make([]float64, len(theta))

					dj, err := s.Dj(k)
					if err != nil {
						return err
					}

					for j := range theta {
						newTheta[k][j] = theta[j] + s.alpha*dj[j]
						if math.IsInf(newTheta[k][j], 0) || math.IsNaN(newTheta[k][j]) {
							return fmt.Errorf("Sorry dude! Learning diverged. Some value of the parameter vector theta is ±Inf or NaN")
						}
					}
				}

				s.Parameters = newTheta
			}

			fmt.Printf("Went through %v iterations.\n", iter)

			return nil
		}()
	} else if s.method == base.StochasticGA {
		err = func() error {
			// if the iterations given is 0, set it to be
			// 5000 (seems reasonable base value)
			if s.maxIterations == 0 {
				s.maxIterations = 5000
			}

			iter := 0

			// Stop iterating if the number of iterations exceeds
			// the limit
			for ; iter < s.maxIterations; iter++ {
				for j := range s.trainingSet {
					newTheta := make([][]float64, len(s.Parameters))
					// go over each parameter vector for each
					// classification value
					for k, theta := range s.Parameters {
						newTheta[k] = make([]float64, len(theta))
						dj, err := s.Dij(j, k)
						if err != nil {
							return err
						}

						// now simultaneously update theta
						for j := range theta {
							newTheta[k][j] = theta[j] + s.alpha*dj[j]
							if math.IsInf(newTheta[k][j], 0) || math.IsNaN(newTheta[k][j]) {
								return fmt.Errorf("Sorry dude! Learning diverged. Some value of the parameter vector theta is ±Inf or NaN")
							}
						}
					}

					s.Parameters = newTheta
				}
			}

			fmt.Printf("Went through %v iterations.\n", iter)

			return nil
		}()
	} else {
		err = fmt.Errorf("Chose a training method not implemented for Softmax regression")
	}

	if err != nil {
		fmt.Printf("\nERROR: Error while learning –\n\t%v\n\n", err)
		return err
	}

	fmt.Printf("Training Completed.\n%v\n\n", s)
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
// NOTE part 4: the optional parameter 'normalize' will
// , if true, normalize all data streamed through the
// channel to unit length. This will affect the outcome
// of the hypothesis, though it could be favorable if
// your data comes in drastically different scales.
//
// Example Online Logistic Regression:
//
//     // create the channel of data and errors
//     stream := make(chan base.Datapoint, 100)
//     errors := make(chan error)
//
//     // notice how we are adding another integer
//     // to the end of the NewSoftmax call. This
//     // tells the model to use that number of features
//     // (2) in leu of finding that from the dataset
//     // like you would with batch/stochastic GD
//     //
//     // Also – the 'base.StochasticGA' doesn't affect
//     // anything. You could put batch or any other model.
//     model := NewSoftmax(base.StochasticGA, 5e-5, 0, 3, 0, nil, nil, 2)
//
//     go model.OnlineLearn(errors, stream, func(theta [][]float64) {
//         // do something with the new theta (persist
//         // to database?) in here.
//     })
//
//     go model.OnlineLearn(errors, stream, func(theta [][]float64) {})
//
//     // start passing data to our datastream
//     //
//     // we could have data already in our channel
//     // when we instantiated the Perceptron, though
//	   go func() {
//         for iter := 0; iter < 3; iter++ {
//             for i := -2.0; i < 2.0; i += 0.15 {
//                 for j := -2.0; j < 2.0; j += 0.15 {
//
//                     if -2*i+j/2-0.5 > 0 && -1*i-j < 0 {
//                              stream <- base.Datapoint{
//                                 X: []float64{float64(i), float64(j)},
//                                 Y: []float64{2.0},
//                             }
//                     } else if -2*i+j/2-0.5 > 0 && -1*i-j > 0 {
//                             stream <- base.Datapoint{
//                                 X: []float64{float64(i), float64(j)},
//                                 Y: []float64{1.0},
//                             }
//                     } else {
//                         stream <- base.Datapoint{
//                                 X: []float64{float64(i), float64(j)},
//                                 Y: []float64{0.0},
//                             }
//                     }
//                 }
//             }
//         }
//
//         // close the dataset
//         close(stream)
//     }()
//
//     // this will block until the error
//     // channel is closed in the learning
//     // function (it will, don't worry!)
//     for {
//         err, more := <-errors
//         if err != nil {
//             panic("THERE WAS AN ERROR!!! RUN!!!!")
//         }
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
func (s *Softmax) OnlineLearn(errors chan error, dataset chan base.Datapoint, onUpdate func([][]float64), normalize ...bool) {
	if errors == nil {
		errors = make(chan error)
	}
	if dataset == nil {
		errors <- fmt.Errorf("ERROR: Attempting to learn with a nil data stream!\n")
		close(errors)
		return
	}

	fmt.Printf("Training:\n\tModel: Softmax Classifier (%v classes)\n\tOptimization Method: Online Stochastic Gradient Descent\n\tFeatures: %v\n\tLearning Rate α: %v\n...\n\n", s.k, len(s.Parameters), s.alpha)

	norm := len(normalize) != 0 && normalize[0]
	var point base.Datapoint
	var more bool

	for {
		point, more = <-dataset

		if more {
			if len(point.Y) != 1 {
				errors <- fmt.Errorf("ERROR: point.Y must have a length of 1. Point: %v", point)
				continue
			}

			if norm {
				base.NormalizePoint(point.X)
			}

			// go over each parameter vector for each
			// classification value
			for k, theta := range s.Parameters {
				dj, err := func(point base.Datapoint, j int) ([]float64, error) {
					grad := make([]float64, len(s.Parameters[0]))

					// account for constant term
					x := append([]float64{1}, point.X...)

					var ident float64
					if abs(point.Y[0]-float64(k)) < 1e-3 {
						ident = 1
					}

					var numerator float64
					var denom float64
					for a := 0; a < s.k; a++ {
						var inside float64

						// calculate theta * x
						for l, val := range s.Parameters[int(k)] {
							inside += val * x[l]
						}

						if a == k {
							numerator = math.Exp(inside)
						}

						denom += math.Exp(inside)
					}

					for a := range grad {
						grad[a] += x[a] * (ident - numerator/denom)
					}

					// add in the regularization term
					// λ*θ[j]
					//
					// notice that we don't count the
					// constant term
					for j := range grad {
						grad[j] += s.regularization * s.Parameters[k][j]
					}

					return grad, nil
				}(point, k)
				if err != nil {
					errors <- err
					return
				}

				// now simultaneously update theta
				for j := range theta {
					newθ := theta[j] + s.alpha*dj[j]
					if math.IsInf(newθ, 0) || math.IsNaN(newθ) {
						errors <- fmt.Errorf("Sorry dude! Learning diverged. Some value of the parameter vector theta is ±Inf or NaN")
						close(errors)
						return
					}
					s.Parameters[k][j] = newθ
				}
			}

			go onUpdate(s.Parameters)

		} else {
			fmt.Printf("Training Completed.\n%v\n\n", s)
			close(errors)
			return
		}
	}
}

// String implements the fmt interface for clean printing. Here
// we're using it to print the model as the equation h(θ)=...
// where h is the softmax hypothesis model
func (s *Softmax) String() string {
	if len(s.Parameters) == 0 {
		fmt.Printf("ERROR: Attempting to print model with the 0 vector as it's parameter vector! Train first!\n")
	}
	var buffer bytes.Buffer

	buffer.WriteString(fmt.Sprintf("h(θ,x)[i] = exp(θ[i]x) / Σ exp(θ[j]x)\n\tθ ∊ ℝ^(%v x %v)\n", len(s.Parameters), len(s.Parameters[0])))

	return buffer.String()
}

// Dj returns the partial derivative of the cost function J(θ)
// with respect to theta[k] where theta is the parameter vector
// associated with our hypothesis function Predict (upon which
// we are optimizing.
//
// k is the classification value you are finding the gradient
// for (because the parameter vactor is actually a vector _of_
// vectors!)
func (s *Softmax) Dj(k int) ([]float64, error) {
	if k > s.k || k < 0 {
		return nil, fmt.Errorf("Given k (%v) is not valid with respect to the model", k)
	}

	sum := make([]float64, len(s.Parameters[0]))

	for i := range s.trainingSet {
		// account for constant term
		x := append([]float64{1}, s.trainingSet[i]...)

		var ident float64
		// 1{y == k}
		if int(s.expectedResults[i]) == k {
			ident = 1
		}

		var numerator float64
		var denom float64
		for a := 0; a < s.k; a++ {
			var inside float64

			// calculate theta * x
			for l := range s.Parameters[k] {
				inside += s.Parameters[k][l] * x[l]
			}

			if a == k {
				numerator = math.Exp(inside)
			}

			denom += math.Exp(inside)

		}

		c := ident - numerator/denom
		for a := range sum {
			sum[a] += x[a] * c
		}
	}

	// add in the regularization term
	// λ*θ[j]
	//
	// notice that we don't count the
	// constant term
	for j := range sum {
		sum[j] += s.regularization * s.Parameters[k][j]
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
func (s *Softmax) Dij(i, k int) ([]float64, error) {
	if k > s.k || k < 0 {
		return nil, fmt.Errorf("Given k (%v) is not valid with respect to the model", k)
	}

	grad := make([]float64, len(s.Parameters[0]))

	// account for constant term
	x := append([]float64{1}, s.trainingSet[i]...)

	var ident float64
	if abs(s.expectedResults[i]-float64(k)) < 1e-3 {
		ident = 1
	}

	var numerator float64
	var denom float64
	for a := 0; a < s.k; a++ {
		var inside float64

		// calculate theta * x
		for l, val := range s.Parameters[int(k)] {
			inside += val * x[l]
		}

		if a == k {
			numerator = math.Exp(inside)
		}

		denom += math.Exp(inside)
	}

	for a := range grad {
		grad[a] += x[a] * (ident - numerator/denom)
	}

	// add in the regularization term
	// λ*θ[j]
	//
	// notice that we don't count the
	// constant term
	for j := range grad {
		grad[j] += s.regularization * s.Parameters[k][j]
	}

	return grad, nil
}

// Theta returns the parameter vector θ for use in persisting
// the model, and optimizing the model through gradient descent
// ( or other methods like Newton's Method)
func (s *Softmax) Theta() [][]float64 {
	return s.Parameters
}

// PersistToFile takes in an absolute filepath and saves the
// parameter vector θ to the file, which can be restored later.
// The function will take paths from the current directory, but
// functions
//
// The data is stored as JSON because it's one of the most
// efficient storage method (you only need one comma extra
// per feature + two brackets, total!) And it's extendable.
func (s *Softmax) PersistToFile(path string) error {
	if path == "" {
		return fmt.Errorf("ERROR: you just tried to persist your model to a file with no path!! That's a no-no. Try it with a valid filepath")
	}

	bytes, err := json.Marshal(s.Parameters)
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
func (s *Softmax) RestoreFromFile(path string) error {
	if path == "" {
		return fmt.Errorf("ERROR: you just tried to restore your model from a file with no path! That's a no-no. Try it with a valid filepath")
	}

	bytes, err := ioutil.ReadFile(path)
	if err != nil {
		return err
	}

	err = json.Unmarshal(bytes, &s.Parameters)
	if err != nil {
		return err
	}

	return nil
}
