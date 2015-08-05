// Package base declares models, interfaces, and
// methods to be used when working with the rest
// of the goml library. It also includes common functions
// both used by the rest of the library and for
// the user's convenience for working with data,
// persisting it to files, and optimizing functions
package base

// OptimizationMethod defines a type enum which
// (using constants declared below) lets a user
// pass in a optimization method to use when
// creating a new model
type OptimizationMethod string

// Constants declare the types of optimization
// methods you can use.
const (
	BatchGA      OptimizationMethod = "Batch Gradient Ascent"
	StochasticGA                    = "Stochastic Gradient Descent"
)

// Model is an interface that can Train based on
// a 2D array of data (called x) and an array (y)
// of solution data. Model trains in a supervised
// manor. Predict takes in a vector of floats and
// returns a real number response (float, again)
// and an error if any
type Model interface {

	// The variadic argument in Predict is an
	// optional arg which (if true) tells the
	// function to first normalize the input to
	// vector unit length. Use (and only use) this
	// if you trained on normalized inputs.
	Predict([]float64, ...bool) ([]float64, error)

	// PersistToFile and RestoreFromFile both take
	// in paths (absolute paths!) to files and
	// persists the necessary data to the filepath
	// such that you can RestoreFromFile later and
	// have the same instance. Helpful when you want
	// to train a model, save it to a file, then
	// open it later for prediction
	PersistToFile(string) error
	RestoreFromFile(string) error
}

// OnlineModel differs from Model because the learning
// can take place in a goroutine because the data
// is passed through a channel, ending when the
// channel is closed.
type OnlineModel interface {
	Predict([]float64) ([]float64, error)

	// OnlineLearn has no outputs so you can run the data
	// within a separate goroutine! A channel of
	// errors is passed so you know when there's been
	// an error in learning, though learning will
	// just ignore the datapoint that caused the
	// error and continue on.
	//
	// Most times errors are caused when passed
	// datapoints are not of a consistent dimension.
	//
	// The function passed is a callback that is called
	// whenever the parameter vector theta is updated
	OnlineLearn(chan error, func([]float64))

	// UpdateStream updates the datastream channel
	// used in learning for the algorithm
	UpdateStream(chan Datapoint)

	// PersistToFile and RestoreFromFile both take
	// in paths (absolute paths!) to files and
	// persists the necessary data to the filepath
	// such that you can RestoreFromFile later and
	// have the same instance. Helpful when you want
	// to train a model, save it to a file, then
	// open it later for prediction
	PersistToFile(string) error
	RestoreFromFile(string) error
}

// OnlineTextModel holds the interface for text
// classifiers. They have the refular learn
// & predict functions, but don't include
// an updating callback func in OnlineLearn
// because the parameter vector passed would
// very often be _huge_, and therefore would
// be a detriment to performance.
type OnlineTextModel interface {
	// Predict takes a document and returns the
	// expected class found by the model
	Predict(string) uint8

	// OnlineLearn has no outputs so you can run the data
	// within a separate goroutine! A channel of
	// errors is passed so you know when there's been
	// an error in learning, though learning will
	// just ignore the datapoint that caused the
	// error and continue on.
	OnlineLearn(chan<- error)

	// UpdateStream updates the datastream channel
	// used in learning for the algorithm
	UpdateStream(chan TextDatapoint)

	// PersistToFile and RestoreFromFile both take
	// in paths (absolute paths!) to files and
	// persists the necessary data to the filepath
	// such that you can RestoreFromFile later and
	// have the same instance. Helpful when you want
	// to train a model, save it to a file, then
	// open it later for prediction
	PersistToFile(string) error
	RestoreFromFile(string) error
}

// Ascendable is an interface that can be used
// with batch gradient descent where the parameter
// vector theta is in one dimension only (so
// softmax regression would need it's own model,
// for example)
type Ascendable interface {
	// LearningRate returns the learning rate α
	// to be used in Gradient Descent as the
	// modifier term
	LearningRate() float64

	// Dj returns the derivative of the cost function
	// J(θ) with respect to the j-th parameter of
	// the hypothesis, θ[j]. Called as Dj(j)
	Dj(int) (float64, error)

	// Theta returns a pointer to the parameter vector
	// theta, which is 1D vector of floats
	Theta() []float64

	// MaxIterations returns the maximum number of
	// iterations to try using gradient ascent. Might
	// return after less if strong convergance is
	// detected, but it'll let the user set a cap.
	MaxIterations() int
}

// StochasticAscendable is an interface that can be used
// with stochastic gradient descent where the parameter
// vector theta is in one dimension only (so
// softmax regression would need it's own model,
// for example)
type StochasticAscendable interface {
	// LearningRate returns the learning rate α
	// to be used in Gradient Descent as the
	// modifier term
	LearningRate() float64

	// Examples returns the number of examples in the
	// training set the model is using
	Examples() int

	// Dj returns the derivative of the cost function
	// J(θ) with respect to the j-th parameter of
	// the hypothesis, θ[j], for the training example
	// x[i]. Called as Dij(i,j)
	Dij(int, int) (float64, error)

	// Theta returns a pointer to the parameter vector
	// theta, which is 1D vector of floats
	Theta() []float64

	// MaxIterations returns the maximum number of
	// iterations to try using gradient ascent. Might
	// return after less if strong convergance is
	// detected, but it'll let the user set a cap.
	MaxIterations() int
}

// Datapoint is used in some models where it is cleaner
// to pass data as a struct rather than just as 1D and
// 2D arrays like Generalized Linear Models are doing,
// for example. X corresponds to the inputs and Y
// corresponds to the result of the hypothesis.
//
// This is used with the Perceptron, for example, so
// data can be easily passed in channels while staying
// encapsulated well.
type Datapoint struct {
	X []float64 `json:"x"`
	Y []float64 `json:"y"`
}

// TextDatapoint is the data structure expected
// for text classification models. The passed
// types, therefore, are inherently different
// from the other structures. X is now a string
// (or, document. Usually this would be a
// sentence or multiple sentences.) Y is now
// a uint8 denoting the class, because you can't
// regress on text classification (at least not
// well/effectively)
type TextDatapoint struct {
	X string `json:"x"`
	Y uint8  `json:"y"`
}
