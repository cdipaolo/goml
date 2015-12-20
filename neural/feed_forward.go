/*
Package neural implements neural network models
with online (passing data to stochastic GD
through channels) and batch methods.
*/
package neural

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"

	"github.com/cdipaolo/goml/base"
)

type FeedForwardNet struct {
	// learning rate and number of
	// iterations to train on
	alpha         float64
	maxIterations int

	// layers is the nuber of layers
	// in the network. dimension is
	// the number of neurons in each
	// layer (where dimension[l] is
	// the number of neurons in the l-th
	// layer)
	//
	// transforms is the defined nonlinearity
	// for each layer such that transforms[l]
	// is the transform (Identity, ReLu, Sigmoid, Tanh, etc.)
	// for that layer.
	Features   uint64         `json:"features,omitempty"`
	Layers     uint8          `json:"layers,omitempty"`
	Dimension  []uint64       `json:"dimension,omitempty"`
	Transforms []NonLinearity `json:"transforms,omitempty"`

	// weight vector (technically a
	// tensor here, but the dimensions
	// per layer may/will differ)
	//
	//.Weights[l][j][i] is the i-th weight
	// of the j-th neuron in the l-th
	// layer of the network. Indexing
	// is 'backwards' because it makes sense
	// to instantiate the.Weights by layer
	// and neuron first.
	Weights [][][]float64 `json:.Weights,omitempty"`

	// outputs is the output from
	// each neuron (saved to help train
	// faster). sums is the w*x that
	// is unputted into the neuron
	// to transform.
	//
	// outputs[j][l] is the output
	// of the j-th neuron in the l-th layer
	//
	// sums[j][l] is the input of the j-th
	// neuron in the l-th layer
	outputs [][]float64
	sums    [][]float64

	// delta is the error backpropogated
	// during training. delta[j][l] is
	// the error for the j-th output of
	// the l-th layer
	delta [][]float64

	/* Training Data */
	trainingSet     [][]float64
	expectedResults [][]float64
}

// NewFeedForwardNet takes in the number of features of
// input vetors, the learning rate alpha, a slice of
// uint8's where each value is the number of neurons
// in that layer, and the training set.
//
// Giving a nil training set is fine because you
// can train in online SGD mode.
func NewFeedForwardNet(features uint64, alpha float64, iterations int, layers []uint64, transforms []NonLinearity, trainingSet [][]float64, expectedResults [][]float64) *FeedForwardNet {
	if len(transforms) != len(layers) || len(trainingSet) != len(expectedResults) {
		return nil
	}
	numLayers := len(layers)
	weights := make([][][]float64, numLayers)
	for l := range weights {
		weights[l] = make([][]float64, layers[l])
		for j := range weights[l] {
			if l == 0 {
				weights[l][j] = make([]float64, features+1)
			} else {
				weights[l][j] = make([]float64, layers[l-1]+1)
			}

			// instantiate.Weights randomly
			// weights in [-1.0, 1.0]
			for i := range weights[l][j] {
				weights[l][j][i] = 2 * (rand.Float64()*2 - 1.0)
			}
		}
	}

	outputs := make([][]float64, numLayers)
	sums := make([][]float64, numLayers)
	deltas := make([][]float64, numLayers)
	for i := range outputs {
		layer := layers[i]
		outputs[i] = make([]float64, layer)
		sums[i] = make([]float64, layer)
		deltas[i] = make([]float64, layer)
	}

	return &FeedForwardNet{
		alpha:         alpha,
		maxIterations: iterations,

		Features:   features,
		Layers:     uint8(numLayers),
		Dimension:  layers,
		Transforms: transforms,

		Weights: weights,

		outputs: outputs,
		sums:    sums,
		delta:   deltas,

		trainingSet:     trainingSet,
		expectedResults: expectedResults,
	}
}

// UpdateTrainingSet takes in a new training
// set and expected results and updates the
// model's training set, returning any errors.
func (n *FeedForwardNet) UpdateTrainingSet(trainingSet [][]float64, expectedResults [][]float64) error {
	if len(trainingSet) == 0 {
		return fmt.Errorf("Error: length of given training set is 0! Need data!")
	}
	if len(expectedResults) == 0 {
		return fmt.Errorf("Error: length of given result data set is 0! Need expected results!")
	}
	if len(trainingSet) != len(expectedResults) {
		return fmt.Errorf("Error: length of training set should equal length of results")
	}

	n.trainingSet = trainingSet
	n.expectedResults = expectedResults

	return nil
}

// UpdateLearningRate takes in a learning rate
// alpha and updates the networks learning rate.
func (n *FeedForwardNet) UpdateLearningRate(a float64) {
	n.alpha = a
}

// Examples returns the number of training examples
// the model is currently training from.
func (n *FeedForwardNet) Examples() int {
	return len(n.trainingSet)
}

// MaxIterations returns the maximum number of
// iterations the neural network will go through
// in stochastic gradient descent.
func (n *FeedForwardNet) MaxIterations() int {
	return n.maxIterations
}

/* Learning and Predicting */

// Predict performs a forward pass of the
// input through the network and returns the
// result to the user.
func (n *FeedForwardNet) Predict(x []float64) ([]float64, error) {
	if len(x) != int(n.Features) {
		return nil, fmt.Errorf("Error: x input dimension (%v) does not equal the expcted network feature dimensions (%v)", len(x), n.Features)
	}

	n.forward(x)

	return n.outputs[n.Layers-1], nil
}

// forward performs the forward pass of the network,
// storing intermediate results to perform backpropogation
// later if there is a label
func (n *FeedForwardNet) forward(x []float64) {

	// bottom layer takes from input x
	var sum float64
	for j := 0; j < int(n.Dimension[0]); j++ {
		sum = n.Weights[0][j][0]
		for i := 1; i < len(n.Weights[0][j]); i++ {
			sum += x[i-1] * n.Weights[0][j][i]
		}

		n.outputs[0][j] = n.Transforms[0].F(sum)
	}

	// other layers take from lower layers
	for l := 1; l < int(n.Layers); l++ {
		for j := 0; j < int(n.Dimension[l]); j++ {
			sum = n.Weights[l][j][0]
			for i := 1; i < len(n.Weights[l][j]); i++ {
				sum += n.outputs[l-1][i-1] * n.Weights[l][j][i]
			}

			n.outputs[l][j] = n.Transforms[l].F(sum)
		}
	}
}

// backwards performs a backwards pass through the
// network, updating weights and recording deltas.
//
// This is an internal step, so inputs are assumed
// to have the correct dimensions.
func (n *FeedForwardNet) backwards(x, y []float64) {
	// first go through last layer
	for j := 0; j < int(n.Dimension[n.Layers-1]); j++ {
		n.delta[n.Layers-1][j] = (n.outputs[n.Layers-1][j] - y[j]) * n.Transforms[n.Layers-1].DF(n.outputs[n.Layers-1][j])
	}

	// then go through the rest of the layers
	if n.Layers > 1 {
		for l := int(n.Layers - 2); l >= 0; l-- {
			for i := 0; i < int(n.Dimension[l]); i++ {
				var sum float64
				dl := n.Transforms[l].DF(n.outputs[l][i])
				for j := 0; j < int(n.Dimension[l+1]); j++ {
					sum += n.delta[l][j] * n.Weights[l+1][j][i]
				}
				n.delta[l][i] = sum * dl
			}
		}
	}

	// now perform gradient descent on the example
	for j := 0; j < int(n.Dimension[0]); j++ {
		n.Weights[0][j][0] -= n.alpha * n.delta[0][j]
		for i := 0; i < len(n.Weights[0][j]); i++ {
			n.Weights[0][j][0] -= n.alpha * n.delta[0][j] * x[i]
		}
	}
	for l := 1; l < int(n.Layers); l++ {
		for j := 0; j < int(n.Dimension[l]); j++ {
			n.Weights[l][j][0] -= n.alpha * n.delta[l][j]
			for i := 1; i < len(n.Weights[l][j]); i++ {
				n.Weights[l][j][i] -= n.alpha * n.delta[l][j] * n.outputs[l-1][i-1]
			}
		}
	}
}

// computeDerivative computes the actual derivative
// of the cost function with respect to the i-th
// weight in the j-th neuron in the l-th layer
func (n *FeedForwardNet) computeDerivative(i, j, l int, x, y []float64) float64 {
	n.forward(x)

	// first go through last layer
	for j := 0; j < int(n.Dimension[n.Layers-1]); j++ {
		n.delta[n.Layers-1][j] = (n.outputs[n.Layers-1][j] - y[j]) * n.Transforms[n.Layers-1].DF(n.outputs[n.Layers-1][j])
	}

	// then go through the rest of the layers
	if n.Layers > 1 {
		for l := int(n.Layers - 2); l >= 0; l-- {
			for i := 0; i < int(n.Dimension[l]); i++ {
				var sum float64
				dl := n.Transforms[l].DF(n.outputs[l][i])
				for j := 0; j < int(n.Dimension[l+1]); j++ {
					sum += n.delta[l+1][j] * n.Weights[l+1][j][i+1]
				}
				n.delta[l][i] = sum * dl
			}
		}
	}

	if i == 0 {
		return n.delta[l][j]
	}
	if l == 0 {
		return n.delta[0][j] * x[i-1]
	}
	return n.delta[l][j] * n.outputs[l-1][i-1]
}

// computeNumericalDerivative computes the derivative
// of the cost function with respect to the i-th
// weight in the j-th neuron in the l-th layer
func (n *FeedForwardNet) computeNumericalDerivative(i, j, l int, x, y []float64, epsilon float64) float64 {
	n.Weights[l][j][i] += epsilon
	right, err := n.Cost(x, y)
	if err != nil {
		return 0.0
	}

	n.Weights[l][j][i] -= 2 * epsilon
	left, err := n.Cost(x, y)
	if err != nil {
		return 0.0
	}

	n.Weights[l][j][i] += epsilon
	return (right - left) / (2 * epsilon)
}

// Learn takes in a FeedForwardNet and trains it
// with the pre-assigned dataset and parameters,
// giving information about the training
func (n *FeedForwardNet) Learn() error {
	if n.trainingSet == nil || n.expectedResults == nil {
		err := fmt.Errorf("ERROR: Attempting to learn with no training examples!\n")
		print(err.Error())
		return err
	}

	examples := len(n.trainingSet)
	if examples == 0 || len(n.trainingSet[0]) == 0 {
		err := fmt.Errorf("ERROR: Attempting to learn with no training examples!\n")
		print(err.Error())
		return err
	}
	if len(n.expectedResults) == 0 {
		err := fmt.Errorf("ERROR: Attempting to learn with no expected results!\n")
		print(err.Error())
		return err
	}

	fmt.Printf("Training:\n\tModel: Feedforward Neural Network\n\tOptimization Method: Stochastic Gradient Descent\n\tFeatures: %v\n\tLearning Rate α: %v\n\tTransforms: %v\n\tDimensions: %v\n...\n\n", n.Features, n.alpha, EncodeNonLinearitySlice(n.Transforms), n.Dimension)

	for j := 0; j < n.maxIterations; j++ {
		for i := range n.trainingSet {
			n.forward(n.trainingSet[i])
			n.backwards(n.trainingSet[i], n.expectedResults[i])
		}
		if j%15 == 0 {
			print(".")
		}
	}

	fmt.Printf("\n\nTraining Completed in %v iterations! Model:\n%v\n", n.maxIterations, n)
	return nil
}

func (n *FeedForwardNet) OnlineLearn(errors chan error, dataset chan base.Datapoint, onUpdate func([][]float64), normalize ...bool) {
	if errors == nil {
		errors = make(chan error)
	}
	if dataset == nil {
		errors <- fmt.Errorf("ERROR: Attempting to learn with a nil data stream!\n")
		close(errors)
		return
	}

	fmt.Printf("Training:\n\tModel: Feedforward Neural Network\n\tOptimization Method: Online Stochastic Gradient Descent\n\tFeatures: %v\n\tLearning Rate α: %v\n\tTransforms: %v\n\tDimensions: %v\n...\n\n", n.Features, n.alpha, EncodeNonLinearitySlice(n.Transforms), n.Dimension)

	var point base.Datapoint
	var more bool

	for {
		point, more = <-dataset

		if more {
			if len(point.Y) != int(n.Dimension[n.Layers-1]) {
				errors <- fmt.Errorf("ERROR: point.Y must have a length of 1. Point: %v", point)
			}

			n.forward(point.X)
			n.backwards(point.X, point.Y)

		} else {
			fmt.Printf("Training Completed.\n%v\n\n", n)
			close(errors)
			return
		}
	}
}

/* Cost Functions */

// J returns the Least Squares cost function of the given
// model. Could be useful in testing convergence
//
// J = 1/(2n) ∑  ⃦y - h()  ⃦^2
func (n *FeedForwardNet) J() (float64, error) {
	var sum float64

	for i := range n.trainingSet {
		prediction, err := n.Predict(n.trainingSet[i])
		if err != nil {
			return 0, err
		}

		er := 0.0
		for j := range n.expectedResults[i] {
			miss := n.expectedResults[i][j] - prediction[j]
			er += miss * miss
		}
		sum += er
	}
	return sum / float64(2*len(n.trainingSet)), nil
}

// Cost computes the cost function J() on
// one point (used in numerical gradient computation)
func (n *FeedForwardNet) Cost(x, y []float64) (float64, error) {
	prediction, err := n.Predict(x)
	if err != nil {
		return 0, err
	}

	er := 0.0
	for j := range y {
		miss := y[j] - prediction[j]
		er += miss * miss
	}
	return er / 2, nil
}

/* Encoding */

/*
String returns the network in the following
format:

	Layers: [256, 115, 213, 2]
	Transforms: [ReLu, ReLu, ReLu, Sigmoid]
	Learning Rate: 0.1
	MaxIterations: 106
*/
func (n *FeedForwardNet) String() string {
	return fmt.Sprintf("Layers: %v\nTransforms: %v\nLearning Rate: %v\nMaxIterations: %v", n.Dimension, EncodeNonLinearitySlice(n.Transforms), n.alpha, n.maxIterations)
}

// PersistToFile takes in a filepath and stores
// the model there in JSON format, returning any
// errors.
func (n *FeedForwardNet) PersistToFile(path string) error {
	if path == "" {
		return fmt.Errorf("ERROR: you just tried to persist your model to a file with no path!! That's a no-no. Try it with a valid filepath")
	}

	bytes, err := json.Marshal(n)
	if err != nil {
		return err
	}

	err = ioutil.WriteFile(path, bytes, os.ModePerm)
	if err != nil {
		return err
	}

	return nil
}

// RestoreFromFile takes in a filepath to a
// (presumably) json formatted FeedForwardNet
// model and restores that model into the
// called struct.
func (n *FeedForwardNet) RestoreFromFile(path string) error {
	if path == "" {
		return fmt.Errorf("ERROR: you just tried to restore your model from a file with no path! That's a no-no. Try it with a valid filepath")
	}

	bytes, err := ioutil.ReadFile(path)
	if err != nil {
		return err
	}

	restored := FeedForwardNet{}
	err = json.Unmarshal(bytes, &restored)
	if err != nil {
		return err
	}

	n.Weights = restored.Weights
	n.Layers = restored.Layers
	n.Dimension = restored.Dimension
	n.Transforms = restored.Transforms
	n.Features = restored.Features

	return nil
}
