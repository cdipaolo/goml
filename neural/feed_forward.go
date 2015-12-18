/*
Package neural implements neural network models
with online (passing data to stochastic GD
through channels) and batch methods.
*/
package neural

import "math/rand"

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
	Features   uint8          `json:"features,omitempty"`
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
func NewFeedForwardNet(features uint8, alpha float64, iterations int, layers []uint64, transforms []NonLinearity, trainingSet [][]float64, expectedResults [][]float64) *FeedForwardNet {
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
				weights[l][j][i] = (rand.Float64()*2 - 1.0)
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
		Layers:     numLayers,
		Dimensions: layers,
		Transforms: transforms,

		Weights: weights,

		outputs: outputs,
		sums:    sums,
		deltas:  deltas,

		trainingSet:     trainingSet,
		expectedResults: expectedResults,
	}
}

// UpdateTrainingSet takes in a new training
// set and expected results and updates the
// model's training set, returning any errors.
func (n *FeedForwardNet) UpdateTrainingSet(trainingSet [][]float64, expectedResults []float64) error {
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

func (n *FeedForwardNet) forward(x []float64) error {
	if len(x) != n.Features {
		return fmt.Errorf("Error: x input dimension (%v) does not equal the expcted network feature dimensions (%v)", len(x), n.Features)
	}

	// bottom layer takes from input x
	for j := range n.Weights[0] {
		n.sums[0][j] = n.Weights[0][j][0]
		for i := 1; i < len(n.Weights[0][j]); i++ {
			n.sums[0][j] += x[i-1] * n.Weights[0][j][i]
		}
	}

	// other layers take from lower layers
	for l := 1; l < n.Layers; l++ {

	}
}

/* Cost Functions */

// J returns the Least Squares cost function of the given
// model. Could be usefull in testing convergance
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
	return fmt.Sprintf("Layers: %v\nTransforms: %v\nLearning Rate: %v\nMaxIterations: %v", n.dimension, EncodeNonLinearitySlice(n.transforms), n.alpha, n.maxIterations)
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
	n.Features = restores.Features

	return nil
}
