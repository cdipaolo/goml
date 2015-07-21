package cluster

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"time"

	"github.com/cdipaolo/goml/base"
)

// diff returns the square magnitude of the
// vector subtrraction between u and v. This
// is also known as the Squared Euclidean
// Distance:
//
// diff(u, v) == |u - v|^2
//
// **NOTE** The function assumes that u and
// v are the same dimension to avoid constant
// checking from within algorithms.
func diff(u, v []float64) float64 {
	var sum float64
	for i := range u {
		sum += (u[i] - v[i]) * (u[i] - v[i])
	}

	return sum
}

/*
KMeans implements the k-means unsupervised
clustering algorithm.

https://en.wikipedia.org/wiki/K-means_clustering

Example KMeans Model Usage:

	// initialize data with 2 clusters
	double := [][]float64{}
	for i := -10.0; i < -3; i += 0.1 {
		for j := -10.0; j < 10; j += 0.1 {
			double = append(double, []float64{i, j})
		}
	}

	for i := 3.0; i < 10; i += 0.1 {
		for j := -10.0; j < 10; j += 0.1 {
			double = append(double, []float64{i, j})
		}
	}

	model := NewKMeans(2, 30, double)

	if model.Learn() != nil {
		panic("Oh NO!!! There was an error learning!!")
	}

	// now predict with the same training set and
	// make sure the classes are the same within
	// each block
	c1, err := model.Predict([]float64{-7.5, 0})
	if err != nil {
		panic("prediction error")
	}

	c2, err := model.Predict([]float64{7.5, 0})
	if err != nil {
		panic("prediction error")
	}

	// now you can predict like normal!
	guess, err := model.Predict([]float64{-3, 6})
	if err != nil {
		panic("prediction error")
	}

	// or if you want to get the clustering
	// results from the data
	results := model.Guesses()

	// you can also concat that with the
	// training set and save it to a file
	// (if you wanted to plot it or something)
	err = model.SaveClusteredData("/tmp/.goml/KMeansResults.csv")
	if err != nil {
		panic("file save error")
	}

	// you can also persist the model to a
	// file
	err = model.PersistToFile("/tmp/.goml/KMeans.csv")
	if err != nil {
		panic("file save error")
	}

	// and also restore from file (at a
	// later time if you want)
	err = model.RestoreFromFile("/tmp/.goml/KMeans.csv")
	if err != nil {
		panic("file save error")
	}
*/
type KMeans struct {
	// maxIterations is the number of iterations
	// the learning will be cut off at in a
	// non-online setting.
	maxIterations int

	// trainingSet and guesses are the
	// 'x', and 'y' of the data, expressed as
	// vectors, that the model can optimize from.
	//
	// Note that because K-Means is an
	// unsupervised algorithm, the 'guesses'
	// parameter is set while learning.
	// If you want to use the training
	// not only to predict but just cluster
	// an existing dataset, this storage
	// will let the user export the predictions
	//
	// [][]float64{guesses[i]} == Predict(trainingSet[i])
	trainingSet [][]float64
	guesses     []int

	Centroids [][]float64 `json:"centroids"`
}

// NewKMeans returns a pointer to the k-means
// model, which clusters given inputs in an
// unsupervised manner. The algorithm only has
// one optimization method (unless learning with
// the online variant which is more of a generalization
// than the same algorithm) so you aren't allowed
// to pass one in as an option.
func NewKMeans(k, maxIterations int, trainingSet [][]float64) *KMeans {
	var features int
	if len(trainingSet) != 0 {
		features = len(trainingSet[0])
	}

	// start all guesses with the zero vector.
	// they will be changed during learning
	var guesses []int
	guesses = make([]int, len(trainingSet))

	rand.Seed(time.Now().UTC().Unix())
	centroids := make([][]float64, k)
	for i := range centroids {
		centroids[i] = make([]float64, features)
		for j := range centroids[i] {
			centroids[i][j] = 10 * (rand.Float64() - 0.5)
		}
	}

	return &KMeans{
		maxIterations: maxIterations,

		trainingSet: trainingSet,
		guesses:     guesses,

		Centroids: centroids,
	}
}

// UpdateTrainingSet takes in a new training set (variable x.)
//
// Will reset the hidden 'guesses' param of the KMeans model.
func (k *KMeans) UpdateTrainingSet(trainingSet [][]float64) error {
	if len(trainingSet) == 0 {
		return fmt.Errorf("Error: length of given training set is 0! Need data!")
	}

	k.trainingSet = trainingSet
	k.guesses = make([]int, len(trainingSet))

	return nil
}

// Examples returns the number of training examples (m)
// that the model currently is training from.
func (k *KMeans) Examples() int {
	return len(k.trainingSet)
}

// MaxIterations returns the number of maximum iterations
// the model will go through in GradientAscent, in the
// worst case
func (k *KMeans) MaxIterations() int {
	return k.maxIterations
}

// Predict takes in a variable x (an array of floats,) and
// finds the value of the hypothesis function given the
// current parameter vector θ
//
// if normalize is given as true, then the input will
// first be normalized to unit length. Only use this if
// you trained off of normalized inputs and are feeding
// an un-normalized input
func (k *KMeans) Predict(x []float64, normalize ...bool) ([]float64, error) {
	if len(x) != len(k.Centroids[0]) {
		return nil, fmt.Errorf("Error: Centroid vector should be the same length as input vector!\n\tLength of x given: %v\n\tLength of centroid: %v\n", len(x), len(k.Centroids[0]))
	}

	if len(normalize) != 0 && normalize[0] {
		base.NormalizePoint(x)
	}

	var guess int
	minDiff := diff(x, k.Centroids[0])
	for j := 1; j < len(k.Centroids); j++ {
		difference := diff(x, k.Centroids[j])
		if difference < minDiff {
			minDiff = difference
			guess = j
		}
	}

	return []float64{float64(guess)}, nil
}

// Learn takes the struct's dataset and expected results and runs
// batch gradient descent on them, optimizing theta so you can
// predict based on those results
func (k *KMeans) Learn() error {
	if k.trainingSet == nil {
		err := fmt.Errorf("ERROR: Attempting to learn with no training examples!\n")
		fmt.Printf(err.Error())
		return err
	}

	examples := len(k.trainingSet)
	if examples == 0 || len(k.trainingSet[0]) == 0 {
		err := fmt.Errorf("ERROR: Attempting to learn with no training examples!\n")
		fmt.Printf(err.Error())
		return err
	}

	centroids := len(k.Centroids)
	features := len(k.trainingSet[0])

	fmt.Printf("Training:\n\tModel: K-Means Classification\n\tTraining Examples: %v\n\tFeatures: %v\n\tClasses: %v\n...\n\n", examples, features, centroids)

	iter := 0
	for ; iter < k.maxIterations; iter++ {

		// set new guesses
		//
		// store counts when assigning classes
		// so you won't have to sum them again later
		classTotal := make(map[int][]float64)
		classCount := make(map[int]int64)

		for j := range k.Centroids {
			classTotal[j] = make([]float64, features)
		}

		for i, x := range k.trainingSet {
			k.guesses[i] = 0
			minDiff := diff(x, k.Centroids[0])
			for j := 1; j < len(k.Centroids); j++ {
				difference := diff(x, k.Centroids[j])
				if difference < minDiff {
					minDiff = difference
					k.guesses[i] = j
				}
			}

			classCount[k.guesses[i]]++
			for j := range x {
				classTotal[k.guesses[i]][j] += x[j]
			}
		}

		newCentroids := append([][]float64{}, k.Centroids...)
		for j := range k.Centroids {
			// if no objects are in the same class,
			// reinitialize it to a random vector
			if classCount[j] == 0 {
				for l := range k.Centroids[j] {
					k.Centroids[j][l] = 10 * (rand.Float64() - 0.5)
				}
				continue
			}

			for l := range k.Centroids[j] {
				k.Centroids[j][l] = classTotal[j][l] / float64(classCount[j])
			}
		}

		// only update if something was deleted
		if len(newCentroids) != len(k.Centroids) {
			k.Centroids = newCentroids
		}
	}

	fmt.Printf("Training Completed in %v iterations.\n%v\n", iter, k)

	if len(k.Centroids) != centroids {
		fmt.Printf("Now have %v classes (had %v before) because one/some held no examples\n", len(k.Centroids), centroids)
	}

	return nil
}

// String implements the fmt interface for clean printing. Here
// we're using it to print the model as the equation h(θ)=...
// where h is the k-means hypothesis model
func (k *KMeans) String() string {
	return fmt.Sprintf("h(θ,x) = argmin_j | x[i] - μ[j] |^2\n\tμ = %v", k.Centroids)
}

// Guesses returns the hidden parameter for the
// unsupervised classification assigned during
// learning.
//
//    model.Guesses[i] = E[k.trainingSet[i]]
func (k *KMeans) Guesses() []int {
	return k.guesses
}

// Distortion returns the distortion of the clustering
// currently given by the k-means model. This is the
// function the learning algorithm tries to minimize.
//
// Distorition() = Σ |x[i] - μ[c[i]]|^2
// over all training examples
func (k *KMeans) Distortion() float64 {
	var sum float64
	for i := range k.trainingSet {
		sum += diff(k.trainingSet[i], k.Centroids[int(k.guesses[i])])
	}

	return sum
}

// SaveClusteredData takes operates on a k-means
// model, concatenating the given dataset with the
// assigned class from clustering and saving it to
// file.
//
// Basically just a wrapper for the base.SaveDataToCSV
// with the K-Means data.
func (k *KMeans) SaveClusteredData(filepath string) error {
	floatGuesses := []float64{}
	for _, val := range k.guesses {
		floatGuesses = append(floatGuesses, float64(val))
	}

	return base.SaveDataToCSV(filepath, k.trainingSet, floatGuesses, true)
}

// PersistToFile takes in an absolute filepath and saves the
// centroid vector to the file, which can be restored later.
// The function will take paths from the current directory, but
// functions
//
// The data is stored as JSON because it's one of the most
// efficient storage method (you only need one comma extra
// per feature + two brackets, total!) And it's extendable.
func (k *KMeans) PersistToFile(path string) error {
	if path == "" {
		return fmt.Errorf("ERROR: you just tried to persist your model to a file with no path!! That's a no-no. Try it with a valid filepath")
	}

	bytes, err := json.Marshal(k.Centroids)
	if err != nil {
		return err
	}

	err = ioutil.WriteFile(path, bytes, os.ModePerm)
	if err != nil {
		return err
	}

	return nil
}

// RestoreFromFile takes in a path to a centroid vector
// and assigns the model it's operating on's parameter vector
// to that.
//
// The path must ba an absolute path or a path from the current
// directory
//
// This would be useful in persisting data between running
// a model on data, or for graphing a dataset with a fit in
// another framework like Julia/Gadfly.
func (k *KMeans) RestoreFromFile(path string) error {
	if path == "" {
		return fmt.Errorf("ERROR: you just tried to restore your model from a file with no path! That's a no-no. Try it with a valid filepath")
	}

	bytes, err := ioutil.ReadFile(path)
	if err != nil {
		return err
	}

	err = json.Unmarshal(bytes, &k.Centroids)
	if err != nil {
		return err
	}

	return nil
}
