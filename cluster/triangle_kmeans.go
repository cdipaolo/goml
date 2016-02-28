package cluster

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"math/rand"
	"os"
	"time"

	"github.com/cdipaolo/goml/base"
)

/*
TriangleKMeans implements the k-means unsupervised
clustering algorithm sped up to use the Triangle
Inequality to reduce the number of reduntant
distance calculations between datapoints and
clusters. The revised algorithm does use O(n)
auxillary space (more like 3n + k^2), and computes
a lot on these extra data structures, but

	"Our algorithm reduces the number of distance
	 calculations so dramatically that its overhead
	 time is often greater than the time spent on
	 distance calculations. However, the total
	 execution time is always much less than the
	 time required by standard k-means"
    (Eklan 2003, University of California, San Diego)

Note that this algorithm also uses k-means++
instantiation for more reliable clustering. The
Triangle Inequality optimizations operate on a
different sector of the algorithm.

http://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf
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

	// note that this is the only
	// different line from regular k-means
	model := NewTriangleKMeans(2, 30, double)

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

	// or if you just want to get the clustering
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
	err = model.PersistToFile("/tmp/.goml/KMeans.json")
	if err != nil {
		panic("file save error")
	}

	// and also restore from file (at a
	// later time if you want)
	err = model.RestoreFromFile("/tmp/.goml/KMeans.json")
	if err != nil {
		panic("file save error")
	}
*/
type TriangleKMeans struct {
	// maxIterations is the number of iterations
	// the learning will be cut off at in a
	// non-online setting.
	maxIterations int

	// alpha is only used in the
	// online setting of the algorithm
	alpha float64

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
	info        []pointInfo

	Centroids [][]float64 `json:"centroids"`

	// centroidDist is a K x K matrix of
	// the distances such that centroidDist[i][j]
	// is the distance from centroid i to centroid
	// j. minCentroidDist is the minimum distance
	// from the i-th centroid to any other
	// centroid. Note that the minCentroidDist
	// distances are actually half of the actual
	// min. This is because, as seen in the paper,
	// that's all we need to bound our distance
	// calculations
	centroidDist    [][]float64
	minCentroidDist []float64

	// Output is the io.Writer to write logs
	// and output from training to
	Output io.Writer
}

// pointInfo stores information needed to use
// the Triangle Inequality to reduce the number
// of distance calculations.
type pointInfo struct {
	// lower holds lower bounds for the distance
	// from a point to each cluster (so length
	// of lower is k). Upper holds the upper bound
	// on the distance to _any_ cluster, and
	// hence has only one value. The upper bound
	// filters most distance calculations later
	// in the training.
	lower []float64
	upper float64

	// recompute tells the algorithm to recompute
	// the distance distance to it's center. If false
	// just use the upper bound.
	recompute bool
}

// NewTriangleKMeans returns a pointer to the k-means
// model, which clusters given inputs in an
// unsupervised manner. The differences between
// this algorithm and the standard k-means++
// algorithm implemented in NewKMeans are discribed
// in the struct comments and in the paper URL
// found within those.
func NewTriangleKMeans(k, maxIterations int, trainingSet [][]float64) *TriangleKMeans {
	var features int
	if len(trainingSet) != 0 {
		features = len(trainingSet[0])
	}

	// start all guesses with the zero vector.
	// they will be changed during learning
	var guesses []int
	var info []pointInfo
	guesses = make([]int, len(trainingSet))
	info = make([]pointInfo, len(trainingSet))
	for i := range info {
		info[i] = pointInfo{
			lower:     make([]float64, k),
			upper:     0,
			recompute: true,
		}
	}

	rand.Seed(time.Now().UTC().Unix())
	centroids := make([][]float64, k)
	centroidDist := make([][]float64, k)
	minCentroidDist := make([]float64, k)
	for i := range centroids {
		centroids[i] = make([]float64, features)
		centroidDist[i] = make([]float64, k)
	}

	return &TriangleKMeans{
		maxIterations: maxIterations,

		trainingSet: trainingSet,
		guesses:     guesses,
		info:        info,

		Centroids:       centroids,
		centroidDist:    centroidDist,
		minCentroidDist: minCentroidDist,

		Output: os.Stdout,
	}
}

// UpdateTrainingSet takes in a new training set (variable x.)
//
// Will reset the hidden 'guesses' param of the KMeans model.
func (k *TriangleKMeans) UpdateTrainingSet(trainingSet [][]float64) error {
	if len(trainingSet) == 0 {
		return fmt.Errorf("Error: length of given training set is 0! Need data!")
	}

	k.trainingSet = trainingSet
	k.guesses = make([]int, len(trainingSet))

	return nil
}

// Examples returns the number of training examples (m)
// that the model currently is training from.
func (k *TriangleKMeans) Examples() int {
	return len(k.trainingSet)
}

// MaxIterations returns the number of maximum iterations
// the model will go through
func (k *TriangleKMeans) MaxIterations() int {
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
func (k *TriangleKMeans) Predict(x []float64, normalize ...bool) ([]float64, error) {
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

// computeCentroidDistanceMatrix, as said in the
// function name, computes the centroid distance
// matrix, saving it to the model.
//
// Note that because we only need 0.5*dist[i][j]
// in the algorithm, that's what's computed here
// and not just the distances
func (k *TriangleKMeans) computeCentroidDistanceMatrix() {
	// note that we can just compute the lower triangle
	// and then copy values over to maintain functionality
	for i := range k.Centroids {
		for j := 0; j < i; j++ {
			k.centroidDist[i][j] = 0.5 * diff(k.Centroids[i], k.Centroids[j])
		}
	}

	// now that we've computed the lower triangle, copy
	// values over to the other half
	for i := range k.Centroids {
		for j := len(k.Centroids) - 1; j > i; j-- {
			k.centroidDist[i][j] = k.centroidDist[j][i]
		}

	}

	// compute the min distance to any cluster
	// to save
	for i := range k.Centroids {
		var min float64
		if i == 0 && len(k.Centroids) > 1 {
			min = k.centroidDist[i][1]
		} else {
			min = k.centroidDist[i][0]
		}

		for j := range k.Centroids {
			if i == j {
				continue
			}
			if k.centroidDist[i][j] < min {
				min = k.centroidDist[i][j]
			}
		}

		k.minCentroidDist[i] = min
	}
}

// recalculateCentroids assigns each centroid to
// the mean of all points assigned to it. This
// method is abstracted within the Triangle
// accelerated KMeans variant because you are
// skipping a lot of distance calculations within
// the actual algorithm, so finding the mean of
// assigned points is harder to embed.
//
// The method returns the new centers instead of
// modifying the model's centroids.
func (k *TriangleKMeans) recalculateCentroids() [][]float64 {
	classTotal := make([][]float64, len(k.Centroids))
	classCount := make([]int64, len(k.Centroids))

	for j := range k.Centroids {
		classTotal[j] = make([]float64, len(k.trainingSet[0]))
	}

	for i, x := range k.trainingSet {
		classCount[k.guesses[i]]++
		for j := range x {
			classTotal[k.guesses[i]][j] += x[j]
		}
	}

	centroids := append([][]float64{}, k.Centroids...)
	for j := range centroids {
		// if no objects are in the same class,
		// reinitialize it to a random vector
		if classCount[j] == 0 {
			for l := range centroids[j] {
				centroids[j][l] = 10 * (rand.Float64() - 0.5)
			}
			continue
		}

		for l := range centroids[j] {
			centroids[j][l] = classTotal[j][l] / float64(classCount[j])
		}
	}

	return centroids
}

// Learn takes the struct's dataset and expected results and runs
// batch gradient descent on them, optimizing theta so you can
// predict based on those results
//
// This batch version of the model uses the k-means++
// instantiation method to generate a consistantly better
// model than regular, randomized instantiation of
// centroids.
// Paper: http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
//
// As stated many times for TriangleKMeans (check out the
// struct comments) this model uses the Triangle Inequality
// to decrease significantly the number of required distance
// calculations. The origininal paper is seen here:
//     http://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf
func (k *TriangleKMeans) Learn() error {
	if k.trainingSet == nil {
		err := fmt.Errorf("ERROR: Attempting to learn with no training examples!\n")
		fmt.Fprintf(k.Output, err.Error())
		return err
	}

	examples := len(k.trainingSet)
	if examples == 0 || len(k.trainingSet[0]) == 0 {
		err := fmt.Errorf("ERROR: Attempting to learn with no training examples!\n")
		fmt.Fprintf(k.Output, err.Error())
		return err
	}

	centroids := len(k.Centroids)
	features := len(k.trainingSet[0])

	fmt.Fprintf(k.Output, "Training:\n\tModel: Triangle Inequality Accelerated K-Means++ Classification\n\tTraining Examples: %v\n\tFeatures: %v\n\tClasses: %v\n...\n\n", examples, features, centroids)

	/* Step 0 */

	// instantiate the centroids using k-means++
	k.Centroids[0] = k.trainingSet[rand.Intn(len(k.trainingSet))]

	distances := make([]float64, len(k.trainingSet))
	for i := 1; i < len(k.Centroids); i++ {
		var sum float64
		for j, x := range k.trainingSet {
			minDiff := diff(x, k.Centroids[0])
			for l := 1; l < i; l++ {
				difference := diff(x, k.Centroids[l])
				if difference < minDiff {
					minDiff = difference
				}
			}

			distances[j] = minDiff * minDiff
			sum += distances[j]
		}

		target := rand.Float64() * sum
		j := 0
		for sum = distances[0]; sum < target; sum += distances[j] {
			j++
		}
		k.Centroids[i] = k.trainingSet[j]
	}

	/* Step 0.5 */

	// loop over dataset and assign each point to
	// the closest cluster
	for i, x := range k.trainingSet {
		k.guesses[i] = 0
		minDiff := diff(x, k.Centroids[0])
		k.info[i].lower[0] = minDiff
		for j := 1; j < len(k.Centroids); j++ {
			// avoid redundant distance computations
			if k.minCentroidDist[j] >= minDiff {
				continue
			}

			difference := diff(x, k.Centroids[j])
			k.info[i].lower[j] = difference
			if difference < minDiff {
				minDiff = difference
				k.guesses[i] = j
			}
		}

		// assign upper bound to the distance
		// to the nearest centroid.
		k.info[i].upper = minDiff
	}

	iter := 0
	for ; iter < k.maxIterations; iter++ {

		/* Step 1 */
		// compute the centroid distance matrix
		// and minimum distances to adjacent
		// centroids
		k.computeCentroidDistanceMatrix()

		var upper float64
		for i, x := range k.trainingSet {
			upper = k.info[i].upper
			/* Step 2 */
			if upper <= k.minCentroidDist[k.guesses[i]] {
				continue
			}

			/* Step 3 */
			for j := range k.Centroids {
				if j == k.guesses[i] && //                        (i)
					upper <= k.info[i].lower[j] && //             (ii)
					upper <= k.centroidDist[k.guesses[i]][j] { // (iii)
					continue
				}
				guess := k.guesses[i]

				/* Step 3.a */
				// proactively use the otherwise case
				distToCentroid := upper
				if k.info[i].recompute {
					// then recompute the distance to the assigned
					// centroid
					distToCentroid = diff(x, k.Centroids[guess])
					k.info[i].lower[guess] = distToCentroid
					k.info[i].upper = distToCentroid
					k.info[i].recompute = false
				}

				/* Step 3.b */
				if distToCentroid > k.info[i].lower[j] ||
					distToCentroid > k.centroidDist[guess][j] {
					// only now compute the distance to the
					// centroid
					dist := diff(x, k.Centroids[j])
					k.info[i].lower[j] = dist
					if dist < distToCentroid {
						k.guesses[i] = j

					}
				}
			}
		}

		/* Step 4 */
		newCentroids := k.recalculateCentroids()
		for i := range k.trainingSet {
			/* Step 5 */
			for j := range k.Centroids {
				// calculate the shift to the new centroid
				shift := k.info[i].lower[j] - diff(k.Centroids[j], newCentroids[j])

				// bound the shift at 0 and assign it
				// as the new lower bound
				if shift < 0 {
					shift = 0
				}
				k.info[i].lower[j] = shift
			}

			/* Step 6 */
			// reassign the upper bound to account
			// for the centroid shift
			k.info[i].upper += diff(newCentroids[k.guesses[i]], k.Centroids[k.guesses[i]])
			k.info[i].recompute = true
		}

		/* Step 7 */
		k.Centroids = newCentroids
	}

	fmt.Fprintf(k.Output, "Training Completed in %v iterations.\n%v\n", iter, k)

	return nil
}

// String implements the fmt interface for clean printing. Here
// we're using it to print the model as the equation h(θ)=...
// where h is the k-means hypothesis model
func (k *TriangleKMeans) String() string {
	return fmt.Sprintf("h(θ,x) = argmin_j | x[i] - μ[j] |^2\n\tμ = %v", k.Centroids)
}

// Guesses returns the hidden parameter for the
// unsupervised classification assigned during
// learning.
//
//    model.Guesses[i] = E[k.trainingSet[i]]
func (k *TriangleKMeans) Guesses() []int {
	return k.guesses
}

// Distortion returns the distortion of the clustering
// currently given by the k-means model. This is the
// function the learning algorithm tries to minimize.
//
// Distorition() = Σ |x[i] - μ[c[i]]|^2
// over all training examples
func (k *TriangleKMeans) Distortion() float64 {
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
func (k *TriangleKMeans) SaveClusteredData(filepath string) error {
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
func (k *TriangleKMeans) PersistToFile(path string) error {
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
func (k *TriangleKMeans) RestoreFromFile(path string) error {
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
