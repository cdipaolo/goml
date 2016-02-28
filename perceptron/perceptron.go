// Package perceptron holds the online
// perceptron model of learning. A perceptron
// works by 'reacting' to bad predictions, and
// only updating it's parameter vector when it
// does make a bad prediction. If you want to
// read more about the details of the perceptron
// itself, go to the Perceptron struct documentation.
//
// The package implements the training of a
// perceptron as running on a buffered channel
// of base.Datapoint's. This lets you run the
// learning of the model reactively off of a data
// stream, an API for example, only pushing new
// data into the pipeline when it's recieved.
// You are given an OnUpdate callback with the
// Perceptron struct, which is called whenever
// the model updates it parameter vector. It passes
// a copy of the new parameter vector as a copy
// and runs the callback in a new goroutine.
// This would let the user persist the model to
// a database of their choosing in realtime,
// calling update to a table consistantly within
// the callback.
//
// The Perceptron also takes in a channel of
// errors when it learns, which lets the user
// see any errors while learning but not actually
// interrupting the learning itself. The model
// just ignores errors (usually caused by a
// mismatch of dimension on the input vector)
// and goes to the next datapoint. The channel
// of errors is closed when learning is done
// so you know when your model is finished working
// its way though the dataset (this implies that
// you closed the data stream, though.)
//
// Example Online, Binary Perceptron (no layers, etc.):
//
//      // create the channel of data and errors
//      stream := make(chan base.Datapoint, 100)
//      errors := make(chan error)
//
//      model := NewPerceptron(0.1, 1, stream)
//
//      go model.OnlineLearn(errors, stream, func (theta []float64) {
//          fmt.Fprintf(p.Output, "Theta updated to %v!\n", theta)
//      })
//
//      // start passing data to our datastream
//      //
//      // we could have data already in our channel
//      // when we instantiated the Perceptron, though
//      //
//      // and note that this data could be coming from
//      // some web server, or whatever!!
// 		go func() {
// 			for i := -500.0; abs(i) > 1; i *= -0.997 {
// 				if 10 + (i-20)/2 > 0 {
// 					stream <- base.Datapoint{
// 						X: []float64{i-20},
// 						Y: []float64{1.0},
// 					}
// 				} else {
// 					stream <- base.Datapoint{
// 						X: []float64{i-20},
// 						Y: []float64{0},
// 			        }
// 			    }
// 			}
// 		}()
//
//      // close the dataset
//      close(stream)
//      for {
//          err, more := <- errors
//          if err != nil {
//              fmt.Fprintf(p.Output, "Error passed: %v", err)
//          } else {
//              // training is done!
//              break
//          }
//      }
//
//      // now you can predict!!
//      // note that guess is a []float64 of len() == 1
//      // when it isn't nil
//      guess, err := model.Predict([]float64{i})
//      if err != nil {
//           panic("EGATZ!! I FOUND AN ERROR! BETTER CHECK YOUR INPUT DIMENSIONS!")
//      }
package perceptron

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"

	"github.com/cdipaolo/goml/base"
)

// Perceptron represents the perceptron online
// learning model, where you input features and
// the model's state reacts to the input and changes
// weights (parameter vector theta) only when
// the guess by the algorithm is wrong
//
// The hypothesis of the Perceptron is similar to
// logistic regression in that the extremes tend
// to 0 and 1, but the actual hypothesis runs the
// input with weights through a step function, not
// a sigmoid, namely:
//     if (θx * y < 0) {
//         θ := θ + α*yx
//     }
//
// In this implementation, data is passed through a
// channel, where the learn function is run in a
// separate goroutine and stops when the channel is
// closed.
//
// http://cs229.stanford.edu/notes/cs229-notes6.pdf
// https://en.wikipedia.org/wiki/Perceptron
//
// Perceptron implements the OnlineModel interface,
// not the Model interface, because it uses online
// learning
//
// Unlike the General Linear Models, for example,
// the data is expected to be passed as a Dataset
// struct so it can be easily passed through the
// data pipeline channel
//
// Data results in this binary class model are
// expected to be either -1 or 1 (ie. the
// base.Datapoint's you pass should, called point,
// have point.Y be either [-1] or [1])
type Perceptron struct {
	// alpha is the learning rate of the perceptron
	// algorithm
	alpha float64

	Parameters []float64 `json:"theta"`

	// Output is the io.Writer used for logging
	// and printing. Defaults to os.Stdout.
	Output io.Writer
}

// NewPerceptron takes in a learning rate alpha, the
// number of features (not including the constant
// term) being evaluated by the model, the update
// callback called whenever the perceptron updates the
// parameter vector theta (whenever it makes a wrong
// guess), and a channel of datapoints that will be
// used in training and returns an instantiated model.
//
// Again! Features _does not_ include the constant
// term!
//
// Also, learning rate of 0.1 seems to work well in
// many cases. (I also heard that in a lecture video
// from a UW professor)
func NewPerceptron(alpha float64, features int) *Perceptron {
	var params []float64
	params = make([]float64, features+1)

	return &Perceptron{
		alpha: alpha,

		// initialize θ as the zero vector (that is,
		// the vector of all zeros)
		Parameters: params,
		Output:     os.Stdout,
	}
}

// UpdateLearningRate set's the learning rate of the model
// to the given float64.
func (p *Perceptron) UpdateLearningRate(a float64) {
	p.alpha = a
}

// Predict takes in a variable x (an array of floats,) and
// finds the value of the hypothesis function given the
// current parameter vector θ
func (p *Perceptron) Predict(x []float64, normalize ...bool) ([]float64, error) {
	if len(x)+1 != len(p.Parameters) {
		return nil, fmt.Errorf("Error: Parameter vector should be 1 longer than input vector!\n\tLength of x given: %v\n\tLength of parameters: %v\n", len(x), len(p.Parameters))
	}

	if len(normalize) != 0 && normalize[0] {
		base.NormalizePoint(x)
	}

	// include constant term in sum
	sum := p.Parameters[0]

	for i := range x {
		sum += x[i] * p.Parameters[i+1]
	}

	result := -1.0
	if sum > 0 {
		result = 1
	}

	return []float64{result}, nil
}

// OnlineLearn runs off of the datastream within the Perceptron
// structure. Whenever the model makes a wrong prediction
// the parameter vector theta is updated to reflect that,
// as discussed in the documentation for the Perceptron
// struct itself, and the OnUpdate function is called with
// the newly updated parameter vector. Learning will stop
// when the data channel is closed and all remaining
// datapoints within the channel have been read.
//
// The errors channel will be closed when learning is
// completed so you know when it's done if you're relying
// on that for whatever reason
//
// onUpdate func ([]float64):
//
// onUpdate is a function that is called whenever
// the perceptron updates it's parameter vector
// theta. This acts almost like a callback and
// passes the newly updated parameter vector
// theta as a slice of floats.
//
// This might be useful is you want to maintain
// an up to date persisted model in a database of
// your choosing and you'd like to update it
// constantly.
//
// This will be spawned into a new goroutine, so
// don't worry about the function taking a long
// time, or blocking.
//
// If you want to monitor errors happening within
// this function, just have a channel of errors
// you send do within this channel, or some other
// method if it fits your scenario better.
//
// NOTE that there is an optional last parameter which,
// when true, will normalize all data given on the
// stream. This will potentially help gradient descent
// converge faster. This is given as a parameter because
// you won't have direct access to the dataset before
// hand like you would in batch/stochastic settings.
//
// Example Online, Binary Perceptron (no layers, etc.):
//
//      // create the channel of data and errors
//      stream := make(chan base.Datapoint, 100)
//      errors := make(chan error)
//
//      model := NewPerceptron(0.1, 1, stream)
//
//      go model.OnlineLearn(errors, stream, func (theta []float64) {
//          // do something with the new theta (persist
//          // to database?) in here.
//          fmt.Fprintf(p.Output, "Theta updated to %v!\n", theta)
//      })
//
//      // start passing data to our datastream
//      //
//      // we could have data already in our channel
//      // when we instantiated the Perceptron, though
//      //
//      // and note that this data could be coming from
//      // some web server, or whatever!!
// 	    go func() {
//             for i := -500.0; abs(i) > 1; i *= -0.997 {
//                 if 10 + (i-20)/2 > 0 {
//                     stream <- base.Datapoint{
//                         X: []float64{i-20},
//                         Y: []float64{1.0},
//                     }
//                 } else {
//                     stream <- base.Datapoint{
//                         X: []float64{i-20},
//                         Y: []float64{0},
//                     }
//                 }
//             }
//         }()
//
//      // close the dataset
//      close(stream)
//      for {
//          err, more := <- errors
//          if err != nil {
//              fmt.Fprintf(p.Output, "Error passed: %v", err)
//          } else {
//              // training is done!
//              break
//          }
//      }
//
//      // now you can predict!!
//      // note that guess is a []float64 of len() == 1
//      // when it isn't nil
//      guess, err := model.Predict([]float64{i})
//      if err != nil {
//           panic("EGATZ!! I FOUND AN ERROR! BETTER CHECK YOUR INPUT DIMENSIONS!")
//      }
func (p *Perceptron) OnlineLearn(errors chan error, dataset chan base.Datapoint, onUpdate func([][]float64), normalize ...bool) {
	if errors == nil {
		errors = make(chan error)
	}
	if dataset == nil {
		errors <- fmt.Errorf("ERROR: Attempting to learn with a nil data stream!\n")
		close(errors)
		return
	}

	fmt.Fprintf(p.Output, "Training:\n\tModel: Perceptron Classifier\n\tOptimization Method: Online Perceptron\n\tFeatures: %v\n\tLearning Rate α: %v\n...\n\n", len(p.Parameters), p.alpha)

	norm := len(normalize) != 0 && normalize[0]

	var point base.Datapoint
	var more bool

	for {
		point, more = <-dataset

		if more {
			// have a datapoint, predict and update!
			//
			// Predict also checks if the point is of the
			// correct dimensions
			if norm {
				base.NormalizePoint(point.X)
			}

			guess, err := p.Predict(point.X)
			if err != nil {
				// send the error channel some info and
				// skip this datapoint
				errors <- err
				continue
			}

			if len(point.Y) != 1 {
				errors <- fmt.Errorf("The binary perceptron model requires that the data results (y) have length 1 - given %v", len(point.Y))
				continue
			}

			if len(point.X) != len(p.Parameters)-1 {
				errors <- fmt.Errorf("The binary perceptron model requires that the length of input data (currently %v) be one less than the length of the parameter vector (%v)", len(point.X), len(p.Parameters))
				continue
			}

			// update the parameters if the guess
			// is wrong
			if guess[0] != point.Y[0] {
				p.Parameters[0] += p.alpha * (point.Y[0] - guess[0])

				for i := 1; i < len(p.Parameters); i++ {
					p.Parameters[i] += p.alpha * (point.Y[0] - guess[0]) * point.X[i-1]
				}

				// call the OnUpdate callback with the new theta
				// appended to a blank slice so the vector is
				// passed by value and not by reference
				go onUpdate([][]float64{p.Parameters})
			}

		} else {
			fmt.Fprintf(p.Output, "Training Completed.\n%v\n\n", p)
			close(errors)
			return
		}
	}
}

// String implements the fmt interface for clean printing. Here
// we're using it to print the model as the equation h(θ)=...
// where h is the perceptron hypothesis model.
//
// Note that I'm using the terniary operator to represent
// the perceptron:
//     h(θ,x) = θx > 0 ? 1 : 0
func (p *Perceptron) String() string {
	features := len(p.Parameters) - 1
	if len(p.Parameters) == 0 {
		fmt.Fprintf(p.Output, "ERROR: Attempting to print model with the 0 vector as it's parameter vector! Train first!\n")
	}
	var buffer bytes.Buffer

	buffer.WriteString(fmt.Sprintf("h(θ,x) = θx > 0 ? 1 : 0\nθx = %.3f + ", p.Parameters[0]))

	length := features + 1
	for i := 1; i < length; i++ {
		buffer.WriteString(fmt.Sprintf("%.5f(x[%d])", p.Parameters[i], i))

		if i != features {
			buffer.WriteString(fmt.Sprintf(" + "))
		}
	}

	return buffer.String()
}

// PersistToFile takes in an absolute filepath and saves the
// parameter vector θ to the file, which can be restored later.
// The function will take paths from the current directory, but
// functions
//
// The data is stored as JSON because it's one of the most
// efficient storage method (you only need one comma extra
// per feature + two brackets, total!) And it's extendable.
func (p *Perceptron) PersistToFile(path string) error {
	if path == "" {
		return fmt.Errorf("ERROR: you just tried to persist your model to a file with no path!! That's a no-no. Try it with a valid filepath")
	}

	bytes, err := json.Marshal(p.Parameters)
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
func (p *Perceptron) RestoreFromFile(path string) error {
	if path == "" {
		return fmt.Errorf("ERROR: you just tried to restore your model from a file with no path! That's a no-no. Try it with a valid filepath")
	}

	bytes, err := ioutil.ReadFile(path)
	if err != nil {
		return err
	}

	err = json.Unmarshal(bytes, &p.Parameters)
	if err != nil {
		return err
	}

	return nil
}
