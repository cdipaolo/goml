package perceptron

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/cdipaolo/goml/base"
)

// KernelPerceptron represents the perceptron online
// learning model, where you input features and
// the model's state reacts to the input and changes
// weights (parameter vector theta) only when
// the guess by the algorithm is wrong. Unlike the
// base Perceptron model, which maps inputs using
// an affine feature space, the KernelPerceptron
// can use different kernels which may map to
// infinite-dimension feature spaces.
//
// The hypothesis is a generalized version of the
// regular perceptron, where i∈M where M are all
// misclassified examples:
//      sgn(Σ αy[i] * K(x[i], x))
//
// In this implementation, data is passed through a
// channel, where the learn function is run in a
// separate goroutine and stops when the channel is
// closed.
//
// http://cs229.stanford.edu/notes/cs229-notes6.pdf
// https://en.wikipedia.org/wiki/Perceptron
// https://en.wikipedia.org/wiki/Kernel_perceptron
//
// KernelPerceptron implements the OnlineModel interface,
// not the Model interface, because it uses online
// learning only
//
// Data results in this binary class model are
// expected to be either -1 or 1 (ie. the
// base.Datapoint's you pass should, called point,
// have point.Y be either [-1] or [1])
//
// You must pass in a valid kernel function with
// the NewKernelPerceptron function. You can find
// premade, valid kernels in the `base` package
// if you want to use those.
type KernelPerceptron struct {
	// SV stores the KernelPerceptron's support
	// vectors
	SV []base.Datapoint `json:"support_vectors,omitempty"`

	Kernel func([]float64, []float64) float64
}

// NewKernelPerceptron takes in a learning rate alpha, the
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
//
// Weight is the importance given to newer support
// vectors in prediction. Should be 0 < w
func NewKernelPerceptron(kernel func([]float64, []float64) float64) *KernelPerceptron {
	return &KernelPerceptron{
		Kernel: kernel,
	}
}

// Predict takes in a variable x (an array of floats,) and
// finds the value of the hypothesis function given the
// current parameter vector θ
func (p *KernelPerceptron) Predict(x []float64, normalize ...bool) ([]float64, error) {
	if len(normalize) != 0 && normalize[0] {
		base.NormalizePoint(x)
	}

	var sum float64
	for i := range p.SV {
		sum += p.SV[i].Y[0] * p.Kernel(p.SV[i].X, x)
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
// the perceptron updates it's support vectors
// This acts almost like a callback and
// passes the newly added support vector
// as a slice of floats.
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
// stream. This will potentially help the optimization
// converge faster. This is given as a parameter because
// you won't have direct access to the dataset before
// hand like you would in batch/stochastic settings.
//
// Example Online Kernel Perceptron:
//
//     // create the channel of data and errors
//     stream := make(chan base.Datapoint, 100)
//     errors := make(chan error)
//
//     // The kernel could be any kernel from the Base
//     // package, or it could be your own function!
//     // I suggest you look at the code for the kernels
//     // if you want to make sense of them. It's pretty
//     // intuitive and simple.
//     model := NewKernelPerceptron(base.GaussianKernel(50))
//
//     go model.OnlineLearn(errors, stream, func(SV [][]float64) {
//         // do something with the newly added support
//         // vector (persist to database?) in here.
//     })
//
//     go func() {
//         for iterations := 0; iterations < 20; iterations++ {
//             for i := -200.0; abs(i) > 1; i *= -0.7 {
//                 for j := -200.0; abs(j) > 1; j *= -0.7 {
//                     for k := -200.0; abs(k) > 1; k *= -0.7 {
//                         for l := -200.0; abs(l) > 1; l *= -0.7 {
//                             if i/2+2*k-4*j+2*l+3 > 0 {
//                                 stream <- base.Datapoint{
//                                     X: []float64{i, j, k, l},
//                                     Y: []float64{1.0},
//                                 }
//                             } else {
//                                 stream <- base.Datapoint{
//                                     X: []float64{i, j, k, l},
//                                     Y: []float64{-1.0},
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//
//         // close the dataset to tell the model
//         // to stop learning when it finishes reading
//         // what's left in the channel
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
func (p *KernelPerceptron) OnlineLearn(errors chan error, dataset chan base.Datapoint, onUpdate func([][]float64), normalize ...bool) {
	if dataset == nil {
		errors <- fmt.Errorf("ERROR: Attempting to learn with a nil data stream!\n")
		close(errors)
		return
	}

	if errors == nil {
		errors = make(chan error)
	}

	fmt.Printf("Training:\n\tModel: Kernel Perceptron Classifier\n\tOptimization Method: Online Kernel Perceptron\n...\n\n")

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

			// update the parameters if the guess
			// is wrong
			if guess[0] != point.Y[0] {
				p.SV = append(p.SV, point)

				// call the OnUpdate callback with the new vector
				// appended to a blank slice so the vector is
				// passed by value and not by reference
				go onUpdate([][]float64{append(point.X, point.Y...)})
			}

		} else {
			fmt.Printf("Training Completed.\n%v\n\n", p)
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
//     h(θ,x) = Σ y[i]*K(x[i], x`) > 0 ? 1 : 0
func (p *KernelPerceptron) String() string {
	return fmt.Sprintf("h(θ,x) = Σ y[i]*K(x[i], x`) > 0 ? 1 : 0\n\tTotal Support Vectors: %v\n", len(p.SV))
}

// PersistToFile takes in an absolute filepath and saves the
// parameter vector θ to the file, which can be restored later.
// The function will take paths from the current directory, but
// functions
//
// The data is stored as JSON because it's one of the most
// efficient storage method (you only need one comma extra
// per feature + two brackets, total!) And it's extendable.
func (p *KernelPerceptron) PersistToFile(path string) error {
	if path == "" {
		return fmt.Errorf("ERROR: you just tried to persist your model to a file with no path!! That's a no-no. Try it with a valid filepath")
	}

	bytes, err := json.Marshal(p.SV)
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
func (p *KernelPerceptron) RestoreFromFile(path string) error {
	if path == "" {
		return fmt.Errorf("ERROR: you just tried to restore your model from a file with no path! That's a no-no. Try it with a valid filepath")
	}

	bytes, err := ioutil.ReadFile(path)
	if err != nil {
		return err
	}

	err = json.Unmarshal(bytes, &p.SV)
	if err != nil {
		return err
	}

	return nil
}
