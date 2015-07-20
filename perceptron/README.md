## [Perceptron](http://cs229.stanford.edu/notes/cs229-notes6.pdf)
### `import "github.com/cdipaolo/goml/perceptron"`

[![GoDoc](https://godoc.org/github.com/cdipaolo/goml/perceptron?status.svg)](https://godoc.org/github.com/cdipaolo/goml/perceptron)

The perceptron model holds easy to implement, online, reactive perceptrons that work with Golang channels (streams) of data to make pipelined training more efficient.

The perceptron is similar to regular Logistic Regression, except that it returns discrete values rather than the probability of a result being true (or 1,) which could be 0.76543, for example. The perceptron uses a setp function instead of a sigmoid transform its inputs into a hypothesis. It's model comes primarily from biology theory, representing a neuron in the brain.

![The Perceptron](https://upload.wikimedia.org/wikipedia/commons/8/8c/Perceptron_moj.png)

The optimization method for a perceptron also operates differently than logistic regression, which results in some cool properties. The perceptron doesn't update for each training example. Instead, it guesses what the correct classification should be, then only if it gets it wrong will the perceptron update it's parameter vector θ (also known as 'the weights.') What this allows you to do is constantly feed data into a perceptron, whereby it can be continually update and learn (when learning, obviously this won't work when predicting as this is not an unsupervised model.) Note the update rule below that we're using within the [binary, online perceptron](perceptron.go) if this doesn't make much sense initially.

```go
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
	go p.OnUpdate(append([]float64{}, p.Parameters...))
}
```

It should be noted that you want interspersion between the results you pass, that is, you want to 'alternate' sending positive and negative results with a decently quick frequency. If you instead pass a string of 500 negative results followed by 500 positive ones, the perceptron will start by thinking that all results are negative, then correcting completely and thinking that most results are positive, but it will never converge along the optimal boundary hyperplane between the binary classification.

### implemented models

- [binary, online perceptron](perceptron.go)
- [binary, online kernel perceptron](kernel_perceptron.go)
	* this model uses more memory than the regular perceptron, but by using the kernel trick it allows you to input theoretically infinite feature spaces into it as well as fitting non-linear decision boundaries with the model! You can use ready-made (though custimizable) kernels from the `goml/base` package. It will take longer to train, as well.

# example binary, online perceptron

This example is pretty much verbatim from the tests. If you want to see other simple examples of Perceptrons in action, check out the tests for each model!

```go
// create the channel of data and errors
//
// note that we are buffering the data stream
// channel
stream := make(chan base.Datapoint, 100)
errors := make(chan error)

model := NewPerceptron(0.1, 1, func (theta []float64) {}, stream)

// make the model learn in a different goroutine,
// passing errors to our error channel so we know
// the lowdown on our training
go model.Learn(errors)

// start passing data to our stream
//
// we could have data already in our channel
// when we instantiated the Perceptron, though
//
// and note that this data could be coming from
// some web server, or whatever!!
for i := -500.0; abs(i) > 1; i *= -0.997 {
	if 10 + (i-20)/2 > 0 {
		stream <- base.Datapoint{
			X: []float64{i-20},
			Y: []float64{1.0},
		}
	} else {
		stream <- base.Datapoint{
			X: []float64{i-20},
			Y: []float64{0},
	    }
    }
}

// close the dataset
close(stream)
for {
    err, more := <- errors
    if more {
        // there is another error
        fmt.Printf("Error passed: %v", err)
    } else {
        // training is done!
        break
    }
}

// now you can predict!!
// note that guess is a []float64 of len() == 1
// when it isn't nil (it is in the case of 
// an error)
guess, err := model.Predict([]float64{i})
if err != nil {
     panic("EGATZ!! I FOUND AN ERROR! BETTER CHECK YOUR INPUT DIMENSIONS!")
}
```