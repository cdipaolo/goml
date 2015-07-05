## Perceptron
### `import "github.com/cdipaolo/goml/perceptron"`

[![GoDoc](https://godoc.org/github.com/cdipaolo/goml/perceptron?status.svg)](https://godoc.org/github.com/cdipaolo/goml/perceptron)

The perceptron model holds easy to implement, online, reactive perceptrons that work with Golang channels (streams) of data to make pipelined training more efficient.

### implemented models

- [binary, online perceptron](perceptron.go)

# example binary, online perceptron

This example is pretty much verbatim from the tests. If you want to see other simple examples of Perceptrons in action, check out the tests for each model!

```go
// create the channel of data and errors
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