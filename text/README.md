## Text Classification
### `import "github.com/cdipaolo/goml/text"`

[![GoDoc](https://godoc.org/github.com/cdipaolo/goml/text?status.svg)](https://godoc.org/github.com/cdipaolo/goml/text)

This package implements text classification algorithms. For algorithms that could be used numberically (most/all of them,) this package makes working with text documents easier than hand-rolling a bag-of-words model and integrating it with other models

### implemented models

- [multiclass naive bayes](bayes.go)

### example ordinary least squares

This is the general text classification example from the GoDoc package comment. Look there and at the tests for more detailed and varied examples of usage:
```go
// create the channel of data and errors
stream := make(chan base.TextDatapoint, 100)
errors := make(chan error)

// make a new NaiveBayes model with
// 2 classes expected (classes in
// datapoints will now expect {0,1}.
// in general, given n as the classes
// variable, the model will expect
// datapoint classes in {0,...,n-1})
//
// Note that the model is filtering
// the text to omit anything except
// words and numbers (and spaces
// obviously)
model := NewNaiveBayes(stream, 2, base.OnlyWordsAndNumbers)

go model.OnlineLearn(errors)

stream <- base.TextDatapoint{
	X: "I love the city",
	Y: 1,
}

stream <- base.TextDatapoint{
	X: "I hate Los Angeles",
	Y: 0,
}

stream <- base.TextDatapoint{
	X: "My mother is not a nice lady",
	Y: 0,
}

close(stream)

for {
	err, more := <- errors
	if err != nil {
		fmt.Printf("Error passed: %v", err)
	} else {
		// training is done!
		break
	}
}

// now you can predict like normal
class := model.Predict("My mother is in Los Angeles") // 0
```