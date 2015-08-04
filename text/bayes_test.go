package text

import (
	"fmt"
	"os"
	"testing"

	"github.com/cdipaolo/goml/base"

	"github.com/stretchr/testify/assert"
)

func init() {
	// create the /tmp/.goml/ dir for persistance testing
	// if it doesn't already exist!
	err := os.MkdirAll("/tmp/.goml", os.ModePerm)
	if err != nil {
		panic(fmt.Sprintf("You should be able to create the directory for goml model persistance testing.\n\tError returned: %v\n", err.Error()))
	}
}

func TestExampleClassificationShouldPass1(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.TextDatapoint, 100)
	errors := make(chan error)

	// make a new NaiveBayes model with
	// 2 classes expected (classes in
	// datapoints will now expect {0,1}.
	// in general, given n as the classes
	// variable, the model will expect
	// datapoint classes in {0,...,n-1})
	model := NewNaiveBayes(stream, 3, base.OnlyWordsAndNumbers)

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
		err, more := <-errors
		if more {
			fmt.Printf("Error passed: %v", err)
		} else {
			// training is done!
			break
		}
	}

	fmt.Printf("Words: %v", model.Words)

	// now you can predict like normal
	class := model.Predict("My mother is in Los Angeles") // 0
	assert.EqualValues(t, 0, class, "Class should be 0")
}

func TestAreaClassificationShouldPass1(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.TextDatapoint, 100)
	errors := make(chan error)

	// make a new NaiveBayes model with
	// 2 classes expected (classes in
	// datapoints will now expect {0,1}.
	// in general, given n as the classes
	// variable, the model will expect
	// datapoint classes in {0,...,n-1})
	model := NewNaiveBayes(stream, 3, base.OnlyWordsAndNumbers)

	go model.OnlineLearn(errors)

	stream <- base.TextDatapoint{
		X: "Indian cities look alright",
		Y: 2,
	}

	stream <- base.TextDatapoint{
		X: "New Delhi, a city in India, gets very hot",
		Y: 2,
	}

	stream <- base.TextDatapoint{
		X: "Indian food is oftentimes based on vegetables",
		Y: 2,
	}

	stream <- base.TextDatapoint{
		X: "China is a large country",
		Y: 0,
	}

	stream <- base.TextDatapoint{
		X: "Chinese food tastes good",
		Y: 0,
	}

	stream <- base.TextDatapoint{
		X: "Chinese, as a country, has a lot of people in it",
		Y: 0,
	}

	stream <- base.TextDatapoint{
		X: "Japan makes sushi and cars",
		Y: 1,
	}

	stream <- base.TextDatapoint{
		X: "Many Japanese people are Buddhist",
		Y: 1,
	}

	stream <- base.TextDatapoint{
		X: "Japanese architecture looks nice",
		Y: 1,
	}

	close(stream)

	for {
		err, more := <-errors
		if more {
			fmt.Printf("Error passed: %v", err)
		} else {
			// training is done!
			break
		}
	}

	fmt.Printf("Words: %v", model.Words)

	// now you can predict like normal
	class := model.Predict("a lot of Japanese People live in Japan") // 0
	assert.EqualValues(t, 1, class, "Class should be 0")
}
