package text

import (
	"fmt"
	"os"
	"sort"
	"testing"

	"github.com/cdipaolo/goml/base"

	"github.com/stretchr/testify/assert"
)

func init() {
	// create the /tmp/.goml/ dir for persistance testing
	// if it doesn't already exist!
	err := os.MkdirAll("/tmp/.goml", os.ModePerm)
	if err != nil {
		panic(err.Error())
	}
}

func TestExampleTFIDFShouldPass1(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.TextDatapoint, 40)
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
	}

	stream <- base.TextDatapoint{
		X: "I hate Los Angeles",
	}

	stream <- base.TextDatapoint{
		X: "My mother is not a nice lady lady lady lady",
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

	// cast NaiveBayes model to TFIDF
	tf := TFIDF(*model)

	greater := tf.TFIDF("I", "I don't think my mother is not a nice lady and I know you're wrong and I can prove it!!!!")
	lesser := tf.TFIDF("lady", "I don't think my mother is not a nice lady and I know you're wrong and I can prove it!!!!")
	assert.True(t, greater > lesser, "TFIDF for 'I' (%v) should be greater than TFIDF for 'lady' (%v)", greater, lesser)

	freq := tf.MostImportantWords("I don't think my mother is not a nice lady and I know you're wrong!", 4)
	assert.Len(t, freq, 4, "Length of Frequencies (%v) should be 4", freq)
	assert.True(t, sort.IsSorted(sort.Reverse(freq)), "Frequencies (%v) should be sorted!", freq)

	fmt.Printf("Freq: %v\nI: %v\tlady: %v\n\n", freq, greater, lesser)
}

func TestAreaTFIDFShouldPass1(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.TextDatapoint, 40)
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
	}

	stream <- base.TextDatapoint{
		X: "New Delhi, a city in India, gets very hot",
	}

	stream <- base.TextDatapoint{
		X: "Indian food is oftentimes based on vegetables",
	}

	stream <- base.TextDatapoint{
		X: "China is a large country",
	}

	stream <- base.TextDatapoint{
		X: "Chinese food tastes good",
	}

	stream <- base.TextDatapoint{
		X: "Chinese, as a country, has a lot of people in it",
	}

	stream <- base.TextDatapoint{
		X: "Japan makes sushi and cars",
	}

	stream <- base.TextDatapoint{
		X: "Many Japanese people are Buddhist",
	}

	stream <- base.TextDatapoint{
		X: "Japanese architecture looks nice",
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

	// cast NaiveBayes model to TFIDF
	tf := TFIDF(*model)

	greater := tf.TFIDF("sushi", "sushi is my favorite buddhist related food and sushi is fun")
	lesser := tf.TFIDF("buddhist", "sushi is my favorite buddhist related food and sushi is fun")
	assert.True(t, greater > lesser, "TFIDF for 'buddhist' (%v) should be less than TFIDF for 'sushi' (%v)", lesser, greater)

	freq := tf.MostImportantWords("Sushi is really great and sushi is awesome and sushi sushi sushi sushi sushi sushi!!", 4)
	assert.Len(t, freq, 4, "Length of Frequencies (%v) should be 4", freq)
	assert.True(t, sort.IsSorted(sort.Reverse(freq)), "Frequencies (%v) should be sorted!", freq)

	fmt.Printf("Freq: %v\nSushi: %v\tBuddhist: %v\n\n", freq, greater, lesser)
}
