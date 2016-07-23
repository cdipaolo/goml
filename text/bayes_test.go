package text

import (
	"fmt"
	"io/ioutil"
	"os"
	"strings"
	"sync"
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

	// now you can predict like normal
	class := model.Predict("My mo~~~ther is in Los Angeles") // 0
	assert.EqualValues(t, 0, class, "Class should be 0")

	// test small document classification
	class, p := model.Probability("Mother Los Angeles")
	assert.EqualValues(t, 0, class, "Class should be 0")
	assert.True(t, p > 0.75, "There should be a greater than 75 percent chance the document is negative - Given %v", p)

	class, p = model.Probability("Love the CiTy")
	assert.EqualValues(t, 1, class, "Class should be 0")
	assert.True(t, p > 0.75, "There should be a greater than 75 percent chance the document is positive - Given %v", p)
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

	// now you can predict like normal
	class := model.Predict("a lot of Japanese People live in Japan")
	assert.EqualValues(t, 1, class, "Class should be 1")
}

func TestPersistPerceptronShouldPass1(t *testing.T) {
	// create the channel of data and errors
	stream := make(chan base.TextDatapoint, 100)
	errors := make(chan error)

	model := NewNaiveBayes(stream, 3, base.OnlyWordsAndNumbers)

	go model.OnlineLearn(errors)

	stream <- base.TextDatapoint{
		X: "I love the city",
		Y: 0,
	}

	stream <- base.TextDatapoint{
		X: "I hate Los Angeles",
		Y: 1,
	}

	stream <- base.TextDatapoint{
		X: "My mother is not a nice lady",
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

	// now you can predict like normal
	class := model.Predict("My mother is in Los Angeles") // 0
	assert.EqualValues(t, 1, class, "Class should be 0")

	// now persist to file
	err := model.PersistToFile("/tmp/.goml/NaiveBayes.json")
	assert.Nil(t, err, "Persistance error should be nil")

	// reset model

	model = NewNaiveBayes(stream, 3, base.OnlyWordsAndNumbers)

	class = model.Predict("My mother is in Los Angeles") // 0
	assert.EqualValues(t, 0, class, "Class should be 0")

	// restore from file
	err = model.RestoreFromFile("/tmp/.goml/NaiveBayes.json")
	assert.Nil(t, err, "Persistance error should be nil")

	// now you can predict like normal
	class = model.Predict("My mother is in Los Angeles") // 0
	assert.EqualValues(t, 1, class, "Class should be 0")

	// reset again

	model = NewNaiveBayes(stream, 3, base.OnlyWordsAndNumbers)

	class = model.Predict("My mother is in Los Angeles") // 0
	assert.EqualValues(t, 0, class, "Class should be 0")

	// restore file straight from bytes now
	bytes, err := ioutil.ReadFile("/tmp/.goml/NaiveBayes.json")
	assert.Nil(t, err, "Read file error should be nil")

	assert.Nil(t, model.Restore(bytes), "Model restore error should be nil")

	class = model.Predict("My mother is in Los Angeles") // 0
	assert.EqualValues(t, 1, class, "Class should be 0")
}

// make sure that calling predict while the model is still training does
// not cause a runtime panic because of concurrent map reads & writes
func TestConcurrentPredictionAndLearningShouldNotFail(t *testing.T) {
	c := make(chan base.TextDatapoint, 100)
	model := NewNaiveBayes(c, 2, base.OnlyWords)
	errors := make(chan error)

	// fill the buffer
	var i uint8
	for i = 0; i < 99; i++ {
		c <- base.TextDatapoint{
			X: strings.Repeat("a whole bunch of words that will take some time to iterate through", 50),
			Y: i % 2,
		}
	}

	// spin off a "long" running loop of predicting
	// and then start another goroutine for OnlineLearn
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		// fmt.Println("beginning predicting")
		for i := 0; i < 500; i++ {
			model.Predict(strings.Repeat("some stuff that might be in the training data like iterate", 25))
		}
		// fmt.Println("done predicting")
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		// fmt.Println("beginning learning")
		model.OnlineLearn(errors)
		// fmt.Println("done learning")
	}()

	go func() {
		for err, more := <-errors; more; err, more = <-errors {
			if err != nil {
				t.Logf("Error passed: %s\n", err.Error())
				t.Fail()
			}
		}
	}()

	close(c)
	wg.Wait()
}
