package base

import (
	"fmt"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

var x [][]float64
var y []float64

func init() {

	// create the /tmp/.goml/ dir for persistance testing
	// if it doesn't already exist!
	err := os.MkdirAll("/tmp/.goml", os.ModePerm)
	if err != nil {
		panic(fmt.Sprintf("You should be able to create the directory for goml model persistance testing.\n\tError returned: %v\n", err.Error()))
	}

	x = [][]float64{}
	y = []float64{}

	for i := -100; i < 100; i++ {
		row := []float64{}
		for j := 0; j < 10; j++ {
			row = append(row, float64(j))
		}

		x = append(x, row)
		y = append(y, float64(i))
	}
}

func TestSaveDataToCSVShouldPass1(t *testing.T) {
	err := SaveDataToCSV("/tmp/.goml/CSV.csv", x, y, true)
	assert.Nil(t, err, "Error saving data should be nil")

	newX, newY, err := LoadDataFromCSV("/tmp/.goml/CSV.csv")
	assert.Nil(t, err, "Error loading CSV data should be nil")

	for i := -100; i < 100; i++ {
		for j := 0; j < 10; j++ {
			assert.Equal(t, float64(j), newX[i+100][j], "New value from saved CSV should match old value")
		}
		assert.Equal(t, float64(i), newY[i+100], "New y should from saved CSV should match old value")
	}
}

func TestSaveDataToCSVShouldPass2(t *testing.T) {
	err := SaveDataToCSV("/tmp/.goml/CSV2.csv", x, y, false)
	assert.Nil(t, err, "Error saving data should be nil")

	newX, newY, err := LoadDataFromCSV("/tmp/.goml/CSV2.csv")
	assert.Nil(t, err, "Error loading CSV data should be nil")

	for i := -100; i < 100; i++ {
		for j := 0; j < 10; j++ {
			assert.Equal(t, float64(j), newX[i+100][j], "New value from saved CSV should match old value")
		}
		assert.Equal(t, float64(i), newY[i+100], "New y should from saved CSV should match old value")
	}
}

func TestSaveDataToCSVShouldFail1(t *testing.T) {
	err := SaveDataToCSV("/tmp/.goml/CSVFail1.csv", x, y, true)
	assert.Nil(t, err, "Error saving data should be nil")

	_, _, err = LoadDataFromCSV("/tmp/.goml/THIS/PATH/DOES/NOT/EXIST/SDFGJHGFDSASDFGHJHGFDSDFGHJHGFDSDFGHJHGFDSDFGHJHGFDSDFGHJ/data.csv")
	assert.NotNil(t, err, "Error loading CSV data should not be nil")
}

func TestSaveDataToCSVShouldFail2(t *testing.T) {
	err := SaveDataToCSV("/tmp/.goml/CSVFail2.csv", [][]float64{}, y, false)
	assert.NotNil(t, err, "Error saving data should not be nil")
}

func TestSaveDataToCSVShouldFail3(t *testing.T) {
	err := SaveDataToCSV("/tmp/.goml/CSVFail3.csv", [][]float64{[]float64{}, []float64{}}, y, false)
	assert.NotNil(t, err, "Error saving data should not be nil")
}

func TestSaveDataToCSVShouldFail4(t *testing.T) {
	err := SaveDataToCSV("/tmp/.goml/CSVFail4.csv", x, []float64{}, false)
	assert.NotNil(t, err, "Error saving data should not be nil")
}

func TestSaveDataToCSVShouldFail5(t *testing.T) {
	err := SaveDataToCSV("/tmp/.goml/CSVFail5.csv", [][]float64{[]float64{1.0}, []float64{1.0}, []float64{1.0}}, []float64{1.0}, false)
	assert.NotNil(t, err, "Error saving data should not be nil")
}

func TestLoadDataFromCSVToStreamShouldPass1(t *testing.T) {
	err := SaveDataToCSV("/tmp/.goml/CSV_stream.csv", x, y, true)
	assert.Nil(t, err, "Error saving data should be nil")

	data := make(chan Datapoint, 100)
	errors := make(chan error)

	go LoadDataFromCSVToStream("/tmp/.goml/CSV_stream.csv", data, errors)

	point := Datapoint{}
	var more bool

	for i := -100; i < 100; i++ {
		point, more = <-data

		if more {
			assert.Equal(t, float64(i), point.Y[0], "Y should equal i-100")

			for j := 0; j < 10; j++ {
				assert.Equal(t, float64(j), point.X[j], "X[j] should equal j")
			}
		} else {
			assert.Equal(t, 99, i, "Stream should pass 200 examples")
			break
		}
	}

	count := 0
	for {
		_, more := <-errors
		count++
		if !more {
			assert.Equal(t, 1, count, "Learning error should be nil")
			break
		}
	}
}

func TestLoadDataFromCSVToStreamShouldFail1(t *testing.T) {
	err := SaveDataToCSV("/tmp/.goml/CSV_stream_fail.csv", x, y, true)
	assert.Nil(t, err, "Error saving data should be nil")

	data := make(chan Datapoint, 50)
	errors := make(chan error)

	go LoadDataFromCSVToStream("/tmp/.goml/PATH_THAT_DOES_NOT_EXIT_AT_ALL_OR_EVER_HOPEFULLYARKNGALRKGNALFJGNA.csv", data, errors)

	count := 0
	for {
		_, more := <-errors
		count++
		if !more {
			assert.NotEqual(t, 1, count, "Learning error should not be nil")
			break
		}
	}

	i := 0
	for {
		_, more := <-data

		if !more {
			assert.Equal(t, 0, i, "Stream should pass 0 examples")
			break
		}
		i++
	}
}
