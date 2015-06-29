package base

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

var x [][]float64
var y []float64

func init() {
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
