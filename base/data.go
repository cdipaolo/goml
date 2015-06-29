package base

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
)

// LoadDataFromCSV takes in a path to a CSV file and
// loads that data into a Golang 2D array of 'X' values
// and a Golang 1D array of 'Y', or expected result,
// values.
//
// Errors are returned if there are any problems
//
// Expected Data Format:
// - There should be no header/text lines.
// - The 'Y' (expected value) line should be the last
//     column of the CSV.
//
// Example CSV file with 2 input parameters:
//     >>>>>>> BEGIN FILE
//     1.06,2.30,17
//     17.62,12.06,18.92
//     11.623,1.1,15.093
//     12.01,6,15.032
//     ...
//     >>>>>>> END FILE
func LoadDataFromCSV(filepath string) ([][]float64, []float64, error) {
	_, err := os.Stat(filepath)
	if err != nil {
		return nil, nil, err
	}

	file, err := os.Open(filepath)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	x := [][]float64{}
	y := []float64{}

	record, err := reader.Read()
	if err != nil {
		return nil, nil, err
	}

	fmt.Printf("Loading Data From CSV <%v>\n", filepath)
	// parse until the end of the file
	for err != io.EOF {
		var row []float64

		for i, val := range record {
			float, err := strconv.ParseFloat(val, 64)
			if err != nil {
				return nil, nil, err
			}

			if i < len(record)-1 {
				row = append(row, float)
			} else {
				y = append(y, float)
			}
		}

		x = append(x, row)

		record, err = reader.Read()
	}

	if len(x) == 0 || len(x[0]) == 0 || len(y) == 0 {
		return nil, nil, fmt.Errorf("ERROR: Training set has no valid examples (either for x or y or both)")
	}

	fmt.Printf("Finished Loading Data From <%v>\n\tTraining Examples: %v\n\tFeatures: %v\n", filepath, len(y), len(x[0]))
	return x, y, nil
}

// SaveDataToCSV takes in a absolute filepath, as well
// as a 2D array of 'X' values and a 1D array of 'Y',
// or expected values, concatenates the format to the
// same as LoadDataFromCSV, and saves that data to a
// file, returning any errors.
//
// highPrecision is a boolean where if true the values
// will be stored with a 64 bit precision when converting
// the floats to strings. Otherwise (if it's false) it
// uses 32 bits.
func SaveDataToCSV(filepath string, x [][]float64, y []float64, highPrecision bool) error {
	if len(x) == 0 || len(x[0]) == 0 || len(y) == 0 || len(y) != len(x) {
		return fmt.Errorf("ERROR: Training set (either x or y or both) has no examples or the lengths of the dataset don't match")
	}

	_, err := os.Stat(filepath)
	if err != nil && !os.IsNotExist(err) {
		return err
	}

	file, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer file.Close()

	var precision int
	if highPrecision {
		precision = 64
	} else {
		precision = 32
	}

	writer := csv.NewWriter(file)
	records := [][]string{}

	fmt.Printf("Writing Data To <%v>\n\tTraining Examples: %v\n\tFeatures: %v\n", filepath, len(x), len(x[0]))
	// parse until the end of the file
	for i := range x {
		record := []string{}

		for j := range x[i] {
			record = append(record, strconv.FormatFloat(x[i][j], 'g', -1, precision))
		}

		record = append(record, strconv.FormatFloat(y[i], 'g', -1, precision))

		records = append(records, record)
	}

	// now save the record to file
	err = writer.WriteAll(records)
	if err != nil {
		return err
	}

	fmt.Printf("Finished Writing Data To <%v>\n\n", filepath)
	return nil
}
