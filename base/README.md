## Base Package
### `import "github.com/cdipaolo/goml/base"`

[![GoDoc](https://godoc.org/github.com/cdipaolo/goml/base?status.svg)](https://godoc.org/github.com/cdipaolo/goml/base)

This package helps define common patterns (interfaces,) as well as letting you work with data, get it into your programs, and munge through it.

This package also implements optimization algorithms which can be made available to a user's own models by implementing easy to use interfaces.

### functions for working with data

- [func LoadDataFromCSV(filepath string) ([][]float64, []float64, error)](data.go)
  * takes a training set (in the format specified on the function's comments/documentation) and returns a 2D slice of float64's of the input features, as well as a 1D slice of the results of those inputs.
- [func SaveDataToCSV(filepath string, x [][]float64, y []float64, highPrecision bool) error](data.go)
  * takes datasets you might have within the memory and save them to disk. Could be useful if you edit data within a program and want to save a new version of that somewhere.