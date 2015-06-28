## Generalized Linear Models
### `import "github.com/cdipaolo/goml/linear"`

This part of the `goml` package implements varied generalized linear models using gradient descent (currently, though more options for optimization methods might be available in the future.)

### example ordinary least squares

this is mostly from from the `linear_test.go` tests. You can find more examples from the testing files. The line given is `z = 10 + (x/10) + (y/5)`
```
// initialize data
threeDLineX = [][]float64{}
threeDLineY = []float64{}
// the line z = 10 + (x/10) + (y/5)
for i := -10; i < 10; i++ {
    for j := -10; j < 10; j++ {
        threeDLineX = append(threeDLineX, []float64{float64(i), float64(j)})
        threeDLineY = append(threeDLineY, 10+float64(i)/10+float64(j)/5)
    }
}

// initialize model
model, err := linear.NewLeastSquares(.0001, 1000, threeDLineX, threeDLineY)
if err != nil {
    panic("Your training set (either x or y) was nil/zero length")
}

// learn
err = model.Learn()
if err != nil {
    panic("There was some error learning")
}

// predict based on model
guess, err = model.Predict([]float64{12.016, 6.523})
if err != nil {
    panic("There was some error in the prediction")
}
```

### gradient descent optimization

Here's some data relating the cost function J(θ) and the number of iterations of the data using a 3d model. Note that in this case the data modeled off of was perfectly linear, so obviously the cost function wouldn't and shouldn't bottom out at 0.000... for real world data!

![Nice Looking Graph!](cost_function_vs_iterations.png "Ordinary Least Squares Cost Function vs. Iterations on Gradient Descent")