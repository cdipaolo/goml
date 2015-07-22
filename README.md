## `goml`
### Golang Machine Learning, On The Wire

[![GoDoc](https://godoc.org/github.com/cdipaolo/goml?status.svg)](https://godoc.org/github.com/cdipaolo/goml)
[![wercker status](https://app.wercker.com/status/50a8cfa6170784809e3308941212cef4/s "wercker status")](https://app.wercker.com/project/bykey/50a8cfa6170784809e3308941212cef4)

goml (pronounced like the data format 'toml') is a batteries included machine learning library written entirely in Golang. It lets you create models of data stored as float64's, persist them to disk, and predict other values from them. **The coolest part, among many cool parts, is that you can train most models in an on-line fashion, learning in a 'reactive' manner while waiting for further data on channels!** Most models can also be trained in batch settings, using either stochastic _or_ batch gradient descent!

Each of the packages have individual README's where you can learn about how to use each of the models. Even better than the short summaries in the README is the **extensive documentation with examples and descriptions** in GoDoc (look at the button above.) You could also look at our **comprehensive tests** to see error handline and other details, as well as look at the **clean, expressive, and modular source code**.

## Installation

```bash
go get github.com/cdipaolo/goml/base

# This could be any other model package if you want
#
# Also, the base package is imported already
# by many of the packages so you might not even
# need to `go get` the package explicitly
go get github.com/cdipaolo/goml/perceptron
```

## Documentation

All the code is well documented, and the source is/should be really readable if you'd like to make sense of it all! Look at each package (like right now, in GitHub,) and you will see a link to Godoc as well as an explanation of the package and an example usage. You can even click on the main bullets below and it'll take you to those packages. Also you could just use the Godoc link at the top of this README and navigate to the package you'd like to see more about.

Sub-bullets below will take you directly to the source code of the model.

## Currently Implemented Models

- [Generalized Linear Models](linear/) (all have stochastic GA, batch GA, and online options)
  * [Ordinary Least Squares](linear/linear.go)
  * [Locally Weighted Linear Regression](linear/local_linear.go)
  * [Logistic Regression](linear/logistic.go)
  * [Softmax (Multiclass Logistic) Regression](linear/softmax.go)
- [Perceptron](perceptron/) only in online options
  * [Online, Binary Perceptron](perceptron/perceptron.go)
  * [Online, Binary Kernel Perceptron](perceptron/kernel_perceptron.go)
- [Clustering](cluster/)
  * [K-Means Clustering](cluster/kmeans.go)
  	* Both online and batch versions

## Contributing!

see [CONTRIBUTING](CONTRIBUTING.md).

I'd love help with any of this if anybody thinks that they would like to implement a model that isn't here, or if they have improvements to current models implemented, or if they want to help with documentation (this would be greatly appreciated, believe me, writing great documentation takes time! :+1:)

## LICENSE - MIT

see [LICENSE](LICENSE)
