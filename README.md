## `goml`
### Golang Machine Learning, On The Wire

[![GoDoc](https://godoc.org/github.com/cdipaolo/goml?status.svg)](https://godoc.org/github.com/cdipaolo/goml)
[![wercker status](https://app.wercker.com/status/50a8cfa6170784809e3308941212cef4/s "wercker status")](https://app.wercker.com/project/bykey/50a8cfa6170784809e3308941212cef4)

`goml` is a machine learning library written entirely in Golang which lets the average developer include machine learning into their applications. (pronounced like the data format 'toml')

While models include traditional, batch learning interfaces, `goml` includes many models which let you learn in an online, reactive manner by passing data to streams held on channels.

The library includes **comprehensive tests**, **extensive documentation**, and **clean, expressive, modular source code**. Community contribution is heavily encouraged.

Each package (mentioned below) includes individual README's to learn more about the function, and purpose of the models. Above all, if you want to learn about models, read the GoDoc reference for the package. All models are, as mentioned above, heavily documented.

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

- [Generalized Linear Models](linear/) (all have stochastic GA, batch GA, and online options except for locally weighted linear regression)
  * [Ordinary Least Squares](linear/linear.go)
  * [Locally Weighted Linear Regression](linear/local_linear.go)
  * [Logistic Regression](linear/logistic.go)
  * [Softmax (Multiclass Logistic) Regression](linear/softmax.go)
- [Perceptron](perceptron/) only in online options
  * [Online, Binary Perceptron](perceptron/perceptron.go)
  * [Online, Binary Kernel Perceptron](perceptron/kernel_perceptron.go)
- [Clustering](cluster/)
  * [K-Means Clustering](cluster/kmeans.go)
    * Uses k-means++ instantiation for more reliable clusters ([this paper](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf) discusses the method and it's benefits over regular, random instantiation)
  	* Both online and batch versions
    * Includes a version which uses the [Triangle Inequality](https://en.wikipedia.org/wiki/Triangle_inequality) to dramatically reduce the number of distance calculations at the expense of auxillary data structures, as describes in [this paper](http://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf)
  * [K-Nearest-Neighbors Clustering](cluster/knn.go)
  	* Can use any distance metric, with L-p Norm, Euclidean Distance, and Manhattan Distance pre-defined within the `goml/base` package
- [Text Classification](text/)
  * [Multinomial (Multiclass) Text-Based Naive Bayes](text/bayes.go)
  * [Term Frequency - Inverse Document Frequency](text/tfidf.go)
    * this lets you find keywords/important words from documents
    * because it's so similar to Bayes under the hood, you cast a NaiveBayes model to TFIDF to get a model. [Look at these tests to see an example](text/tfidf_test.go)

## Contributing!

see [CONTRIBUTING](CONTRIBUTING.md).

I'd love help with any of this if anybody thinks that they would like to implement a model that isn't here, or if they have improvements to current models implemented, or if they want to help with documentation (this would be greatly appreciated, believe me, writing great documentation takes time! :+1:)

## LICENSE - MIT

see [LICENSE](LICENSE)
