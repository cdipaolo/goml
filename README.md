## `goml`
### Golang Machine Learning

[![wercker status](https://app.wercker.com/status/50a8cfa6170784809e3308941212cef4/s "wercker status")](https://app.wercker.com/project/bykey/50a8cfa6170784809e3308941212cef4)

goml (pronounced like the data format 'toml') is a batteries included machine learning library written entirely in Golang. It lets you create models of data stored as float64's, persist them to disk, and predict other values from them.

You could use the `golearn` package @sjwhitworth wrote, but mine has kind error messages! :smile:. On a more serious note, I'm making this library to learn about machine learning. It works, don't misunderstand, and due to that the code is really well commented and structured for ease of extension in case you want to write a more individual model (in many of those cases I'd hope you made it into a pull request as well.)

Each of the packages have individual README's where you can learn about how to use each of the models.

## Installation

yes. It's this easy.

```bash
go get github.com/cdipaolo/goml
```

## Currently Implemented Models

- [Generalized Linear Models](linear/)
  * [Ordinary Least Squares](linear/linear.go)
  * [Logistic Regression](linear/logistic.go)
- [Perceptron](perceptron/)
  * [Online, Binary Perceptron](perceptron/perceptron.go)

## Contributing!

see [CONTRIBUTING](CONTRIBUTING.md).

I'd love help with any of this if anybody thinks that they would like to implement a model that isn't here, or if they have improvements to current models implemented, or if they want to help with documentation (this would be greatly appreciated, believe me, writing great documentation takes time! :+1:)

## LICENSE - MIT

see [LICENSE](LICENSE)
