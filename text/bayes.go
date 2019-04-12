/*
Package text holds models which
make text classification easy. They
are regular models, but take strings
as arguments so you can feed in
documents rather than large,
hand-constructed word vectors. Although
models might represent the words as
these vectors, the munging of a
document is hidden from the user.

The simplest model, although suprisingly
effective, is Naive Bayes. If you
want to read more about the specific
model, check out the docs for the
NaiveBayes struct/model.

The following example is an online Naive
Bayes model used for sentiment analysis.

Example Online Naive Bayes Text Classifier (multiclass):

	// create the channel of data and errors
	stream := make(chan base.TextDatapoint, 100)
	errors := make(chan error)

	// make a new NaiveBayes model with
	// 2 classes expected (classes in
	// datapoints will now expect {0,1}.
	// in general, given n as the classes
	// variable, the model will expect
	// datapoint classes in {0,...,n-1})
	//
	// Note that the model is filtering
	// the text to omit anything except
	// words and numbers (and spaces
	// obviously)
	model := NewNaiveBayes(stream, 2, base.OnlyWordsAndNumbers)

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
		err, more := <- errors
		if err != nil {
			fmt.Fprintf(b.Output, "Error passed: %v", err)
		} else {
			// training is done!
			break
		}
	}

	// now you can predict like normal
	class := model.Predict("My mother is in Los Angeles") // 0
*/
package text

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"os"
	"strings"
	"sync"

	"golang.org/x/text/transform"

	"github.com/cdipaolo/goml/base"
)

/*
NaiveBayes is a general classification
model that calculates the probability
that a datapoint is part of a class
by using Bayes Rule:
	P(y|x) = P(x|y)*P(y)/P(x)
The unique part of this model is that
it assumes words are unrelated to
eachother. For example, the probability
of seeing the word 'penis' in spam emails
if you've already seen 'viagra' might be
different than if you hadn't seen it. The
model ignores this fact because the
computation of full Bayesian model would
take much longer, and would grow significantly
with each word you see.

https://en.wikipedia.org/wiki/Naive_Bayes_classifier
http://cs229.stanford.edu/notes/cs229-notes2.pdf

Based on Bayes Rule, we can easily calculate
the numerator (x | y is just the number of
times x is seen and the class=y, and P(y) is
just the number of times y=class / the number
of positive training examples/words.) The
denominator is also easy to calculate, but
if you recognize that it's just a constant
because it's just the probability of seeing
a certain document given the dataset we can
make the following transformation to be
able to classify without as much classification:
	Class(x) = argmax_c{P(y = c) * ∏P(x|y = c)}
And we can use logarithmic transformations to
make this calculation more computer-practical
(multiplying a bunch of probabilities on [0,1]
will always result in a very small number
which could easily underflow the float value):
	Class(x) = argmax_c{log(P(y = c)) + Σ log(P(x|y = c)0}
Much better. That's our model!
*/
type NaiveBayes struct {
	// Words holds a map of words
	// to their corresponding Word
	// structure
	Words concurrentMap `json:"words"`

	// Count holds the number of times
	// class i was seen as Count[i]
	Count []uint64 `json:"count"`

	// Probabilities holds the probability
	// that class Y is class i as
	// Probabilities[i] for
	Probabilities []float64 `json:"probabilities"`

	// DocumentCount holds the number of
	// documents that have been seen
	DocumentCount uint64 `json:"document_count"`

	// DictCount holds the size of the
	// NaiveBayes model's vocabulary
	DictCount uint64 `json:"vocabulary_size"`

	// sanitize is used by a model
	// to sanitize input of text
	sanitize transform.Transformer

	// stream holds the datastream
	stream <-chan base.TextDatapoint

	// tokenizer is used by a model
	// to split the input into tokens
	Tokenizer Tokenizer `json:"tokenizer"`

	// Output is the io.Writer used for logging
	// and printing. Defaults to os.Stdout.
	Output io.Writer `json:"-"`
}

// Tokenizer accepts a sentence as input and breaks
// it down into a slice of tokens
type Tokenizer interface {
	Tokenize(string) []string
}

// SimpleTokenizer splits sentences
// into tokens delimited by its
// SplitOn string – space, for example
type SimpleTokenizer struct {
	SplitOn string
}

// Tokenize splits input sentences into a lowecase slice
// of strings. The tokenizer's SlitOn string is used as a
// delimiter and it
func (t *SimpleTokenizer) Tokenize(sentence string) []string {
	// is the tokenizer really the best place to be making
	// everything lowercase? is this really a sanitizaion func?
	return strings.Split(strings.ToLower(sentence), t.SplitOn)
}

// concurrentMap allows concurrency-friendly map
// access via its exported Get and Set methods
type concurrentMap struct {
	sync.RWMutex
	words map[string]Word
}

func (m *concurrentMap) MarshalJSON() ([]byte, error) {
	return json.Marshal(m.words)
}

func (m *concurrentMap) UnmarshalJSON(data []byte) error {
	err := json.Unmarshal(data, &m.words)
	if err != nil {
		return err
	}

	return nil
}

// Get looks up a word from h's Word map and should be used
// in place of a direct map lookup. The only caveat is that
// it will always return the 'success' boolean
func (m *concurrentMap) Get(w string) (Word, bool) {
	m.RLock()
	result, ok := m.words[w]
	m.RUnlock()
	return result, ok
}

// Set sets word k's value to v in h's Word map
func (m *concurrentMap) Set(k string, v Word) {
	m.Lock()
	m.words[k] = v
	m.Unlock()
}

// Word holds the structural
// information needed to calculate
// the probability of
type Word struct {
	// Count holds the number of times,
	// (i in Count[i] is the given class)
	Count []uint64

	// Seen holds the number of times
	// the world has been seen. This
	// is than same as
	//    foldl (+) 0 Count
	// in Haskell syntax, but is included
	// you wouldn't have to calculate
	// this every time you wanted to
	// recalc the probabilities (foldl
	// is the same as reduce, basically.)
	Seen uint64

	// DocsSeen is the same as Seen but
	// a word is only counted once even
	// if it's in a document multiple times
	DocsSeen uint64 `json:"-"`
}

// NewNaiveBayes returns a NaiveBayes model the
// given number of classes instantiated, ready
// to learn off the given data stream. The sanitization
// function is set to the given function. It must
// comply with the transform.RemoveFunc interface
func NewNaiveBayes(stream <-chan base.TextDatapoint, classes uint8, sanitize func(rune) bool) *NaiveBayes {
	return &NaiveBayes{
		Words:         concurrentMap{sync.RWMutex{}, make(map[string]Word)},
		Count:         make([]uint64, classes),
		Probabilities: make([]float64, classes),

		sanitize:  transform.RemoveFunc(sanitize),
		stream:    stream,
		Tokenizer: &SimpleTokenizer{SplitOn: " "},

		Output: os.Stdout,
	}
}

// Predict takes in a document, predicts the
// class of the document based on the training
// data passed so far, and returns the class
// estimated for the document.
func (b *NaiveBayes) Predict(sentence string) uint8 {
	sums := make([]float64, len(b.Count))

	sentence, _, _ = transform.String(b.sanitize, sentence)
	words := b.Tokenizer.Tokenize(sentence)
	for _, word := range words {
		w, ok := b.Words.Get(word)
		if !ok {
			continue
		}

		for i := range sums {
			sums[i] += math.Log(float64(w.Count[i]+1) / float64(w.Seen+b.DictCount))
		}
	}

	for i := range sums {
		sums[i] += math.Log(b.Probabilities[i])
	}

	// find best class
	var maxI int
	for i := range sums {
		if sums[i] > sums[maxI] {
			maxI = i
		}
	}

	return uint8(maxI)
}

// Probability takes in a small document, returns the
// estimated class of the document based on the model
// as well as the probability that the model is part
// of that class
//
// NOTE: you should only use this for small documents
// because, as discussed in the docs for the model, the
// probability will often times underflow because you
// are multiplying together a bunch of probabilities
// which range on [0,1]. As such, the returned float
// could be NaN, and the predicted class could be
// 0 always.
//
// Basically, use Predict to be robust for larger
// documents. Use Probability only on relatively small
// (MAX of maybe a dozen words - basically just
// sentences and words) documents.
func (b *NaiveBayes) Probability(sentence string) (uint8, float64) {
	sums := make([]float64, len(b.Count))
	for i := range sums {
		sums[i] = 1
	}

	sentence, _, _ = transform.String(b.sanitize, sentence)
	words := b.Tokenizer.Tokenize(sentence)
	for _, word := range words {
		w, ok := b.Words.Get(word)
		if !ok {
			continue
		}

		for i := range sums {
			sums[i] *= float64(w.Count[i]+1) / float64(w.Seen+b.DictCount)
		}
	}

	for i := range sums {
		sums[i] *= b.Probabilities[i]
	}

	var denom float64
	var maxI int
	for i := range sums {
		if sums[i] > sums[maxI] {
			maxI = i
		}

		denom += sums[i]
	}

	return uint8(maxI), sums[maxI] / denom
}

// OnlineLearn lets the NaiveBayes model learn
// from the datastream, waiting for new data to
// come into the stream from a separate goroutine
func (b *NaiveBayes) OnlineLearn(errors chan<- error) {
	if errors == nil {
		errors = make(chan error)
	}
	if b.stream == nil {
		errors <- fmt.Errorf("ERROR: attempting to learn with nil data stream!\n")
		close(errors)
		return
	}

	fmt.Fprintf(b.Output, "Training:\n\tModel: Multinomial Naïve Bayes\n\tClasses: %v\n", len(b.Count))

	var point base.TextDatapoint
	var more bool

	for {
		point, more = <-b.stream

		if more {
			// sanitize and break up document
			sanitized, _, _ := transform.String(b.sanitize, point.X)
			words := b.Tokenizer.Tokenize(sanitized)

			C := int(point.Y)

			if C > len(b.Count)-1 {
				errors <- fmt.Errorf("ERROR: given document class is greater than the number of classes in the model!\n")
				continue
			}

			// update global class probabilities
			b.Count[C]++
			b.DocumentCount++
			for i := range b.Probabilities {
				b.Probabilities[i] = float64(b.Count[i]) / float64(b.DocumentCount)
			}

			// store words seen in document (to add to DocsSeen)
			seenCount := make(map[string]int)

			// update probabilities for words
			for _, word := range words {
				if len(word) < 3 {
					continue
				}

				w, ok := b.Words.Get(word)

				if !ok {
					w = Word{
						Count: make([]uint64, len(b.Count)),
						Seen:  uint64(0),
					}

					b.DictCount++
				}

				w.Count[C]++
				w.Seen++

				b.Words.Set(word, w)

				seenCount[word] = 1
			}

			// add to DocsSeen
			for term := range seenCount {
				tmp, _ := b.Words.Get(term)
				tmp.DocsSeen++
				b.Words.Set(term, tmp)
			}
		} else {
			fmt.Fprintf(b.Output, "Training Completed.\n%v\n\n", b)
			close(errors)
			return
		}
	}
}

// UpdateStream updates the NaiveBayes model's
// text datastream
func (b *NaiveBayes) UpdateStream(stream chan base.TextDatapoint) {
	b.stream = stream
}

// UpdateSanitize updates the NaiveBayes model's
// text sanitization transformation function
func (b *NaiveBayes) UpdateSanitize(sanitize func(rune) bool) {
	b.sanitize = transform.RemoveFunc(sanitize)
}

// UpdateTokenizer updates NaiveBayes model's tokenizer function.
// The default implementation will convert the input to lower
// case and split on the space character.
func (b *NaiveBayes) UpdateTokenizer(tokenizer Tokenizer) {
	b.Tokenizer = tokenizer
}

// String implements the fmt interface for clean printing. Here
// we're using it to print the model as the equation h(θ)=...
// where h is the perceptron hypothesis model.
func (b *NaiveBayes) String() string {
	return fmt.Sprintf("h(θ) = argmax_c{log(P(y = c)) + Σlog(P(x|y = c))}\n\tClasses: %v\n\tDocuments evaluated in model: %v\n\tWords evaluated in model: %v\n", len(b.Count), int(b.DocumentCount), int(b.DictCount))
}

// PersistToFile takes in an absolute filepath and saves the
// parameter vector θ to the file, which can be restored later.
// The function will take paths from the current directory, but
// functions
//
// The data is stored as JSON because it's one of the most
// efficient storage method (you only need one comma extra
// per feature + two brackets, total!) And it's extendable.
func (b *NaiveBayes) PersistToFile(path string) error {
	if path == "" {
		return fmt.Errorf("ERROR: you just tried to persist your model to a file with no path!! That's a no-no. Try it with a valid filepath")
	}

	bytes, err := json.Marshal(b)
	if err != nil {
		return err
	}

	err = ioutil.WriteFile(path, bytes, os.ModePerm)
	if err != nil {
		return err
	}

	return nil
}

// Restore takes the bytes of a NaiveBayes model and
// restores a model to it. It defaults the sanitizer
// to base.OnlyWordsAndNumbers and the tokenizer to
// to a SimpleTokenizer that splits on spaces.
//
// This would be useful if training a model and saving
// it into a project using go-bindata (look it up) so
// you don't have to persist a large file and deal with
// paths on a production system. This option is included
// in text models vs. others because the text models
// usually have much larger storage requirements.
func (b *NaiveBayes) Restore(data []byte) error {
	return b.RestoreWithFuncs(bytes.NewReader(data), base.OnlyWordsAndNumbers, &SimpleTokenizer{SplitOn: " "})
}

// RestoreWithFuncs takes raw JSON data of a model and
// restores a model from it. The tokenizer and sanitizer
// passed in will be assigned to the restored model.
func (b *NaiveBayes) RestoreWithFuncs(data io.Reader, sanitizer func(rune) bool, tokenizer Tokenizer) error {
	if b == nil {
		return errors.New("Cannot restore a model to a nil pointer")
	}
	err := json.NewDecoder(data).Decode(b)
	if err != nil {
		return err
	}
	b.sanitize = transform.RemoveFunc(sanitizer)
	b.Tokenizer = tokenizer
	return nil
}

// RestoreFromFile takes in a path to a parameter vector theta
// and assigns the model it's operating on's parameter vector
// to that. The only parameters not in the vector are the sanitization
// and tokenization functions which default to base.OnlyWordsAndNumbers
// and SimpleTokenizer{SplitOn: " "}
//
// The path must ba an absolute path or a path from the current
// directory
//
// This would be useful in persisting data between running
// a model on data.
func (b *NaiveBayes) RestoreFromFile(path string) error {
	if path == "" {
		return fmt.Errorf("ERROR: you just tried to restore your model from a file with no path! That's a no-no. Try it with a valid filepath")
	}

	bytes, err := ioutil.ReadFile(path)
	if err != nil {
		return err
	}
	err = b.Restore(bytes)
	if err != nil {
		return err
	}

	return nil
}
