package text

import (
	"math"
	"sort"

	"golang.org/x/text/transform"
)

/*
TFIDF is a Term Frequency- Inverse
Document Frequency model that is created
from a trained NaiveBayes model (they
are very similar so you can just train
NaiveBayes and convert into TDIDF)

This is not a probabalistic model, necessarily,
and doesn't give classification. It can be
used to determine the 'importance' of a
word in a document, though, which is
useful in, say, keyword tagging.

Term frequency is basically just adjusted
frequency of a word within a document/sentence:
termFrequency(word, doc) = 0.5 * ( 0.5 * word.Count ) / max{ w.Count | w ∈ doc }

Inverse document frequency is basically how
little the term is mentioned within all of
your documents:
invDocumentFrequency(word, Docs) = log( len(Docs) ) - log( 1 + |{ d ∈ Docs | t ∈ d}| )

TFIDF is the multiplication of those two
functions, giving you a term that is larger
when the word is more important, and less when
the word is less important
*/
type TFIDF NaiveBayes

// Frequency holds word frequency information
// so you don't have to hold a map[string]float64
// and can, then, sort
type Frequency struct {
	Word      string  `json:"word"`
	Frequency float64 `json:"frequency,omitempty"`
	TFIDF     float64 `json:"tfidf_score,omitempty"`
}

// Frequencies is an array of word frequencies
// (stored as separate type to be able to sort)
type Frequencies []Frequency

//* implement sort.Interface for Frequency list *//

// Len gives the length of a
// frequency array
func (f Frequencies) Len() int {
	return len(f)
}

// Less gives whether the ith element
// of a frequency list has is lesser
// than the jth element by comparing
// their TFIDF values
func (f Frequencies) Less(i, j int) bool {
	return f[i].TFIDF < f[j].TFIDF
}

// Swap swaps two indexed values in
// a frequency slice
func (f Frequencies) Swap(i, j int) {
	f[i], f[j] = f[j], f[i]
}

// TFIDF returns the TermFrequency-
// InverseDocumentFrequency of a word
// within a corpus given by the trained
// NaiveBayes model
//
// Look at the TFIDF docs to see more about how
// this is calculated
func (t *TFIDF) TFIDF(word string, sentence string) float64 {
	sentence, _, _ = transform.String(t.sanitize, sentence)
	document := t.Tokenizer.Tokenize(sentence)

	return t.TermFrequency(word, document) * t.InverseDocumentFrequency(word)
}

// MostImportantWords runs TFIDF on a
// whole document, returning the n most
// important words in the document. If
// n is greater than the number of words
// then all words will be returned.
//
// The returned keyword slice is sorted
// by importance
func (t *TFIDF) MostImportantWords(sentence string, n int) Frequencies {
	sentence, _, _ = transform.String(t.sanitize, sentence)
	document := t.Tokenizer.Tokenize(sentence)

	freq := TermFrequencies(document)
	for i := range freq {
		freq[i].TFIDF = freq[i].Frequency * t.InverseDocumentFrequency(freq[i].Word)
		freq[i].Frequency = float64(0.0)
	}

	// sort into slice
	sort.Sort(sort.Reverse(freq))

	if n > len(freq) {
		return freq
	}

	return freq[:n]
}

// TermFrequency returns the term frequency
// of a word within a corpus defined by the
// trained NaiveBayes model
//
// Look at the TFIDF docs to see more about how
// this is calculated
func (t *TFIDF) TermFrequency(word string, document []string) float64 {
	words := make(map[string]int)
	for i := range document {
		words[document[i]]++
	}

	// find max word frequency
	var maxFreq int
	for i := range words {
		if words[i] > maxFreq {
			maxFreq = words[i]
		}
	}

	return 0.5 * (1 + float64(words[word])/float64(maxFreq))
}

// TermFrequencies gives the TermFrequency of
// all words in a document, and is more efficient
// at doing so than calling that function multiple
// times
func TermFrequencies(document []string) Frequencies {
	words := make(map[string]int)
	for i := range document {
		words[document[i]]++
	}

	var maxFreq int
	for i := range words {
		if words[i] > maxFreq {
			maxFreq = words[i]
		}
	}

	// make frequency map
	frequencies := Frequencies{}
	for i := range words {
		frequencies = append(frequencies, Frequency{
			Word:      i,
			Frequency: 0.5 * (1 + float64(words[i])/float64(maxFreq)),
		})
	}

	return frequencies
}

// InverseDocumentFrequency returns the 'uniqueness'
// of a word within the corpus defined within a
// trained NaiveBayes model.
//
// Look at the TFIDF docs to see more about how
// this is calculated
func (t *TFIDF) InverseDocumentFrequency(word string) float64 {
	w, _ := t.Words.Get(word)
	return math.Log(float64(t.DocumentCount)) - math.Log(float64(w.DocsSeen)+1)
}
