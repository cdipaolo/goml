package text

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"strings"

	"golang.org/x/text/transform"

	"github.com/cdipaolo/goml/base"
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
termFrequency(word, doc) = 0.5 * ( 0.5 * word.Count/doc.CountWords ) / max{ w.Count/doc.CountWords | w ∈ doc }

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

// TFIDF returns the TermFrequency-
// InverseDocumentFrequency of a word
// within a corpus given by the trained
// NaiveBayes model
//
// Look at the TFIDF docs to see more about how
// this is calculated
func (t *TFIDF) TFIDF(word string, document []string) float64 {
	return t.TermFrequency(word, document) * t.InverseDocumentFrequency(word, document)
}

// TermFrequency returns the term frequency
// of a word within a corpus defined by the
// trained NaiveBayes model
//
// Look at the TFIDF docs to see more about how
// this is calculated
func (t *TFIDF) TermFrequency(word string, document []string) float64 {

}

// InverseDocumentFrequency returns the 'uniqueness'
// of a word within the corpus defined within a
// trained NaiveBayes model.
//
// Look at the TFIDF docs to see more about how
// this is calculated
func (t *TFIDF) InverseDocumentFrequency(word string, document []string) float64 {

}
