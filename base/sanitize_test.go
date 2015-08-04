package base

import (
	"testing"

	"golang.org/x/text/transform"

	"github.com/stretchr/testify/assert"
)

func TestWordsAndNumbersShouldPass1(t *testing.T) {
	s, _, _ := transform.String(transform.RemoveFunc(OnlyWordsAndNumbers), "THIS iz A L337 aNd Un'Sani~~~~tized sentence")
	sanitized := []rune(s)

	for i := range sanitized {
		assert.False(t, OnlyWordsAndNumbers(sanitized[i]), "Letter %v should be sanitized", sanitized[i])
	}
}

func TestWordsAndNumbersShouldPass2(t *testing.T) {
	s, _, _ := transform.String(transform.RemoveFunc(OnlyWordsAndNumbers), ")(*&^%$@!@#$%^&*(*&^%$#@#$%")
	sanitized := []rune(s)

	assert.Equal(t, 0, len(sanitized), "Length of string should be 0")
}

func TestWordsShouldPass1(t *testing.T) {
	s, _, _ := transform.String(transform.RemoveFunc(OnlyWords), "THIS iz A L337 aNd Un'Sani~~~~tized sentence")
	sanitized := []rune(s)

	for i := range sanitized {
		assert.False(t, OnlyWords(sanitized[i]), "Letter %v should be sanitized", sanitized[i])
	}
}

func TestWordsShouldPass2(t *testing.T) {
	s, _, _ := transform.String(transform.RemoveFunc(OnlyWords), "08765432123456789)(*&^%$@!@#$%^&*(*&^%$#@#$%")
	sanitized := []rune(s)

	assert.Equal(t, 0, len(sanitized), "Length of string should be 0")
}

func TestLettersShouldPass1(t *testing.T) {
	s, _, _ := transform.String(transform.RemoveFunc(OnlyLetters), "THIS iz A L337 aNd Un'Sani~~~~tized sentence")
	sanitized := []rune(s)

	for i := range sanitized {
		assert.False(t, OnlyLetters(sanitized[i]), "Letter %v should be sanitized", sanitized[i])
	}
}

func TestLettersShouldPass2(t *testing.T) {
	s, _, _ := transform.String(transform.RemoveFunc(OnlyLetters), "0876543212     3456789)(*&^    %$@!@#$%^&    *(*&^%$#@#$%")
	sanitized := []rune(s)

	assert.Equal(t, 0, len(sanitized), "Length of string should be 0")
}
