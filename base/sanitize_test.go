package base

import (
	"testing"

	"golang.org/x/text/transform"

	"github.com/stretchr/testify/assert"
)

var wordsAndNumbers = func(r rune) bool {
	switch {
	case r >= 'A' && r <= 'Z':
		return false
	case r >= 'a' && r <= 'z':
		return false
	case r >= '0' && r <= '1':
		return false
	case r == ' ':
		return false
	default:
		return true
	}
}

var words = func(r rune) bool {
	switch {
	case r >= 'A' && r <= 'Z':
		return false
	case r >= 'a' && r <= 'z':
		return false
	case r == ' ':
		return false
	default:
		return true
	}
}

var letters = func(r rune) bool {
	switch {
	case r >= 'A' && r <= 'Z':
		return false
	case r >= 'a' && r <= 'z':
		return false
	default:
		return true
	}
}

func TestWordsAndNumbersShouldPass1(t *testing.T) {
	s, _, _ := transform.String(OnlyWordsAndNumbers, "THIS iz A L337 aNd Un'Sani~~~~tized sentence")
	sanitized := []rune(s)

	for i := range sanitized {
		assert.False(t, wordsAndNumbers(sanitized[i]), "Letter %v should be sanitized", sanitized[i])
	}
}

func TestWordsAndNumbersShouldPass2(t *testing.T) {
	s, _, _ := transform.String(OnlyWordsAndNumbers, ")(*&^%$@!@#$%^&*(*&^%$#@#$%")
	sanitized := []rune(s)

	assert.Equal(t, 0, len(sanitized), "Length of string should be 0")
}

func TestWordsShouldPass1(t *testing.T) {
	s, _, _ := transform.String(OnlyWords, "THIS iz A L337 aNd Un'Sani~~~~tized sentence")
	sanitized := []rune(s)

	for i := range sanitized {
		assert.False(t, words(sanitized[i]), "Letter %v should be sanitized", sanitized[i])
	}
}

func TestWordsShouldPass2(t *testing.T) {
	s, _, _ := transform.String(OnlyWords, "08765432123456789)(*&^%$@!@#$%^&*(*&^%$#@#$%")
	sanitized := []rune(s)

	assert.Equal(t, 0, len(sanitized), "Length of string should be 0")
}

func TestLettersShouldPass1(t *testing.T) {
	s, _, _ := transform.String(OnlyLetters, "THIS iz A L337 aNd Un'Sani~~~~tized sentence")
	sanitized := []rune(s)

	for i := range sanitized {
		assert.False(t, letters(sanitized[i]), "Letter %v should be sanitized", sanitized[i])
	}
}

func TestLettersShouldPass2(t *testing.T) {
	s, _, _ := transform.String(OnlyLetters, "0876543212     3456789)(*&^    %$@!@#$%^&    *(*&^%$#@#$%")
	sanitized := []rune(s)

	assert.Equal(t, 0, len(sanitized), "Length of string should be 0")
}
