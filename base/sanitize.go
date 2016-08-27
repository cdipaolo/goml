package base

import (
	"unicode"
)

// OnlyAsciiWordsAndNumbers is a transform
// function that will only let 0-9a-zA-Z,
// and spaces through
func OnlyAsciiWordsAndNumbers(r rune) bool {
	switch {
	case r >= 'A' && r <= 'Z':
		return false
	case r >= 'a' && r <= 'z':
		return false
	case r >= '0' && r <= '9':
		return false
	case r == ' ':
		return false
	default:
		return true
	}
}

// OnlyWordsAndNumbers is a transform
// function that lets any unicode letter
// or digit through as well as spaces
func OnlyWordsAndNumbers(r rune) bool {
	return !(r == ' ' || unicode.IsLetter(r) || unicode.IsDigit(r))
}

// OnlyAsciiWords is a transform function
// that will only let a-zA-Z, and
// spaces through
func OnlyAsciiWords(r rune) bool {
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

// OnlyWords is a transform function
// that lets any unicode letter through
// as well as spaces
func OnlyWords(r rune) bool {
	return !(r == ' ' || unicode.IsLetter(r))
}

// OnlyAsciiLetters is a transform function
// that will only let a-zA-Z through
func OnlyAsciiLetters(r rune) bool {
	switch {
	case r >= 'A' && r <= 'Z':
		return false
	case r >= 'a' && r <= 'z':
		return false
	default:
		return true
	}
}

// OnlyLetters is a transform function
// that lets any unicode letter through
func OnlyLetters(r rune) bool {
	return !unicode.IsLetter(r)
}
