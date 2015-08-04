package base

import (
	"golang.org/x/text/transform"
)

// OnlyWordsAndNumbers is a transform
// function that will only let 0-1a-zA-Z,
// and spaces though
var OnlyWordsAndNumbers = transform.RemoveFunc(func(r rune) bool {
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
})

// OnlyWords is a transform function
// that will only let a-zA-Z, and
// spaces though
var OnlyWords = transform.RemoveFunc(func(r rune) bool {
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
})

// OnlyLetters is a transform function
// that will only let a-zA-Z through
var OnlyLetters = transform.RemoveFunc(func(r rune) bool {
	switch {
	case r >= 'A' && r <= 'Z':
		return false
	case r >= 'a' && r <= 'z':
		return false
	default:
		return true
	}
})
