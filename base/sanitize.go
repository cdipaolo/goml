package base

// OnlyWordsAndNumbers is a transform
// function that will only let 0-1a-zA-Z,
// and spaces though
func OnlyWordsAndNumbers(r rune) bool {
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

// OnlyWords is a transform function
// that will only let a-zA-Z, and
// spaces though
func OnlyWords(r rune) bool {
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

// OnlyLetters is a transform function
// that will only let a-zA-Z through
func OnlyLetters(r rune) bool {
	switch {
	case r >= 'A' && r <= 'Z':
		return false
	case r >= 'a' && r <= 'z':
		return false
	default:
		return true
	}
}
