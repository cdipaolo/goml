package base

import (
	"testing"

	"golang.org/x/text/transform"
)

type testCase struct {
	input          string
	expectedOutput string
}

func TestWordsAndNumbers(t *testing.T) {
	tests := []testCase{
		{"THIS iz A L337 aNd Un'Sani~~~~tized sentence", "THIS iz A L337 aNd UnSanitized sentence"},
		{"here're some unicode letters: --Æ.ÒÑ", "herere some unicode letters ÆÒÑ"},
		{")(*&^%$@!@#$%^&*(*&^%$#@#$%", ""},
	}
	for _, test := range tests {
		s, _, _ := transform.String(transform.RemoveFunc(OnlyWordsAndNumbers), test.input)
		if s != test.expectedOutput {
			t.Errorf("got \"%s\" expected \"%s\"\n", s, test.expectedOutput)
		}
	}
}

func TestAsciiWordsAndNumbers(t *testing.T) {
	tests := []testCase{
		{"THIS iz A L337 aNd Un'Sani~~~~tized sentence", "THIS iz A L337 aNd UnSanitized sentence"},
		{"here're some unicode letters: --Æ.ÒÑ", "herere some unicode letters "},
		{")(*&^%$@!@#$%^&*(*&^%$#@#$%", ""},
	}
	for _, test := range tests {
		s, _, _ := transform.String(transform.RemoveFunc(OnlyAsciiWordsAndNumbers), test.input)
		if s != test.expectedOutput {
			t.Errorf("got \"%s\" expected \"%s\"\n", s, test.expectedOutput)
		}
	}
}

func TestWords(t *testing.T) {
	tests := []testCase{
		{"THIS iz A L337 aNd Un'Sani~~~~tized sentence", "THIS iz A L aNd UnSanitized sentence"},
		{"here're some unicode letters: --Æ.ÒÑ", "herere some unicode letters ÆÒÑ"},
		{")(*&^%$@!@#$%^&*(*&^%$#@#$%", ""},
	}
	for _, test := range tests {
		s, _, _ := transform.String(transform.RemoveFunc(OnlyWords), test.input)
		if s != test.expectedOutput {
			t.Errorf("got \"%s\" expected \"%s\"\n", s, test.expectedOutput)
		}
	}
}

func TestAsciiWords(t *testing.T) {
	tests := []testCase{
		{"THIS iz A L337 aNd Un'Sani~~~~tized sentence", "THIS iz A L aNd UnSanitized sentence"},
		{"here're some unicode letters: ÆÒÑ", "herere some unicode letters "},
		{")(*&^%$@!@#$%^&*(*&^%$#@#$%", ""},
	}
	for _, test := range tests {
		s, _, _ := transform.String(transform.RemoveFunc(OnlyAsciiWords), test.input)
		if s != test.expectedOutput {
			t.Errorf("got \"%s\" expected \"%s\"\n", s, test.expectedOutput)
		}
	}
}

func TestLetters(t *testing.T) {
	tests := []testCase{
		{"THIS iz A L337 aNd Un'Sani~~~~tized sentence", "THISizALaNdUnSanitizedsentence"},
		{"here're some unicode letters: --Æ.ÒÑ", "hereresomeunicodelettersÆÒÑ"},
		{")(*&^%$@!@#$%^&*(*&^%$#@#$%", ""},
	}
	for _, test := range tests {
		s, _, _ := transform.String(transform.RemoveFunc(OnlyLetters), test.input)
		if s != test.expectedOutput {
			t.Errorf("got \"%s\" expected \"%s\"\n", s, test.expectedOutput)
		}
	}
}

func TestAsciiLetters(t *testing.T) {
	tests := []testCase{
		{"THIS iz A L337 aNd Un'Sani~~~~tized sentence", "THISizALaNdUnSanitizedsentence"},
		{"here're some unicode letters: --Æ.ÒÑ", "hereresomeunicodeletters"},
		{")(*&^%$@!@#$%^&*(*&^%$#@#$%", ""},
	}
	for _, test := range tests {
		s, _, _ := transform.String(transform.RemoveFunc(OnlyAsciiLetters), test.input)
		if s != test.expectedOutput {
			t.Errorf("got \"%s\" expected \"%s\"\n", s, test.expectedOutput)
		}
	}
}
