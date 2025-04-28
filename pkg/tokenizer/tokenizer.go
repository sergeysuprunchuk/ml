package tokenizer

import (
	"iter"
	"unicode"
)

const (
	Word    = "word"
	Number  = "number"
	Punct   = "punct"
	Special = "special"
	Unknown = "unknown"
)

const (
	SpecTokStart = "</"
	SpecTokEnd   = ">"
)

const (
	BreakLine = SpecTokStart + "bl" + SpecTokEnd
	EndOfSeq  = SpecTokStart + "eos" + SpecTokEnd
)

// Первый элемент — тип, второй — значение
type token [2]string

func Tokenize(text string) iter.Seq2[string, string] {
	return func(yield func(string, string) bool) {
		runes := []rune(text)

		var prev token
		var tok token
		var offset int

		for i := 0; i < len(runes); {
			char := runes[i]

			switch {
			case unicode.IsLetter(char):
				tok, offset = readWord(runes[i:])

			case unicode.IsDigit(char) ||
				((char == '-' || char == '+') &&
					prev[0] != Number &&
					i+1 != len(runes) &&
					unicode.IsDigit(runes[i+1])):
				tok, offset = readNumber(runes[i:])

			case unicode.IsPunct(char):
				tok = token{Punct, string(char)}
				offset = 1

			case char == '\n' &&
				(prev[0] != Special ||
					prev[1] != BreakLine):
				tok = token{Special, BreakLine}
				offset = 1

			case unicode.IsSpace(char):
				i++
				continue

			default:
				tok = token{Unknown, string(char)}
				offset = 1
			}

			prev = tok
			i += offset
			if !yield(tok[0], tok[1]) {
				return
			}
		}
	}
}

func readWord(runes []rune) (token, int) {
	value := make([]rune, 0, 8)

	var i int
loop:
	for ; i < len(runes); i++ {
		char := runes[i]

		switch {
		case unicode.IsLetter(char):
			fallthrough
		case unicode.IsDigit(char) && len(value) != 0:
			fallthrough
		case char == '\'' || char == '’' && len(value) != 0:
			fallthrough
		case char == '-' &&
			len(value) != 0 &&
			i+1 != len(runes) && (unicode.IsLetter(runes[i+1]) || unicode.IsDigit(runes[i+1])):
			value = append(value, unicode.ToLower(char))

			//Ударение
		case char == '́':
			continue loop

		default:
			break loop
		}
	}

	return token{Word, string(value)}, i
}

func readNumber(runes []rune) (token, int) {
	value := make([]rune, 0, 8)

	var i int
loop:
	for ; i < len(runes); i++ {
		char := runes[i]

		switch {
		case unicode.IsDigit(char) ||
			((char == '-' || char == '+') &&
				len(value) == 0 &&
				i+1 != len(runes) &&
				unicode.IsDigit(runes[i+1])):
			value = append(value, char)

		case i+1 != len(runes) && unicode.IsDigit(runes[i+1]) && len(value) != 0:
			switch char {
			case ' ', '\'':
				continue loop
			case '.', ',':
				value = append(value, char)
			default:
				break loop
			}

		default:
			break loop
		}
	}

	return token{Number, string(value)}, i
}
