package tokenizer

import (
	"reflect"
	"testing"
)

func Test_ReadNumber(t *testing.T) {
	tests := []struct {
		runes    []rune
		expected token
	}{
		{
			runes:    []rune("10000"),
			expected: token{Number, "10000"},
		},
		{
			runes:    []rune("10 000"),
			expected: token{Number, "10000"},
		},
		{
			runes:    []rune("10'000"),
			expected: token{Number, "10000"},
		},
		{
			runes:    []rune("10 000.305"),
			expected: token{Number, "10000.305"},
		},
		{
			runes:    []rune("10 000,305"),
			expected: token{Number, "10000,305"},
		},
		{
			runes:    []rune("10 000 305"),
			expected: token{Number, "10000305"},
		},
		{
			runes:    []rune("10 000рублей"),
			expected: token{Number, "10000"},
		},
		{
			runes:    []rune("10 000 рублей"),
			expected: token{Number, "10000"},
		},
		{
			runes:    []rune("10 000 рублей 305 000"),
			expected: token{Number, "10000"},
		},
		{
			runes:    []rune("рублей10 000"),
			expected: token{Number, ""},
		},
		{
			runes:    []rune("рублей 10 000"),
			expected: token{Number, ""},
		},
		{
			runes:    []rune("+ 10 000"),
			expected: token{Number, ""},
		},
		{
			runes:    []rune("- 10 000"),
			expected: token{Number, ""},
		},
		{
			runes:    []rune("-10 000"),
			expected: token{Number, "-10000"},
		},
		{
			runes:    []rune("+10 000"),
			expected: token{Number, "+10000"},
		},
		{
			runes:    []rune("+10 000.305"),
			expected: token{Number, "+10000.305"},
		},
		{
			runes:    []rune("10 000 + 305"),
			expected: token{Number, "10000"},
		},
		{
			runes:    []rune("10 000+305"),
			expected: token{Number, "10000"},
		},
		{
			runes:    []rune("10,005-305"),
			expected: token{Number, "10,005"},
		},
		{
			runes:    []rune("10,005*305"),
			expected: token{Number, "10,005"},
		},
		{
			runes:    []rune("10,005/305"),
			expected: token{Number, "10,005"},
		},
		{
			runes:    []rune("     10 005"),
			expected: token{Number, ""},
		},
		{
			runes:    []rune("     10 005"),
			expected: token{Number, ""},
		},
		{
			runes:    []rune("10 005     "),
			expected: token{Number, "10005"},
		},
		{
			runes:    []rune("р10 005"),
			expected: token{Number, ""},
		},
		{
			runes:    []rune("10 005р"),
			expected: token{Number, "10005"},
		},
		{
			runes:    []rune(".10 005"),
			expected: token{Number, ""},
		},
		{
			runes:    []rune(",10 005"),
			expected: token{Number, ""},
		},
		{
			runes:    []rune("10,10 005"),
			expected: token{Number, "10,10005"},
		},
		{
			runes:    []rune("10, 10 005"),
			expected: token{Number, "10"},
		},
		{
			runes:    []rune("1900год"),
			expected: token{Number, "1900"},
		},
	}

	for i, test := range tests {
		res, _ := readNumber(test.runes)

		if res != test.expected {
			t.Errorf("%d: ReadNumber(%s) -> %v; expected: %v", i+1, string(test.runes),
				res, test.expected)
		}
	}
}

func Test_ReadWord(t *testing.T) {
	tests := []struct {
		runes    []rune
		expected token
	}{
		{
			runes:    []rune("Шахматы"),
			expected: token{Word, "шахматы"},
		},
		{
			runes:    []rune("ШАХМАТЫ"),
			expected: token{Word, "шахматы"},
		},
		{
			runes:    []rune("шахматы"),
			expected: token{Word, "шахматы"},
		},
		{
			runes:    []rune("ШаХмАтЫ"),
			expected: token{Word, "шахматы"},
		},
		{
			runes:    []rune("ШАХматЫ"),
			expected: token{Word, "шахматы"},
		},
		{
			runes:    []rune("Chess"),
			expected: token{Word, "chess"},
		},
		{
			runes:    []rune("CheSS"),
			expected: token{Word, "chess"},
		},
		{
			runes:    []rune("chessшахматы"),
			expected: token{Word, "chessшахматы"},
		},
		{
			runes:    []rune("chessшахматы"),
			expected: token{Word, "chessшахматы"},
		},
		{
			runes:    []rune("КОМПью́терная"),
			expected: token{Word, "компьютерная"},
		},
		{
			runes:    []rune("ИГРА́"),
			expected: token{Word, "игра"},
		},
		{
			runes:    []rune("пи́кИ"),
			expected: token{Word, "пики"},
		},
		{
			runes:    []rune("та́ро"),
			expected: token{Word, "таро"},
		},
		{
			runes:    []rune("Кто-ТО"),
			expected: token{Word, "кто-то"},
		},
		{
			runes:    []rune("ИнТеРнЕт-МАГаЗИН"),
			expected: token{Word, "интернет-магазин"},
		},
		{
			runes:    []rune("Ростов-на-Дону"),
			expected: token{Word, "ростов-на-дону"},
		},
		{
			runes:    []rune("-магазин"),
			expected: token{Word, ""},
		},
		{
			runes:    []rune("интернет - магазин"),
			expected: token{Word, "интернет"},
		},
		{
			runes:    []rune("интернет- магазин"),
			expected: token{Word, "интернет"},
		},
		{
			runes:    []rune("интернет -магазин"),
			expected: token{Word, "интернет"},
		},
		{
			runes:    []rune("CHECK-in"),
			expected: token{Word, "check-in"},
		},
		{
			runes:    []rune("CHECK-in"),
			expected: token{Word, "check-in"},
		},
		{
			runes:    []rune("А100"),
			expected: token{Word, "а100"},
		},
		{
			runes:    []rune("B2B"),
			expected: token{Word, "b2b"},
		},
		{
			runes:    []rune("Т34"),
			expected: token{Word, "т34"},
		},
		{
			runes:    []rune("Т-34"),
			expected: token{Word, "т-34"},
		},
		{
			runes:    []rune("1900год"),
			expected: token{Word, ""},
		},
		{
			runes:    []rune("     интернет"),
			expected: token{Word, ""},
		},
		{
			runes:    []rune(" интернет"),
			expected: token{Word, ""},
		},
		{
			runes:    []rune("интернет "),
			expected: token{Word, "интернет"},
		},
		{
			runes:    []rune("ИНТЕРНЕТ     chess"),
			expected: token{Word, "интернет"},
		},
		{
			runes:    []rune("X5"),
			expected: token{Word, "x5"},
		},
		{
			runes:    []rune("ОБ’явление"),
			expected: token{Word, "об’явление"},
		},
		{
			runes:    []rune("can’t"),
			expected: token{Word, "can’t"},
		},
		{
			runes:    []rune("’t"),
			expected: token{Word, ""},
		},
		{
			runes:    []rune("T’"),
			expected: token{Word, "t’"},
		},
	}

	for i, test := range tests {
		res, _ := readWord(test.runes)

		if res != test.expected {
			t.Errorf("%d: ReadWord(%s) -> %v; expected: %v", i+1, string(test.runes),
				res, test.expected)
		}
	}
}

func Test_tokenize(t *testing.T) {
	tests := []struct {
		text     string
		expected []token
	}{
		{
			text: "Сколько? 10!",
			expected: []token{
				{Word, "сколько"}, {Punct, "?"}, {Number, "10"}, {Punct, "!"},
				{Special, EndOfSeq},
			},
		},
		{
			text: "интернет10.!.",
			expected: []token{
				{Word, "интернет10"}, {Punct, "."}, {Punct, "!"}, {Punct, "."},
				{Special, EndOfSeq},
			},
		},
		{
			text: " ИНТЕРНЕТ10.!. ",
			expected: []token{
				{Word, "интернет10"}, {Punct, "."}, {Punct, "!"}, {Punct, "."},
				{Special, EndOfSeq},
			},
		},
		{
			text: "     ИнтерНЕТ10   .!.     ",
			expected: []token{
				{Word, "интернет10"}, {Punct, "."}, {Punct, "!"}, {Punct, "."},
				{Special, EndOfSeq},
			},
		},
		{
			text: "     ИНТернет10.!?     \n\n\n интернет10! ",
			expected: []token{
				{Word, "интернет10"}, {Punct, "."}, {Punct, "!"}, {Punct, "?"},
				{Special, BreakLine}, {Word, "интернет10"}, {Punct, "!"},
				{Special, EndOfSeq},
			},
		},
		{
			text: " 10РУБЛЕЙ? ",
			expected: []token{
				{Number, "10"}, {Word, "рублей"}, {Punct, "?"},
				{Special, EndOfSeq},
			},
		},
		{
			text: " -10Рублей! ",
			expected: []token{
				{Number, "-10"}, {Word, "рублей"}, {Punct, "!"},
				{Special, EndOfSeq},
			},
		},
		{
			text: " -10 Рублей-_-! ",
			expected: []token{
				{Number, "-10"}, {Word, "рублей"}, {Punct, "-"}, {Punct, "_"}, {Punct, "-"}, {Punct, "!"},
				{Special, EndOfSeq},
			},
		},
		{
			text: "10 - 305=-305+10",
			expected: []token{
				{Number, "10"}, {Punct, "-"}, {Number, "305"}, {Unknown, "="}, {Number, "-305"}, {Unknown, "+"}, {Number, "10"},
				{Special, EndOfSeq},
			},
		},
		{
			text: "-305     =-305\n\n",
			expected: []token{
				{Number, "-305"}, {Unknown, "="}, {Number, "-305"}, {Special, BreakLine},
				{Special, EndOfSeq},
			},
		},
		{
			text: "Видеоигра́ (англ. video game) — игра",
			expected: []token{
				{Word, "видеоигра"}, {Punct, "("}, {Word, "англ"}, {Punct, "."}, {Word, "video"},
				{Word, "game"}, {Punct, ")"}, {Punct, "—"}, {Word, "игра"},
				{Special, EndOfSeq},
			},
		},
		{
			text: "В 1970-х видеоигры",
			expected: []token{
				{Word, "в"}, {Number, "1970"}, {Punct, "-"}, {Word, "х"}, {Word, "видеоигры"},
				{Special, EndOfSeq},
			},
		},
		{
			text: "написано 64 665 035 статей.",
			expected: []token{
				{Word, "написано"}, {Number, "64665035"}, {Word, "статей"}, {Punct, "."},
				{Special, EndOfSeq},
			},
		},
		{
			text: "Веще́ственное число́ (действи́тельное число)",
			expected: []token{
				{Word, "вещественное"}, {Word, "число"}, {Punct, "("}, {Word, "действительное"}, {Word, "число"}, {Punct, ")"},
				{Special, EndOfSeq},
			},
		},
		{
			text: "разрядами числа π = 3,14",
			expected: []token{
				{Word, "разрядами"}, {Word, "числа"}, {Word, "π"}, {Unknown, "="}, {Number, "3,14"},
				{Special, EndOfSeq},
			},
		},
		{
			text: "Шахматы (перс., шах мат — властитель умер) —" +
				" детерминированная игра с совершенной информацией, в ",
			expected: []token{
				{Word, "шахматы"}, {Punct, "("}, {Word, "перс"}, {Punct, "."}, {Punct, ","}, {Word, "шах"}, {Word, "мат"},
				{Punct, "—"}, {Word, "властитель"}, {Word, "умер"}, {Punct, ")"}, {Punct, "—"}, {Word, "детерминированная"},
				{Word, "игра"}, {Word, "с"}, {Word, "совершенной"}, {Word, "информацией"}, {Punct, ","}, {Word, "в"},
				{Special, EndOfSeq},
			},
		},
		{
			text: "Шахматы (перс., шах мат — властитель умер) —\n\n\n" +
				" детерминированная игра \n\nс совершенной информацией, в \n",
			expected: []token{
				{Word, "шахматы"}, {Punct, "("}, {Word, "перс"}, {Punct, "."}, {Punct, ","},
				{Word, "шах"}, {Word, "мат"}, {Punct, "—"}, {Word, "властитель"}, {Word, "умер"},
				{Punct, ")"}, {Punct, "—"}, {Special, BreakLine}, {Word, "детерминированная"}, {Word, "игра"},
				{Special, BreakLine}, {Word, "с"}, {Word, "совершенной"}, {Word, "информацией"}, {Punct, ","},
				{Word, "в"}, {Special, BreakLine}, {Special, EndOfSeq},
			},
		},
		{
			text: "Дзюндзи Ито (яп. 伊藤 潤二 Ито: Дзюндзи, родился 31 июля 1963 года) — " +
				"японский мангака, работающий в жанре ужасов. " +
				"Его самые известные работы: Uzumaki, Gyo и «Томиэ». \n\n" +
				"В 2019 году за работу Frankenstein: Junji Ito Story Collection и в 2021 году " +
				"за Remina и Venus in the Blind Spot получил премию Айснера.",
			expected: []token{
				{Word, "дзюндзи"}, {Word, "ито"}, {Punct, "("}, {Word, "яп"}, {Punct, "."},
				{Word, "伊藤"}, {Word, "潤二"}, {Word, "ито"}, {Punct, ":"}, {Word, "дзюндзи"},
				{Punct, ","}, {Word, "родился"}, {Number, "31"}, {Word, "июля"}, {Number, "1963"},
				{Word, "года"}, {Punct, ")"}, {Punct, "—"},
				{Word, "японский"}, {Word, "мангака"}, {Punct, ","}, {Word, "работающий"}, {Word, "в"},
				{Word, "жанре"}, {Word, "ужасов"}, {Punct, "."},
				{Word, "его"}, {Word, "самые"}, {Word, "известные"}, {Word, "работы"}, {Punct, ":"},
				{Word, "uzumaki"}, {Punct, ","}, {Word, "gyo"}, {Word, "и"}, {Punct, "«"},
				{Word, "томиэ"}, {Punct, "»"}, {Punct, "."}, {Special, BreakLine},
				{Word, "в"}, {Number, "2019"}, {Word, "году"}, {Word, "за"}, {Word, "работу"},
				{Word, "frankenstein"}, {Punct, ":"}, {Word, "junji"}, {Word, "ito"}, {Word, "story"},
				{Word, "collection"}, {Word, "и"}, {Word, "в"}, {Number, "2021"}, {Word, "году"},
				{Word, "за"}, {Word, "remina"}, {Word, "и"}, {Word, "venus"}, {Word, "in"}, {Word, "the"},
				{Word, "blind"}, {Word, "spot"}, {Word, "получил"}, {Word, "премию"}, {Word, "айснера"},
				{Punct, "."}, {Special, EndOfSeq},
			},
		},
	}

	for i, test := range tests {
		res := Tokenize(test.text)
		arr := make([]token, 0)
		for t, v := range res {
			arr = append(arr, token{t, v})
		}

		if !reflect.DeepEqual(arr, test.expected) {
			t.Errorf("%d: tokenize(%s) -> %v; expected: %v", i+1, test.text,
				arr, test.expected)
		}
	}
}
