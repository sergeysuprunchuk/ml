package bpe

import (
	"encoding/json"
	"go.uber.org/zap"
	"ml/pkg/dirreader"
	"ml/pkg/tokenizer"
	"os"
	"path"
	"slices"
	"sort"
	"strings"
)

const UNK = "</unk>"
const PAD = "</pad>"
const EOW = "</w>"
const EOT = "</eot>"

type BPE struct {
	Dict   Dict `json:"dict"`
	UnkPos int  `json:"unkPos"`
	EowPos int  `json:"eowPos"`
	EotPos int  `json:"eotPos"`
	PadPos int  `json:"padPos"`
}

func New() *BPE {
	return &BPE{
		Dict: make(Dict, 0),
	}
}

type Dict []string

func (d *Dict) Has(v string) bool {
	return slices.Contains(*d, v)
}

// пара токенов
type pair [2]string

type word struct {
	//сколько раз слово встречалось в тексте
	n int
	//токены слова: слово -> с, л, о, в, о, конец слова
	toks []string //имеет смысл перейти на связанный список
}

// getPair возвращает наиболее часто встречающуюся пару токенов
// и сколько раз эта пара встречается в тексте.
// возвращает 0, если само слово является токеном.
func (w *word) getPair() (pair, int) {
	counter := make(map[pair]int, len(w.toks)-1)
	var mpair pair
	var mn int

	for i := 0; i < len(w.toks)-1; i++ {
		p := pair{w.toks[i], w.toks[i+1]}
		counter[p]++
		if counter[p] > mn {
			mn = counter[p]
			mpair = p
		}
	}

	return mpair, mn * w.n
}

// merge склеивает пары токенов
func (w *word) merge(p pair) bool {
	var m bool

	for i := 0; i < len(w.toks)-1; i++ {
		if p[0] == w.toks[i] && p[1] == w.toks[i+1] {
			w.toks[i] += w.toks[i+1]
			w.toks = append(w.toks[:i+1], w.toks[i+2:]...)
			m = true
		}
	}

	return m
}

type wordDict struct {
	// Ключ - это слово
	value map[string]*word
}

func newWordDict() *wordDict {
	return &wordDict{
		value: make(map[string]*word, 131072),
	}
}

func (d *wordDict) add(w string) {
	_, ok := d.value[w]
	if !ok {
		toks := []string{w}

		if !strings.HasPrefix(w, tokenizer.SpecTokStart) &&
			!strings.HasSuffix(w, tokenizer.SpecTokEnd) {
			toks = append(strings.Split(w, ""), "</w>")
		}

		d.value[w] = &word{toks: toks}
	}
	d.value[w].n++
}

// toBPEDict создает словарь подслов с размером size.
// отфильтровывает слова с редкостью <= rarity.
//
//	итоговый BPEDict не отсортирован!
func (d *wordDict) toBPEDict(size, rarity int) Dict {
	bpeDict := make(Dict, 0, size)

	//собираю базовый словарь и отфильтровываю редкие слова
	chars := make(map[string]struct{}, 256)
	for key, val := range d.value {
		//слишком редкое слово
		if val.n <= rarity {
			delete(d.value, key)
			continue
		}

		for _, tok := range val.toks {
			if _, ok := chars[tok]; !ok {
				chars[tok] = struct{}{}
				bpeDict = append(bpeDict, tok)
			}
		}
	}

	for range size - len(bpeDict) {
		counter := make(map[pair]int, len(d.value))
		var mpair pair
		var mn int

		for _, val := range d.value {
			p, n := val.getPair()
			counter[p] += n
			if counter[p] > mn {
				mn = counter[p]
				mpair = p
			}
		}

		if mn == 0 {
			break
		}

		bpeDict = append(bpeDict, mpair[0]+mpair[1])

		zap.S().Infof("%d слово в bpe: %s", len(bpeDict), mpair[0]+mpair[1])

		for _, val := range d.value {
			val.merge(mpair)
		}
	}

	return bpeDict
}

func (b *BPE) Learn(src string, tokN int) error {
	wDict := newWordDict()

	var count int

	for file := range dirreader.Read(src) {
		for _, tokVal := range tokenizer.Tokenize(string(file.Data)) {
			wDict.add(tokVal)
		}
		count++
		zap.S().Infof("обработано %d файлов; всего слов %d", count, len(wDict.value))
	}

	zap.S().Info("заполнил словарь слов")

	b.Dict = wDict.toBPEDict(tokN, 2)

	sort.Strings(b.Dict)

	return nil
}

func (b *BPE) Learn2(src []string, tokN int) error {
	wDict := newWordDict()

	var count int

	for _, file := range src {
		for _, tokVal := range tokenizer.Tokenize(file) {
			wDict.add(tokVal)
		}
		count++
		zap.S().Infof("обработано %d файлов; всего слов %d", count, len(wDict.value))
	}

	zap.S().Info("заполнил словарь слов")

	b.Dict = wDict.toBPEDict(tokN, 0)

	b.Dict = append(b.Dict, UNK, PAD, EOW, EOT)

	sort.Strings(b.Dict)

	return nil
}

func (b *BPE) Save(to string) error {
	p := path.Join(to, "tokens.json")

	f, err := os.Create(p)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.
		NewEncoder(f)
	enc.SetIndent("", "\t")

	return enc.
		Encode(b.Dict)
}

func (b *BPE) Load(src string) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}

	err = json.Unmarshal(data, &b.Dict)
	if err != nil {
		return err
	}

	sort.Strings(b.Dict)

	var ok bool
	b.UnkPos, ok = slices.BinarySearch(b.Dict, UNK)
	if !ok {
		panic("!b.UnkPos")
	}
	b.PadPos, ok = slices.BinarySearch(b.Dict, PAD)
	if !ok {
		panic("!b.PadPos")
	}
	b.EotPos, ok = slices.BinarySearch(b.Dict, EOT)
	if !ok {
		panic("b.EotPos")
	}

	return nil
}

func (b *BPE) Tokenize(text string) []string {
	toks := make([]string, 0)

	for tokTyp, tokVal := range tokenizer.Tokenize(text) {
		if tokTyp == tokenizer.Special {
			toks = append(toks, tokVal)
			continue
		}

		loc := append(strings.Split(tokVal, ""), "</w>")

		for {
			end := true

			for i := 0; i < len(loc)-1; {
				if b.Dict.Has(loc[i] + loc[i+1]) {
					loc[i] += loc[i+1]
					loc = append(loc[:i+1], loc[i+2:]...)
					end = false
					i = 0
				}
				i++
			}

			if end {
				break
			}
		}

		toks = append(toks, loc...)

	}

	return toks
}

func (b *BPE) Mark(toks []string) []int {
	marks := make([]int, 0, len(toks))

	for _, tok := range toks {
		m, ok := slices.BinarySearch(b.Dict, tok)
		if !ok {
			marks = append(marks, b.UnkPos)
			continue
		}

		marks = append(marks, m)
	}

	return marks
}
