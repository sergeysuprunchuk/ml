package dirreader

import (
	"path"
	"testing"
)

const tests = "./tests"

func Test_Read(t *testing.T) {
	files := map[string]string{
		path.Join(tests, "16", "16.txt"): "consectetur adipisicing",
		path.Join(tests, "32", "32.txt"): "consectetur",
		path.Join(tests, "8", "4.txt"):   "lorem ipsum dolor",
		path.Join(tests, "8", "8.txt"):   "sit amet",
	}

	//TODO: сделать папку с текстовыми файлами и удалить эту папку после теста

	var got []File

	for file := range Read(tests) {
		got = append(got, file)
	}

	if len(files) != len(got) {
		t.Errorf("len(files) != len(got)")
		return
	}

	for _, i := range got {
		p := path.Join(i.Path...)

		data, ok := files[p]
		if !ok {
			t.Errorf("файл %s отсутствует", p)
			continue
		}

		if data != string(i.Data) {
			t.Errorf("файл %s: ожидалось %s, получено %s",
				p, data, string(i.Data))
		}
	}
}
