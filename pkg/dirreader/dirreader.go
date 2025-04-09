package dirreader

import (
	"iter"
	"os"
	"path"
)

type File struct {
	Path []string
	Data []byte
}

func Read(src string) iter.Seq[File] {
	return func(yield func(File) bool) {
		read([]string{src}, yield)
	}
}

func read(
	p []string,
	yield func(File) bool,
) bool {
	dir, err := os.ReadDir(path.Join(p...))
	if err != nil {
		panic(err)
	}

	for _, entry := range dir {
		newp := append(p, entry.Name())

		if entry.IsDir() {
			if !read(newp, yield) {
				return false
			}
			continue
		}

		data, err := os.ReadFile(path.Join(newp...))
		if err != nil {
			panic(err)
		}

		if !yield(File{Path: newp, Data: data}) {
			return false
		}
	}

	return true
}
