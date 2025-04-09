package mlutil

import (
	"image/jpeg"
	"io"
	"math/rand"
	"ml/pkg/mat"
)

func Shuffle[T any](sl []T) {
	for range len(sl) {
		a, b := rand.Intn(len(sl)), rand.Intn(len(sl))
		sl[a], sl[b] = sl[b], sl[a]
	}
}

func Img2vec(r io.Reader) mat.Mat {
	jpg, err := jpeg.Decode(r)
	if err != nil {
		panic(err)
	}

	maxY := jpg.Bounds().Max.Y
	maxX := jpg.Bounds().Max.X

	m := mat.New(1, maxX*maxY)

	for y := range maxY {
		for x := range maxX {
			r, g, b, _ := jpg.At(x, y).RGBA()

			fr, fg, fb := float64(r>>8), float64(g>>8), float64(b>>8)

			m[0][x+(y*maxX)] = (.3*fr + .585*fg + .115*fb) / 255
		}
	}

	return m
}
