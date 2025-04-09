package num

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go.uber.org/zap"
	"io"
	"iter"
	"ml/pkg/dirreader"
	"ml/pkg/mat"
	"ml/pkg/mlp"
	"ml/pkg/mlutil"
	"os"
	"slices"
	"strconv"
)

type Num struct {
	MLP *mlp.MLP `json:"mlp"`
}

func (num *Num) Learn(src string, epochs, pkgSize int, lrate float64) {
	var data []example
	for ex := range dataset(src) {
		data = append(data, ex)
	}

	for epoch := range epochs {
		mlutil.Shuffle(data)
		var pkgi int

		for pkg := range slices.Chunk(data, pkgSize) {
			var err float64
			dlays := make([]mlp.DLayer, 0, len(num.MLP.Lays))

			for _, ex := range pkg {
				probs := num.MLP.Forward(ex.inp).Softmax()

				truth := mat.New(probs.RowN(), probs.ColN())
				truth[0][ex.ans] = 1

				err += probs.CrossEntropy(truth)

				for i, newDlayer := range num.MLP.Backward(probs.Sub(truth)) {
					if i >= len(dlays) {
						dlays = append(dlays, newDlayer)
						continue
					}

					dlays[i].Weight = dlays[i].Weight.Add(newDlayer.Weight)

					dlays[i].Bias = dlays[i].Bias.Add(newDlayer.Bias)
				}
			}

			zap.S().Infof("эпоха %d, пакет %d: ошибка %.4f",
				epoch+1, pkgi+1, err/float64(pkgSize))

			for i := range dlays {
				dlays[i].Weight = dlays[i].Weight.Scale(1.0 / float64(pkgSize))
				dlays[i].Bias = dlays[i].Bias.Scale(1.0 / float64(pkgSize))
			}

			num.MLP.Update(dlays, lrate)

			pkgi++
		}
	}
}

func (num *Num) Save(to string) error {
	file, err := os.Create(to)
	if err != nil {
		return err
	}
	defer file.Close()

	return json.
		NewEncoder(file).
		Encode(num)
}

func Load(src string) (*Num, error) {
	file, err := os.Open(src)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var num Num

	return &num, json.
		NewDecoder(file).
		Decode(&num)
}

func (num *Num) Query(r io.Reader) mat.Mat {
	return num.MLP.
		Forward(mlutil.Img2vec(r)).
		Softmax()
}

func New(alpha float64, xrown, xcoln int, wcolns ...int) *Num {
	return &Num{
		MLP: mlp.New(alpha, xrown, xcoln, wcolns...),
	}
}

type example struct {
	inp mat.Mat
	ans int
}

func dataset(src string) iter.Seq[example] {
	return func(yield func(example) bool) {
		for file := range dirreader.Read(src) {
			ans, err := strconv.Atoi(file.Path[len(file.Path)-2])
			if err != nil {
				panic(err)
			}

			if !yield(example{
				inp: mlutil.Img2vec(bytes.NewReader(file.Data)),
				ans: ans,
			}) {
				break
			}
		}
	}
}

func (num *Num) Test(src string) {
	var err int

	for ex := range dataset(src) {
		probs := num.MLP.Forward(ex.inp).Softmax()

		maxi := 0
		maxn := probs[0][maxi]
		for i := range probs[0] {
			if probs[0][i] > maxn {
				maxn = probs[0][i]
				maxi = i
			}
		}

		for i, prob := range probs[0] {
			fmt.Print(i, "->", fmt.Sprintf("%.2f", prob), " ")
		}
		fmt.Printf("правильный %d, предсказанный %d\n", ex.ans, maxi)

		if ex.ans != maxi {
			err++
		}
	}

	fmt.Printf("всего ошибок %d\n", err)
}
