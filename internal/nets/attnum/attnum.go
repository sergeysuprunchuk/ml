package attnum

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go.uber.org/zap"
	"io"
	"iter"
	"ml/pkg/attention"
	"ml/pkg/dirreader"
	"ml/pkg/mat"
	"ml/pkg/mlutil"
	"os"
	"strconv"
)

type AttNum struct {
	MH *attention.MultiHead `json:"mh"`
}

func (num *AttNum) Learn(src string, epochs int, lrate float64) {
	var data []example
	for ex := range dataset(src) {
		data = append(data, ex)
	}

	zap.S().Info("start")

	for epoch := range epochs {
		mlutil.Shuffle(data)

		var err float64

		for i, ex := range data {
			probs := num.MH.Forward(ex.inp).Softmax()
			truth := mat.New(probs.RowN(), probs.ColN())
			truth[0][ex.ans] = 1
			err += probs.CrossEntropy(truth)
			num.MH.Backward(probs.Sub(truth), lrate)
			if i%256 == 0 && i != 0 {
				zap.S().Infof("эпоха %d: ошибка %.4f",
					epoch+1, err/float64(i))
				err = 0
			}
		}
	}
}

func (num *AttNum) Save(to string) error {
	file, err := os.Create(to)
	if err != nil {
		return err
	}
	defer file.Close()

	return json.
		NewEncoder(file).
		Encode(num)
}

func Load(src string) (*AttNum, error) {
	file, err := os.Open(src)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var num AttNum

	return &num, json.
		NewDecoder(file).
		Decode(&num)
}

func (num *AttNum) Query(r io.Reader) mat.Mat {
	return num.MH.
		Forward(mlutil.Img2vec(r)).
		Softmax()
}

func New(xrown, xcoln, wcoln, h int) *AttNum {
	return &AttNum{
		MH: attention.NewMultiHead(xrown, xcoln, wcoln, h, 10),
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

func (num *AttNum) Test(src string) {
	var err int

	for ex := range dataset(src) {
		probs := num.MH.Forward(ex.inp).Softmax()

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
