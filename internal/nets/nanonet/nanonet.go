package nanonet

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go.uber.org/zap"
	"iter"
	"ml/pkg/dirreader"
	"ml/pkg/laynorm"
	"ml/pkg/mat"
	"ml/pkg/mlp"
	"ml/pkg/mlutil"
	"os"
	"strconv"
)

type NanoNet struct {
}

//сперва попробовать объединить norm и mlp и обучить числам.
//если работает, то приступать к созданию сети.
//учить буду как большую языковую модель - batch = 1

type Example struct {
	inp mat.Mat
	ans int
}

func Dataset(src string) iter.Seq[Example] {
	return func(yield func(Example) bool) {
		for file := range dirreader.Read(src) {
			ans, err := strconv.Atoi(file.Path[len(file.Path)-2])
			if err != nil {
				panic(err)
			}

			if !yield(Example{
				inp: mlutil.Img2vec(bytes.NewReader(file.Data)),
				ans: ans,
			}) {
				break
			}
		}
	}
}

type NumMLP struct {
	MLP *mlp.MLP
}

func (num *NumMLP) Learn(src string, epochs int, lrate float64) {
	var dataset []Example
	for i := range Dataset(src) {
		dataset = append(dataset, i)
	}

	for epoch := range epochs {
		mlutil.Shuffle(dataset)
		for _, i := range dataset {
			probs := num.MLP.Forward(i.inp).Softmax()

			truth := mat.New(probs.RowN(), probs.ColN())
			truth[0][i.ans] = 1

			zap.S().Infof("%d: error: %.4f", epoch, probs.CrossEntropy(truth))

			num.MLP.BackwardMut(probs.Sub(truth), lrate)
		}
	}
}

func (num *NumMLP) Test(src string) {
	var err int

	for ex := range Dataset(src) {
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

func (num *NumMLP) Save(to string) {
	file, err := os.Create(to)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	err = json.
		NewEncoder(file).
		Encode(num)
	if err != nil {
		panic(err)
	}
}

func NewNumMLP(xrown, xcoln int, wcolns ...int) *NumMLP {
	return &NumMLP{
		MLP: mlp.New(.01, xrown, xcoln, wcolns...),
	}
}

type NumMLPNorm struct {
	MLP     *mlp.MLP
	LayNorm *laynorm.LayNorm
	MLP2    *mlp.MLP
}

func (num *NumMLPNorm) Save(to string) {
	file, err := os.Create(to)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	err = json.
		NewEncoder(file).
		Encode(num)
	if err != nil {
		panic(err)
	}
}

func (num *NumMLPNorm) Learn(src string, epochs int, lrate float64) {
	var dataset []Example
	for i := range Dataset(src) {
		dataset = append(dataset, i)
	}

	for epoch := range epochs {
		mlutil.Shuffle(dataset)
		for _, i := range dataset {
			probs := num.MLP2.Forward(
				num.LayNorm.Forward(
					num.MLP.Forward(i.inp)),
			).Softmax()

			truth := mat.New(probs.RowN(), probs.ColN())
			truth[0][i.ans] = 1

			zap.S().Infof("%d: error: %.4f", epoch, probs.CrossEntropy(truth))

			num.MLP.BackwardMut(
				num.LayNorm.Backward(
					num.MLP2.BackwardMut(probs.Sub(truth), lrate), lrate), lrate)
		}
	}
}

func NewNumMLPNorm(xrown, xcoln int, wcoln1, wcoln2 int) *NumMLPNorm {
	return &NumMLPNorm{
		MLP:     mlp.New(.01, xrown, xcoln, wcoln1),
		LayNorm: laynorm.New(wcoln1),
		MLP2:    mlp.New(.01, xrown, wcoln1, wcoln2),
	}
}

func (num *NumMLPNorm) Test(src string) {
	var err int

	for ex := range Dataset(src) {
		probs := num.MLP2.Forward(
			num.LayNorm.Forward(
				num.MLP.Forward(ex.inp)),
		).Softmax()
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
