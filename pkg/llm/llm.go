package llm

import (
	"encoding/json"
	"fmt"
	"go.uber.org/zap"
	"math"
	"ml/pkg/attention"
	"ml/pkg/bpe"
	"ml/pkg/laynorm"
	"ml/pkg/mat"
	"ml/pkg/mlp"
	"ml/pkg/mlutil"
	"os"
	"strings"
)

type Layer struct {
	MLP     *mlp.MLP             `json:"mlp"`
	MHA     *attention.MultiHead `json:"mha"`
	MHANorm *laynorm.LayNorm     `json:"mhanorm"`
	MLPNorm *laynorm.LayNorm     `json:"mlpnorm"`
	mhaInp  mat.Mat
	mlpInp  mat.Mat
}

func (l *Layer) Forward(x mat.Mat) mat.Mat {
	l.mhaInp = x
	mhaAns := l.MHA.Forward(l.mhaInp)
	l.mlpInp = l.MHANorm.Forward(mhaAns.Add(l.mhaInp))
	mlpAns := l.MLP.Forward(l.mlpInp)
	return l.MLPNorm.Forward(mlpAns.Add(l.mlpInp))
}

func (l *Layer) Backward(do mat.Mat, lrate float64) mat.Mat {
	dMLPNorm := l.MLPNorm.Backward(do, lrate)
	dMLP := l.MLP.BackwardMut(dMLPNorm, lrate)
	dMHANorm := l.MHANorm.Backward(dMLP.Add(dMLPNorm), lrate)
	dMHA := l.MHA.Backward(dMHANorm, lrate)
	return dMHA.Add(dMHANorm)
}

func NewLayer(xrown, xcoln, wcoln, h int, alpha float64) *Layer {
	return &Layer{
		MHA:     attention.NewMultiHead(xrown, xcoln, wcoln, h),
		MLP:     mlp.New(alpha, xrown, xcoln, xcoln*8, xcoln),
		MHANorm: laynorm.New(xcoln),
		MLPNorm: laynorm.New(xcoln),
	}
}

type LLM struct {
	//Отсортированный словарь токенов
	Dict    *bpe.BPE `json:"dict"`
	Embs    mat.Mat  `json:"embs"`
	Layers  []*Layer `json:"layers"`
	CtxSize int      `json:"ctxSize"`
	Pos     mat.Mat  `json:"pos"`

	x, embs mat.Mat
}

func New(layerN,
	ctxSize,
	embSize,
	wcoln,
	headN int,
	alpha float64,
	dictSrc string,
) *LLM {
	layers := make([]*Layer, 0, layerN)
	for range layerN {
		layers = append(layers,
			NewLayer(ctxSize, embSize, wcoln, headN, alpha))
	}

	dict := bpe.New()
	err := dict.Load(dictSrc)
	if err != nil {
		panic(err)
	}

	embs := mat.New(len(dict.Dict), embSize).Rand()
	embs[dict.PadPos] = mat.New(1, embSize)[0]

	return &LLM{
		Dict:    dict,
		Embs:    embs,
		Layers:  layers,
		CtxSize: ctxSize,
		Pos:     mat.New(ctxSize, embSize).Rand(),
	}
}

func (llm *LLM) Learn(text string, lrate float64, fileName string) {
	marks := llm.Dict.Mark(llm.Dict.Tokenize(text))
	oneHot := mat.New(len(marks), llm.Embs.RowN())
	oneHot.OneHot(marks)

	pad := make([]float64, llm.Embs.RowN())
	pad[llm.Dict.PadPos] = 1

	for oneHot.RowN() <= llm.CtxSize {
		oneHot = append(oneHot, pad)
	}

	for i := 0; i+1+llm.CtxSize <= len(oneHot); i++ {
		answer := llm.Forward(oneHot[i : i+llm.CtxSize])
		zap.S().Infof("%s CrossEntropy -> %.4f", fileName, answer.CrossEntropy(oneHot[i+1:i+1+llm.CtxSize]))
		llm.Backward(answer.Sub(oneHot[i+1:i+1+llm.CtxSize]), lrate)
	}
}

// Forward x матрица one-hot
func (llm *LLM) Forward(x mat.Mat) mat.Mat {
	llm.x = x

	embs := x.Mul(llm.Embs).Add(llm.Pos)

	for _, layer := range llm.Layers {
		embs = layer.Forward(embs)
	}

	llm.embs = embs

	return embs.Mul(llm.Embs.T()).Softmax()
}

func (llm *LLM) Backward(do mat.Mat, lrate float64) mat.Mat {
	dlay := do.Mul(llm.Embs)

	for i := len(llm.Layers) - 1; i >= 0; i-- {
		dlay = llm.Layers[i].Backward(dlay, lrate)
	}

	llm.Pos = mlutil.Upd(llm.Pos, dlay, lrate)

	llm.Embs = mlutil.Upd(llm.Embs, llm.x.T().Mul(dlay).
		Add(llm.embs.T().Mul(do).T()), lrate)

	return nil
}

func (llm *LLM) Save(to string) {
	file, err := os.Create(to)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	for _, lay := range llm.Layers {
		lay.MHA.Mask = nil
	}

	err = json.
		NewEncoder(file).
		Encode(llm)
	if err != nil {
		panic(err)
	}
}

func (llm *LLM) Query(query string) {
	marks := llm.Dict.Mark(llm.Dict.Tokenize(query))
	oneHot := mat.New(len(marks), llm.Embs.RowN())
	oneHot.OneHot(marks)

	pad := make([]float64, llm.Embs.RowN())
	pad[llm.Dict.PadPos] = 1

	var padn int
	for oneHot.RowN() < llm.CtxSize {
		oneHot = append(oneHot, pad)
		padn++
	}

	for i := 0; i < 128; {
		ans := llm.Forward(oneHot[i : i+llm.CtxSize])

		_, index := mat.Mat{ans[len(ans)-padn-1]}.MaxIndex()

		newi := make([]float64, llm.Embs.RowN())
		newi[index] = 1

		if padn != 0 {
			oneHot[len(oneHot)-padn] = newi
			padn--
		} else {
			oneHot = append(oneHot, newi)
			i++
		}

		if index == llm.Dict.PadPos {
			continue
		}
		if strings.HasSuffix(llm.Dict.Dict[index], "</w>") {
			fmt.Print(llm.Dict.Dict[index][:len(llm.Dict.Dict[index])-len("</w>")], " ")
			continue
		}
		if llm.Dict.Dict[index] == "</bl>" {
			fmt.Println()
			continue
		}
		fmt.Print(llm.Dict.Dict[index])
	}
}

func Load(src string, xrown int) (*LLM, error) {
	file, err := os.Open(src)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var llm LLM

	err = json.
		NewDecoder(file).
		Decode(&llm)
	if err != nil {
		return nil, err
	}

	for _, layer := range llm.Layers {
		mask := mat.New(xrown, xrown)
		for row := range mask {
			for col := range mask[row] {
				if row < col {
					mask[row][col] = math.Inf(-1)
				}
			}
		}
		layer.MHA.Mask = mask
	}

	for i := range llm.Embs[llm.Dict.PadPos] {
		llm.Embs[llm.Dict.PadPos][i] = 0
	}

	return &llm, nil
}
