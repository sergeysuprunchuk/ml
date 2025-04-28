package attention

import (
	"math"
	"ml/pkg/mat"
	"ml/pkg/mlutil"
)

type Head struct {
	Q                mat.Mat `json:"q"`
	K                mat.Mat `json:"k"`
	V                mat.Mat `json:"v"`
	KLenSqrt         float64 `json:"kLenSqrt"`
	x, a, xQ, xK, xV mat.Mat
}

func NewHead(wrown, wcoln int) *Head {
	return &Head{
		Q:        mat.New(wrown, wcoln).Rand(),
		K:        mat.New(wrown, wcoln).Rand(),
		V:        mat.New(wrown, wcoln).Rand(),
		KLenSqrt: math.Sqrt(float64(wcoln)),
	}
}

func (h *Head) Forward(x, mask mat.Mat) mat.Mat {
	h.x = x
	h.xQ, h.xK, h.xV = h.x.Mul(h.Q), h.x.Mul(h.K), h.x.Mul(h.V)
	s := h.xQ.Mul(h.xK.T()).Scale(1 / h.KLenSqrt).Add(mask)
	h.a = s.Softmax()
	return h.a.Mul(h.xV)
}

func (h *Head) Backward(do mat.Mat, lrate float64) mat.Mat {
	da := do.Mul(h.xV.T())
	sum := h.a.MulElwise(da).RowSum()
	ds := h.a.MulElwise(da.Sub1(sum))
	dxQ := ds.Mul(h.xK.Scale(1 / h.KLenSqrt))
	dxK := ds.T().Mul(h.xQ.Scale(1 / h.KLenSqrt))
	dxV := h.a.T().Mul(do)

	xT := h.x.T()
	dx := dxQ.Mul(h.Q.T()).Add(dxK.Mul(h.K.T())).Add(dxV.Mul(h.V.T()))

	h.Q = mlutil.Upd(h.Q, xT.Mul(dxQ), lrate)
	h.K = mlutil.Upd(h.K, xT.Mul(dxK), lrate)
	h.V = mlutil.Upd(h.V, xT.Mul(dxV), lrate)

	return dx
}

type MultiHead struct {
	Heads []*Head `json:"heads"`
	Mask  mat.Mat `json:"mask"`
	Out   mat.Mat `json:"out"`
	matsc mat.Mat
}

func NewMultiHead(xrown, xcoln, wcoln, h int) *MultiHead {
	heads := make([]*Head, h)
	for i := range h {
		heads[i] = NewHead(xcoln, wcoln)
	}

	mask := mat.New(xrown, xrown)
	for row := range mask {
		for col := range mask[row] {
			if row < col {
				mask[row][col] = math.Inf(-1)
			}
		}
	}

	return &MultiHead{
		Heads: heads,
		Mask:  mask,
		Out:   mat.New(wcoln*h, xcoln),
	}
}

func (mh *MultiHead) Forward(x mat.Mat) mat.Mat {
	matrices := make([]mat.Mat, 0, len(mh.Heads))
	for _, h := range mh.Heads {
		matrices = append(matrices, h.Forward(x, mh.Mask))
	}
	mh.matsc = mat.Concat(matrices...)
	return mh.matsc.Mul(mh.Out)
}

func (mh *MultiHead) Backward(do mat.Mat, lrate float64) mat.Mat {
	mh.Out = mlutil.Upd(mh.Out, mh.matsc.T().Mul(do), lrate)

	ders := mat.Split(
		do.Mul(mh.Out.T()),
		len(mh.Heads),
	)

	var dx mat.Mat
	for i, der := range ders {
		d := mh.Heads[i].Backward(der, lrate)
		if dx == nil {
			dx = d
			continue
		}
		dx = dx.Add(d)
	}

	return dx
}
