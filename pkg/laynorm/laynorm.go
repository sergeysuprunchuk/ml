package laynorm

import (
	"math"
	"ml/pkg/mat"
	"ml/pkg/mlutil"
)

type LayNorm struct {
	Gamma mat.Mat `json:"gamma"`
	Beta  mat.Mat `json:"beta"`

	x, xhat, mean, variance mat.Mat
}

func (ln *LayNorm) Forward(x mat.Mat) mat.Mat {
	ln.x = x

	const eps = 1e-6

	ln.mean = x.Mean()
	ln.variance = x.Var(ln.mean)

	ln.xhat = mat.New(x.RowN(), x.ColN())
	for row := range x {
		for col := range x[row] {
			ln.xhat[row][col] =
				(x[row][col] - ln.mean[row][0]) /
					math.Sqrt(ln.variance[row][0]+eps)
		}
	}

	xnorm := mat.New(x.RowN(), x.ColN())
	for row := range ln.xhat {
		xnorm[row] = mat.Mat{ln.xhat[row]}.
			MulElwise(ln.Gamma).
			Add(ln.Beta)[0]
	}

	return xnorm
}

func (ln *LayNorm) Backward(do mat.Mat, lrate float64) mat.Mat {
	const eps = 1e-6
	mean := do.Sub1(do.Mean())
	dx := mat.New(do.RowN(), do.ColN())
	for row := range dx.RowN() {
		dx[row] = ln.Gamma.Scale(1 / math.Sqrt(ln.variance[row][0]+eps)).
			MulElwise(mat.Mat{mean[row]})[0]
	}

	ln.Gamma = mlutil.Upd(ln.Gamma, ln.xhat.MulElwise(do).ColSum(), lrate)
	ln.Beta = mlutil.Upd(ln.Beta, do.ColSum(), lrate)

	return dx
}

func New(xcoln int) *LayNorm {
	gamma := mat.New(1, xcoln)
	for col := range gamma[0] {
		gamma[0][col] = 1
	}

	return &LayNorm{
		Gamma: gamma,
		Beta:  mat.New(1, xcoln),
	}
}
