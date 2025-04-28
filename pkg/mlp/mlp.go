package mlp

import (
	"ml/pkg/mat"
	"ml/pkg/mlutil"
)

type Layer struct {
	Weight mat.Mat `json:"weight"`
	Bias   mat.Mat `json:"bias"`
	x      mat.Mat
	ans    mat.Mat
}

func (l *Layer) Forward(x mat.Mat) mat.Mat {
	l.x = x
	l.ans = l.x.Mul(l.Weight).Add(l.Bias)
	return l.ans
}

func (l *Layer) Backward(dans mat.Mat) (dx, dweight, dbias mat.Mat) {
	return dans.Mul(l.Weight.T()),
		l.x.T().Mul(dans),
		dans
}

func (l *Layer) BackwardMut(dans mat.Mat, lrate float64) mat.Mat {
	dx := dans.Mul(l.Weight.T())
	dweight := l.x.T().Mul(dans)
	dbias := dans

	l.Weight = mlutil.Upd(l.Weight, dweight, lrate)
	l.Bias = mlutil.Upd(l.Bias, dbias, lrate)

	return dx
}

func (l *Layer) Update(dweight, dbias mat.Mat, lrate float64) {
	l.Weight = l.Weight.
		Sub(dweight.Scale(lrate))

	l.Bias = l.Bias.
		Sub(dbias.Scale(lrate))
}

func NewLayer(xrown, xcoln, wcoln int) *Layer {
	return &Layer{
		Weight: mat.New(xcoln, wcoln).Rand(),
		Bias:   mat.New(xrown, wcoln),
	}
}

type MLP struct {
	Lays  []*Layer `json:"lays"`
	Alpha float64  `json:"alpha"`
}

func (mlp *MLP) Forward(x mat.Mat) mat.Mat {
	for i, l := range mlp.Lays {
		if i != 0 {
			x = x.LeakyReLU(mlp.Alpha)
		}

		x = l.Forward(x)
	}

	return x
}

type DLayer struct {
	Weight mat.Mat `json:"weight"`
	Bias   mat.Mat `json:"bias"`
}

func (mlp *MLP) Backward(dans mat.Mat) []DLayer {
	dlays := make([]DLayer, len(mlp.Lays))

	for i := len(mlp.Lays) - 1; i >= 0; i-- {
		var dweight, dbias mat.Mat

		dans, dweight, dbias = mlp.Lays[i].Backward(dans)

		dlays[i] = DLayer{Weight: dweight, Bias: dbias}

		if i != 0 {
			dans = mlp.Lays[i-1].ans.
				LeakyReLUDer(mlp.Alpha).
				MulElwise(dans)
		}
	}

	return dlays
}

func (mlp *MLP) BackwardMut(dans mat.Mat, lrate float64) mat.Mat {
	for i := len(mlp.Lays) - 1; i >= 0; i-- {
		dans = mlp.Lays[i].BackwardMut(dans, lrate)

		if i != 0 {
			dans = mlp.Lays[i-1].ans.
				LeakyReLUDer(mlp.Alpha).
				MulElwise(dans)
		}
	}

	return dans
}

func (mlp *MLP) Update(dlays []DLayer, lrate float64) {
	for i, dlay := range dlays {
		mlp.Lays[i].Update(dlay.Weight, dlay.Bias, lrate)
	}
}

func New(alpha float64, xrown, xcoln int, wcolns ...int) *MLP {
	mlp := &MLP{Alpha: alpha}

	for _, wcoln := range wcolns {
		mlp.Lays = append(mlp.Lays, NewLayer(xrown, xcoln, wcoln))
		xcoln = wcoln
	}

	return mlp
}
