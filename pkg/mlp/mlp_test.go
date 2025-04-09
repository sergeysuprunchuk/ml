package mlp

import (
	"ml/pkg/mat"
	"reflect"
	"testing"
)

func Test_NewLayer(t *testing.T) {
	tests := []struct {
		xrown, xcoln, wcoln int
	}{
		{
			xrown: 1,
			xcoln: 10,
			wcoln: 20,
		},
		{
			xrown: 20,
			xcoln: 100,
			wcoln: 200,
		},
		{
			xrown: 1,
			xcoln: 1,
			wcoln: 1,
		},
	}

	for i, test := range tests {
		l := NewLayer(test.xrown, test.xcoln, test.wcoln)

		if l.Bias.RowN() != test.xrown || l.Bias.ColN() != test.wcoln {
			t.Errorf("%d: bias %dx%d, правильный ответ %dx%d",
				i+1, l.Bias.RowN(), l.Bias.ColN(), test.xrown, test.wcoln)
		}

		if l.Weight.RowN() != test.xcoln || l.Weight.ColN() != test.wcoln {
			t.Errorf("%d: weight %dx%d, правильный ответ %dx%d",
				i+1, l.Weight.RowN(), l.Weight.ColN(), test.xcoln, test.wcoln)
		}
	}
}

func Test_Layer_Forward(t *testing.T) {
	tests := []struct {
		l      *Layer
		x, ans mat.Mat
	}{
		{
			x: mat.Mat{{.5, 1, 3}},
			l: &Layer{
				Weight: mat.Mat{
					{5, 3},
					{.5, 1},
					{10, 5},
				},
				Bias: mat.Mat{{0, 0}},
			},
			ans: mat.Mat{
				{33, 17.5},
			},
		},
		{
			x: mat.Mat{{.5, 1, 3}},
			l: &Layer{
				Weight: mat.Mat{
					{5, 3},
					{.5, 1},
					{10, 5},
				},
				Bias: mat.Mat{{5, 3}},
			},
			ans: mat.Mat{
				{38, 20.5},
			},
		},
		{
			x: mat.Mat{{.5, 1, 0}},
			l: &Layer{
				Weight: mat.Mat{
					{0, 3},
					{.5, 1},
					{10, 0},
				},
				Bias: mat.Mat{{0, 0}},
			},
			ans: mat.Mat{
				{0.5, 2.5},
			},
		},
		{
			x: mat.Mat{{.5, 1, 0}},
			l: &Layer{
				Weight: mat.Mat{
					{0, 3},
					{.5, 1},
					{10, 0},
				},
				Bias: mat.Mat{{.5, -.3}},
			},
			ans: mat.Mat{
				{1, 2.2},
			},
		},
		{
			x: mat.Mat{{.5, 1, 0}},
			l: &Layer{
				Weight: mat.Mat{
					{0, 3},
					{.5, -1},
					{10, 0},
				},
				Bias: mat.Mat{{.5, -.3}},
			},
			ans: mat.Mat{
				{1, 0.2},
			},
		},
	}

	for i, test := range tests {
		ans := test.l.Forward(test.x)
		if !reflect.DeepEqual(ans, test.ans) {
			t.Errorf("%d: %v != %v", i+1, ans, test.ans)
		}
	}
}

func Test_Layer_Backward(t *testing.T) {
}

func Test_Layer_Update(t *testing.T) {
}
