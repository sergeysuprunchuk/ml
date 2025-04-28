package attention

import (
	"math"
	"ml/pkg/mat"
	"reflect"
	"testing"
)

func Test_Head_Forward(t *testing.T) {
	tests := []struct {
		x, m, ans mat.Mat
		h         *Head
	}{
		{
			x: mat.Mat{
				{.3, .5, -.1},
				{1, 0, .4}},
			h: &Head{
				Q: mat.Mat{
					{.2, 0},
					{-.9, .85},
					{1, .15},
				},
				K: mat.Mat{
					{.2, 0},
					{-.9, .85},
					{1, .15},
				},
				V: mat.Mat{
					{.2, 0},
					{-.9, .85},
					{1, .15},
				},
				KLenSqrt: math.Sqrt(2.),
			},

			m: mat.Mat{
				{0, 0},
				{0, 0},
			},

			ans: mat.Mat{
				{-0.07312266339770607, 0.2761403047607313},
				{0.17497425770462144, 0.19647615578291971},
			},
		},
	}

	for i, test := range tests {
		ans := test.h.Forward(test.x, test.m)
		if !reflect.DeepEqual(ans, test.ans) {
			t.Errorf("%d: %v != %v", i+1, ans, test.ans)
		}
	}
}
