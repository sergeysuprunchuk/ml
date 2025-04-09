package mat

import (
	"math"
	"reflect"
	"testing"
)

func Test_Mul(t *testing.T) {
	tests := []struct {
		a, b, c Mat
	}{
		{
			a: Mat{
				{-2, -1, -5},
				{6, -5, -8},
			},
			b: Mat{
				{8, 9},
				{1, 1},
				{1, -4},
			},
			c: Mat{
				{-22, 1},
				{35, 81},
			},
		},
		{
			a: Mat{
				{-6, -6},
				{-7, 6},
				{-8, -5},
				{4, -5},
			},
			b: Mat{
				{1, 9, -7, 1},
				{-5, 5, -1, 10},
			},
			c: Mat{
				{24, -84, 48, -66},
				{-37, -33, 43, 53},
				{17, -97, 61, -58},
				{29, 11, -23, -46},
			},
		},
		{
			a: Mat{
				{-3, -7, -2},
				{4, 5, 10},
			},
			b: Mat{
				{6, -7, -5, 5},
				{-8, 6, -7, -9},
				{-9, 10, -2, -10},
			},
			c: Mat{
				{56, -41, 68, 68},
				{-106, 102, -75, -125},
			},
		},
		{
			a: Mat{
				{-9, 5, -2},
				{4, -2, 8},
			},
			b: Mat{
				{-8, -4, 10},
				{-4, 1, 3},
				{-1, -8, -9},
			},
			c: Mat{
				{54, 57, -57},
				{-32, -82, -38},
			},
		},
		{
			a: Mat{
				{1, -10},
				{0, 8},
				{-2, -7},
			},
			b: Mat{
				{-8, 2, -6, 2},
				{-4, -4, -2, -9},
			},
			c: Mat{
				{32, 42, 14, 92},
				{-32, -32, -16, -72},
				{44, 24, 26, 59},
			},
		},
		{
			a: Mat{
				{4, 0, 0},
			},
			b: Mat{
				{1, 4},
				{3, 2},
				{0, 0},
			},
			c: Mat{
				{4, 16},
			},
		},
	}

	for i, test := range tests {
		if !reflect.DeepEqual(test.a.Mul(test.b), test.c) {
			t.Errorf("%d: %v * %v != %v", i+1, test.a, test.b, test.c)
		}
	}
}

func Test_Add(t *testing.T) {
	tests := []struct {
		a, b, c Mat
	}{
		{
			a: Mat{
				{-17, -16, -15, 15},
				{5, -17, -12, 11},
				{-18, -7, 6, -17},
			},
			b: Mat{
				{-11, -5, -10, 10},
				{-11, -13, -13, 5},
				{-19, 16, -12, -14},
			},
			c: Mat{
				{-28, -21, -25, 25},
				{-6, -30, -25, 16},
				{-37, 9, -6, -31},
			},
		},
		{
			a: Mat{
				{-18, 11, 13},
				{0, 14, -18},
				{1, 14, -15},
			},
			b: Mat{
				{15, 17, 1},
				{10, 15, 4},
				{-10, 11, -16},
			},
			c: Mat{
				{-3, 28, 14},
				{10, 29, -14},
				{-9, 25, -31},
			},
		},
	}

	for i, test := range tests {
		if !reflect.DeepEqual(test.a.Add(test.b), test.c) {
			t.Errorf("%d: %v + %v != %v", i+1, test.a, test.b, test.c)
		}
	}
}

func Test_Sub(t *testing.T) {
	tests := []struct {
		a, b, c Mat
	}{
		{
			a: Mat{
				{-13, 18, 11, -7},
				{-7, -16, -5, -10},
				{-19, 9, -5, 7},
			},
			b: Mat{
				{-5, 14, 20, 15},
				{3, 5, -5, 12},
				{6, 1, -13, 10},
			},
			c: Mat{
				{-8, 4, -9, -22},
				{-10, -21, 0, -22},
				{-25, 8, 8, -3},
			},
		},
	}

	for i, test := range tests {
		if !reflect.DeepEqual(test.a.Sub(test.b), test.c) {
			t.Errorf("%d: %v - %v != %v", i+1, test.a, test.b, test.c)
		}
	}
}

func Test_Scale(t *testing.T) {
	tests := []struct {
		m, ans Mat
		n      float64
	}{
		{
			m: Mat{
				{4, 2},
				{9, 0},
			},
			n: 5,
			ans: Mat{
				{20, 10},
				{45, 0},
			},
		},
		{
			m: Mat{
				{10, 20},
				{60, 80},
			},
			n: 1.0 / 2.0,
			ans: Mat{
				{5, 10},
				{30, 40},
			},
		},
	}

	for i, test := range tests {
		if !reflect.DeepEqual(test.m.Scale(test.n), test.ans) {
			t.Errorf("%d: %v * %f != %v", i+1, test.m, test.n, test.ans)
		}
	}
}

func Test_MulElwise(t *testing.T) {
	tests := []struct {
		a, b, c Mat
	}{
		{
			a: Mat{
				{5, 10, 30},
				{10, 10, 20},
			},
			b: Mat{
				{1, 5, 10},
				{10, 1, 5},
			},
			c: Mat{
				{5, 50, 300},
				{100, 10, 100},
			},
		},
		{
			a: Mat{
				{5, 10, 30},
				{10, 10, 20},
			},
			b: Mat{
				{0, 0, 0},
				{10, 1, 5},
			},
			c: Mat{
				{0, 0, 0},
				{100, 10, 100},
			},
		},
	}

	for i, test := range tests {
		if !reflect.DeepEqual(test.a.MulElwise(test.b), test.c) {
			t.Errorf("%d: %v @ %v != %v", i+1, test.a, test.b, test.c)
		}
	}
}

func Test_T(t *testing.T) {
	tests := []struct {
		m, ans Mat
	}{
		{
			m: Mat{
				{2, 1},
				{-3, 0},
				{4, -1},
			},
			ans: Mat{
				{2, -3, 4},
				{1, 0, -1},
			},
		},
		{
			m: Mat{
				{2, -3, 4},
				{1, 0, -1},
			},
			ans: Mat{
				{2, 1},
				{-3, 0},
				{4, -1},
			},
		},
	}

	for i, test := range tests {
		if !reflect.DeepEqual(test.m.T(), test.ans) {
			t.Errorf("%d: %vT != %v", i+1, test.m, test.ans)
		}
	}
}

func Test_LeakyReLU(t *testing.T) {
	const alpha float64 = .01

	tests := []struct {
		m, ans Mat
	}{
		{
			m: Mat{
				{-.1, .1, 1, 0, -0},
			},
			ans: Mat{
				{-.1 * alpha, .1, 1, 0, 0},
			},
		},
	}

	for i, test := range tests {
		relu := test.m.LeakyReLU(alpha)
		if !reflect.DeepEqual(relu, test.ans) {
			t.Errorf("%d: %v != %v", i+1, relu, test.ans)
		}
	}
}

func Test_LeakyReLUDer(t *testing.T) {
	const alpha float64 = .01

	tests := []struct {
		m, ans Mat
	}{
		{
			m: Mat{
				{-.1, .1, 1, 0, -0},
			},
			ans: Mat{
				{alpha, 1, 1, 1, 1},
			},
		},
	}

	for i, test := range tests {
		der := test.m.LeakyReLUDer(alpha)
		if !reflect.DeepEqual(der, test.ans) {
			t.Errorf("%d: %v != %v", i+1, der, test.ans)
		}
	}
}

func Test_Softmax(t *testing.T) {
	tests := []struct {
		m, ans Mat
	}{
		{
			m: Mat{
				{.15, .75, .8},
			},
			ans: Mat{
				{21, 38, 40},
			},
		},
		{
			m: Mat{
				{3.15, 4.70, .30},
			},
			ans: Mat{
				{17, 82, 1},
			},
		},
		{
			m: Mat{
				{3.15, 4.70, .30},
				{.15, .75, .8},
			},
			ans: Mat{
				{17, 82, 1},
				{21, 38, 40},
			},
		},
		{
			m: Mat{
				{1_000, 1_050, 930},
			},
			ans: Mat{
				{0, 100, 0},
			},
		},
		{
			m: Mat{
				{-1_000, -1_050, 930},
			},
			ans: Mat{
				{0, 0, 100},
			},
		},
	}

	for i, test := range tests {
		probs := test.m.Softmax()
		for row := range probs {
			for col := range probs[row] {
				probs[row][col] = math.Round(probs[row][col] * 100)
			}
		}

		if !reflect.DeepEqual(probs, test.ans) {
			t.Errorf("%d: %v != %v", i+1, probs, test.ans)
		}
	}
}

func Test_CrossEntropy(t *testing.T) {
	tests := []struct {
		m, truth Mat
		ans      float64
	}{
		{
			m:     Mat{{.75, .15, .10}},
			truth: Mat{{1, 0, 0}},
			ans:   0.2876820724517809,
		},
		{
			m:     Mat{{.75, .15, .10}},
			truth: Mat{{0, 1, 0}},
			ans:   1.8971199848858813,
		},
		{
			m:     Mat{{.75, .15, .10}},
			truth: Mat{{0, 0, 1}},
			ans:   2.3025850929940455,
		},
		{
			m:     Mat{{0, 0, 1}},
			truth: Mat{{0, 0, 1}},
			ans:   0,
		},
		{
			m:     Mat{{.01, .01, .98}},
			truth: Mat{{0, 1, 0}},
			ans:   4.605170185988092,
		},
		{
			m: Mat{
				{.65, .3, .05},
				{.25, .25, .5},
			},
			truth: Mat{
				{0, 1, 0},
				{0, 0, 1},
			},
			ans: 0.9485599924429406,
		},
	}

	for i, test := range tests {
		ce := test.m.CrossEntropy(test.truth)
		if math.Abs(ce-test.ans) > 1e-6 {
			t.Errorf("%d: %f != %f", i+1, ce, test.ans)
		}
	}
}
