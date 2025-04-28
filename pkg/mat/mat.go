package mat

import (
	"math"
	"math/rand/v2"
	"slices"
)

type Mat [][]float64

func (m Mat) RowN() int {
	return len(m)
}

func (m Mat) ColN() int {
	if m.RowN() == 0 {
		return 0
	}
	return len(m[0])
}

func (m Mat) Mul(b Mat) Mat {
	if m.ColN() != b.RowN() {
		panic("Mul: m.ColN() != b.RowN()")
	}

	c := New(m.RowN(), b.ColN())

	for row := range c {
		for col := range c[row] {
			for mcol := range m[row] {
				c[row][col] += m[row][mcol] * b[mcol][col]
			}
		}
	}

	return c
}

func (m Mat) Add(b Mat) Mat {
	if m.RowN() != b.RowN() || m.ColN() != b.ColN() {
		panic("Add: m.RowN() != b.RowN() || m.ColN() != b.ColN()")
	}

	c := New(m.RowN(), m.ColN())

	for row := range c {
		for col := range c[row] {
			c[row][col] = m[row][col] + b[row][col]
		}
	}

	return c
}

func (m Mat) Sub(b Mat) Mat {
	if m.RowN() != b.RowN() || m.ColN() != b.ColN() {
		panic("Sub: m.RowN() != b.RowN() || m.ColN() != b.ColN()")
	}

	c := New(m.RowN(), m.ColN())

	for row := range c {
		for col := range c[row] {
			c[row][col] = m[row][col] - b[row][col]
		}
	}

	return c
}

func (m Mat) Scale(n float64) Mat {
	mat := New(m.RowN(), m.ColN())

	for row := range mat {
		for col := range mat[row] {
			mat[row][col] = m[row][col] * n
		}
	}

	return mat
}

func (m Mat) MulElwise(b Mat) Mat {
	if m.RowN() != b.RowN() || m.ColN() != b.ColN() {
		panic("MulElwise: m.RowN() != b.RowN() || m.ColN() != b.ColN()")
	}

	c := New(m.RowN(), m.ColN())

	for row := range c {
		for col := range c[row] {
			c[row][col] = m[row][col] * b[row][col]
		}
	}

	return c
}

func (m Mat) T() Mat {
	mat := New(m.ColN(), m.RowN())

	for row := range m {
		for col := range m[row] {
			mat[col][row] = m[row][col]
		}
	}

	return mat
}

func (m Mat) LeakyReLU(alpha float64) Mat {
	mat := New(m.RowN(), m.ColN())

	for row := range m {
		for col := range m[row] {
			mat[row][col] = max(m[row][col]*alpha, m[row][col])
		}
	}

	return mat
}

func (m Mat) LeakyReLUDer(alpha float64) Mat {
	mat := New(m.RowN(), m.ColN())

	for row := range m {
		for col := range m[row] {
			if m[row][col] >= 0 {
				mat[row][col] = 1
				continue
			}
			mat[row][col] = alpha
		}
	}

	return mat
}

func (m Mat) Softmax() Mat {
	mat := New(m.RowN(), m.ColN())

	for row := range m {
		var sum float64
		maxn := slices.Max(m[row])

		for col := range m[row] {
			mat[row][col] = math.Exp(m[row][col] - maxn)
			sum += mat[row][col]
		}

		for col := range mat[row] {
			mat[row][col] /= sum
		}
	}

	return mat
}

func (m Mat) CrossEntropy(truth Mat) float64 {
	if m.RowN() != truth.RowN() || m.ColN() != truth.ColN() {
		panic("CrossEntropy: m.RowN() != truth.RowN() || m.ColN() != truth.ColN()")
	}

	const epsilon = 1e-12

	var err float64

	for row := range m {
		for col := range m[row] {
			err += truth[row][col] *
				math.Log(max(epsilon, min(1-epsilon, m[row][col])))
		}
	}

	return -(err / float64(m.RowN()))
}

func (m Mat) Rand() Mat {
	sqrt := math.Sqrt(2. / float64(m.RowN()))

	for row := range m {
		for col := range m[row] {
			m[row][col] = rand.NormFloat64() * sqrt
		}
	}

	return m
}

// Mean вычисляет среднее значение для каждой строки матрицы
func (m Mat) Mean() Mat {
	coln := float64(m.ColN())

	mean := New(m.RowN(), 1)
	for row := range m {
		for col := range m[row] {
			mean[row][0] += m[row][col]
		}
		mean[row][0] /= coln
	}

	return mean
}

// Var вычисляет дисперсию для каждой строки
func (m Mat) Var(mean Mat) Mat {
	coln := float64(m.ColN())

	variance := New(m.RowN(), 1)
	for row := range m {
		for col := range m[row] {
			variance[row][0] += math.Pow(m[row][col]-mean[row][0], 2)
		}
		variance[row][0] /= coln
	}

	return variance
}

func (m Mat) ColSum() Mat {
	mat := New(1, m.ColN())
	for row := range m {
		for col := range m[row] {
			mat[0][col] += m[row][col]
		}
	}
	return mat
}

func (m Mat) RowSum() Mat {
	mat := New(m.RowN(), 1)
	for row := range m {
		for col := range m[row] {
			mat[row][0] += m[row][col]
		}
	}
	return mat
}

func (m Mat) Sub1(b Mat) Mat {
	mat := New(m.RowN(), m.ColN())

	for row := range m {
		for col := range m[row] {
			mat[row][col] = m[row][col] - b[row][0]
		}
	}

	return mat
}

func Concat(matrices ...Mat) Mat {
	if len(matrices) == 0 {
		panic("concat: empty matrices")
	}

	//предполагается, что количество строк в каждой матрице одинаково
	mat := New(matrices[0].RowN(), 0)
	for _, m := range matrices {
		for row := range m {
			mat[row] = append(mat[row], m[row]...)
		}
	}

	return mat
}

func Split(mat Mat, n int) []Mat {
	if mat.RowN() == 0 {
		panic("невозможно разделить матрицу на n равных частей")
	}

	coln := int(math.Ceil(float64(mat.ColN()) / float64(n)))

	mats := make([]Mat, 0, n)
	for range n {
		mats = append(mats, New(mat.RowN(), coln))
	}

	for row := range mat {
		var i int
		for chunk := range slices.Chunk(mat[row], coln) {
			mats[i][row] = chunk
			i++
		}
	}

	return mats
}

func (m Mat) MaxIndex() (int, int) {
	if m.RowN() == 0 || m.ColN() == 0 {
		return 0, 0
	}

	var maxRow, maxCol int
	maxVal := m[maxRow][maxCol]

	for row := range m {
		for col := range m[row] {
			if maxVal < m[row][col] {
				maxRow, maxCol = row, col
				maxVal = m[row][col]
			}
		}
	}

	return maxRow, maxCol
}

func (m Mat) OneHot(labels []int) {
	if m.RowN() != len(labels) {
		panic("m.RowN() != len(labels)")
	}

	for row := range m {
		for col := range m[row] {
			if col == labels[row] {
				m[row][col] = 1
				break
			}
		}
	}
}

func New(rown, coln int) Mat {
	mat := make([][]float64, 0, rown)

	for range rown {
		mat = append(mat, make([]float64, coln))
	}

	return mat
}
