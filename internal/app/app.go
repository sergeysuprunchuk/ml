package app

import (
	"encoding/json"
	"github.com/go-chi/chi/v5"
	"github.com/go-chi/cors"
	"ml/internal/nets/num"
	"net/http"
	"slices"
)

func Run() error {
	n, err := num.Load("./internal/nets/num/data/data")
	if err != nil {
		return err
	}

	r := chi.NewRouter()

	r.Use(cors.AllowAll().Handler)

	r.Post("/num", func(w http.ResponseWriter, r *http.Request) {
		file, _, err := r.FormFile("num")
		if err != nil {
			panic(err)
		}
		defer file.Close()

		ans := n.Query(file)

		type prob struct {
			Num int     `json:"num"`
			Val float64 `json:"val"`
		}

		probs := make([]prob, 0, len(ans[0]))

		for i := range ans[0] {
			probs = append(probs, prob{
				Num: i,
				Val: ans[0][i],
			})
		}

		slices.SortFunc(probs, func(a, b prob) int {
			if b.Val < a.Val {
				return -1
			}
			return 1
		})

		err = json.
			NewEncoder(w).
			Encode(probs)
		if err != nil {
			panic(err)
		}
	})

	return http.ListenAndServe(":8080", r)
}
