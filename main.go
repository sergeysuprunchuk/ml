package main

import (
	"fmt"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"ml/pkg/llm"
)

const (
	layerN  = 1
	ctxSize = 256
	embSize = 192
	wcoln   = 48
	headN   = 4
	alpha   = .01
	dictSrc = "C:\\Users\\sergey\\Desktop\\ml\\llm\\data"
)

func main() {
	cfg := zap.NewDevelopmentConfig()
	cfg.EncoderConfig.EncodeLevel = zapcore.CapitalColorLevelEncoder
	cfg.DisableStacktrace = true
	zap.ReplaceGlobals(zap.Must(cfg.Build()))

	//b := bpe.New()

	//err := b.Learn2(jokes.Jokes, 2048)
	//if err != nil {
	//	panic(err)
	//}

	//err = b.Save(dictSrc)
	//if err != nil {
	//	panic(err)
	//}

	//LLM := llm.New(
	//	layerN,
	//	ctxSize,
	//	embSize,
	//	wcoln,
	//	headN,
	//	alpha,
	//	dictSrc+"\\tokens.json",
	//)

	LLM, err := llm.Load("C:\\Users\\sergey\\Desktop\\ml\\llm\\data\\llm", ctxSize)
	if err != nil {
		panic(err)
	}

	query := "попал под машину"
	fmt.Printf("запрос: %s\nответ: ", query)
	LLM.Query(query)

	//for epoch := range 4 {
	//	for i, joke := range jokes.Jokes {
	//		LLM.Learn(joke, .0001, fmt.Sprintf("%d %d", epoch, i))
	//	}
	//}

	//LLM.Save("C:\\Users\\sergey\\Desktop\\ml\\llm\\data\\llm")
}
