package main

import (
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"ml/internal/app"
)

func main() {
	cfg := zap.NewDevelopmentConfig()
	cfg.EncoderConfig.EncodeLevel = zapcore.CapitalColorLevelEncoder
	cfg.DisableStacktrace = true
	zap.ReplaceGlobals(zap.Must(cfg.Build()))

	//n, err := num.Load("./internal/nets/num/data/data")
	//if err != nil {
	//	panic(err)
	//}
	//
	//n.Learn("C:\\Users\\sergey\\Desktop\\llm\\dataset", 10, 64, 1)
	//
	//err = n.Save("./internal/nets/num/data/dataultra")
	//if err != nil {
	//	panic(err)
	//}
	//
	//n.Test("C:\\Users\\sergey\\Desktop\\llm\\dataset")

	err := app.Run()
	if err != nil {
		panic(err)
	}
}
