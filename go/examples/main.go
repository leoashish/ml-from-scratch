package main

import (
	"fmt"
	"math"
	"math/rand"

	"mlfromscratch/anomaly"
	"mlfromscratch/classification"
	"mlfromscratch/clustering"
	"mlfromscratch/dimred"
	"mlfromscratch/forecasting"
	"mlfromscratch/generative"
	"mlfromscratch/mat"
	"mlfromscratch/regression"
	"mlfromscratch/rl"
)

func main() {
	rng := rand.New(rand.NewSource(0))
	X := mat.NewMatrix(200, 3)
	y := make(mat.Vec, 200)
	for i := 0; i < 200; i++ {
		for j := 0; j < 3; j++ {
			X[i][j] = rng.NormFloat64()
		}
		y[i] = 2*X[i][0] - 1*X[i][1] + 0.5*X[i][2] + rng.NormFloat64()*0.5
	}
	lr := regression.NewLinearRegression("normal", 0, 0, 0, true)
	lr.Fit(X, y)
	pred := lr.Predict(X)
	fmt.Println("LR sample pred:", pred[0])

	Xc := mat.NewMatrix(200, 2)
	yc := make(mat.Vec, 200)
	for i := 0; i < 100; i++ {
		Xc[i][0] = rng.NormFloat64() - 1.5
		Xc[i][1] = rng.NormFloat64() - 1.5
		yc[i] = 0
	}
	for i := 100; i < 200; i++ {
		Xc[i][0] = rng.NormFloat64() + 1.5
		Xc[i][1] = rng.NormFloat64() + 1.5
		yc[i] = 1
	}
	logr := classification.NewLogisticRegression(0.1, 300, 0, true)
	logr.Fit(Xc, yc)
	fmt.Println("Logistic done")

	Xb := mat.NewMatrix(150, 2)
	for i := 0; i < 150; i++ {
		c := i / 50
		Xb[i][0] = float64(c)*5 + rng.NormFloat64()*0.5
		Xb[i][1] = float64(c)*5 + rng.NormFloat64()*0.5
	}
	km := clustering.NewKMeans(3, 100, 1, 0)
	km.Fit(Xb)
	fmt.Println("KMeans centers:", km.Centers)

	pca := dimred.NewPCA(2)
	Z := pca.FitTransform(X)
	fmt.Println("PCA shape:", Z.Rows(), Z.Cols())

	Xa := mat.NewMatrix(100, 2)
	for i := 0; i < 100; i++ {
		Xa[i][0] = rng.NormFloat64()
		Xa[i][1] = rng.NormFloat64()
	}
	iso := anomaly.NewIsolationForest(50, 64, 0.1, 0)
	iso.Fit(Xa)
	fmt.Println("IF done")

	series := make(mat.Vec, 60)
	for i := range series {
		series[i] = float64(i)*0.05 + 5*math.Sin(float64(i))
	}
	ar := forecasting.NewARIMA(2, 1, 1)
	ar.Fit(series)
	fmt.Println("ARIMA forecast:", ar.Forecast(3))

	dm := generative.NewDiffusion(2, 8, 10, 1e-4, 0.02, 1e-3, 20, 16, 0)
	dm.Fit(mat.NewMatrix(50, 2))
	S := dm.Sample(5, 1)
	fmt.Println("Diffusion sample shape:", S.Rows(), S.Cols())

	env := rl.NewGridWorld(4, [2]int{3, 3}, [2]int{1, 1})
	ag := rl.NewQLearning(env.Size*env.Size, 4, 0.1, 0.99, 1.0, 0.995, 0.01, 0)
	_ = ag.Train(env, 50, 50)
	fmt.Println("Q-learning done")

	env2 := rl.NewCartPole()
	ppo := rl.NewPPO(4, 2, 16, 0, 0.99, 0.95, 0.2, 1, 64, 0)
	rh := ppo.Train(env2, 64, 32)
	fmt.Println("PPO rewards:", rh)
	fmt.Println("All Go demos ran.")
}
