package com.aimath.core;

/**
 * 경사하강법 구현
 * AI 모델 학습의 핵심 알고리즘
 */
public class GradientDescent {

    /**
     * 간단한 1차원 경사하강법
     * f(x)의 최솟값을 찾는다
     *
     * @param gradient 그래디언트 함수 (x -> df/dx)
     * @param initialX 시작점
     * @param learningRate 학습률
     * @param iterations 반복 횟수
     * @return 최솟값 위치
     */
    public static double minimize1D(
            java.util.function.DoubleUnaryOperator gradient,
            double initialX,
            double learningRate,
            int iterations
    ) {
        double x = initialX;
        for (int i = 0; i < iterations; i++) {
            double grad = gradient.applyAsDouble(x);
            x = x - learningRate * grad;
        }
        return x;
    }

    /**
     * 다차원 경사하강법
     * f(x1, x2, ..., xn)의 최솟값을 찾는다
     *
     * @param gradient 그래디언트 함수 (x[] -> ∇f)
     * @param initialX 시작점
     * @param learningRate 학습률
     * @param iterations 반복 횟수
     * @return 최솟값 위치
     */
    public static double[] minimize(
            java.util.function.Function<double[], double[]> gradient,
            double[] initialX,
            double learningRate,
            int iterations
    ) {
        double[] x = initialX.clone();
        for (int i = 0; i < iterations; i++) {
            double[] grad = gradient.apply(x);
            for (int j = 0; j < x.length; j++) {
                x[j] = x[j] - learningRate * grad[j];
            }
        }
        return x;
    }

    /**
     * SGD with Momentum
     * 관성을 추가하여 진동을 줄이고 수렴 속도 향상
     */
    public static double[] minimizeWithMomentum(
            java.util.function.Function<double[], double[]> gradient,
            double[] initialX,
            double learningRate,
            double momentum,
            int iterations
    ) {
        double[] x = initialX.clone();
        double[] velocity = new double[x.length];

        for (int i = 0; i < iterations; i++) {
            double[] grad = gradient.apply(x);
            for (int j = 0; j < x.length; j++) {
                velocity[j] = momentum * velocity[j] - learningRate * grad[j];
                x[j] = x[j] + velocity[j];
            }
        }
        return x;
    }

    /**
     * 경사하강법 시각화용 히스토리 기록
     */
    public static class TrainingHistory {
        public double[] losses;
        public double[][] positions;

        public TrainingHistory(int iterations, int dimensions) {
            losses = new double[iterations];
            positions = new double[iterations][dimensions];
        }
    }

    /**
     * 히스토리 기록과 함께 경사하강법 실행
     */
    public static TrainingHistory minimizeWithHistory(
            java.util.function.Function<double[], Double> lossFunction,
            java.util.function.Function<double[], double[]> gradient,
            double[] initialX,
            double learningRate,
            int iterations
    ) {
        TrainingHistory history = new TrainingHistory(iterations, initialX.length);
        double[] x = initialX.clone();

        for (int i = 0; i < iterations; i++) {
            history.losses[i] = lossFunction.apply(x);
            history.positions[i] = x.clone();

            double[] grad = gradient.apply(x);
            for (int j = 0; j < x.length; j++) {
                x[j] = x[j] - learningRate * grad[j];
            }
        }
        return history;
    }
}
