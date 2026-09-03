package com.aimath.chapter6;

import com.aimath.core.GradientDescent;

/**
 * 6장 예제: 경사하강법
 * 기본 GD, 학습률 실험, Momentum, Adam
 */
public class GradientDescentDemo {

    public static void main(String[] args) {
        System.out.println("=== 6장: 경사하강법 ===\n");

        // 1. 기본 경사하강법
        System.out.println("1. 기본 경사하강법");
        basicGradientDescent();
        System.out.println();

        // 2. 학습률의 영향
        System.out.println("2. 학습률(Learning Rate)의 영향");
        learningRateExperiment();
        System.out.println();

        // 3. 2D 경사하강법
        System.out.println("3. 2차원 경사하강법");
        twoDimensionalGD();
        System.out.println();

        // 4. Momentum
        System.out.println("4. Momentum - 관성 추가");
        momentumDemo();
        System.out.println();

        // 5. 손실 감소 시각화
        System.out.println("=== 학습 과정 시각화 ===");
        trainingVisualization();
    }

    private static void basicGradientDescent() {
        System.out.println("목표: f(x) = x² 의 최솟값 찾기");
        System.out.println("도함수: f'(x) = 2x");
        System.out.println("알고리즘: x_new = x - lr × f'(x)");

        double x = 10.0;  // 시작점
        double lr = 0.1;  // 학습률

        System.out.printf("\n시작점: x = %.1f, f(x) = %.1f%n", x, x * x);
        System.out.println("\n반복 | x값    | f(x)   | 그래디언트");
        System.out.println("-----|--------|--------|----------");

        for (int i = 1; i <= 10; i++) {
            double grad = 2 * x;  // f'(x) = 2x
            x = x - lr * grad;
            System.out.printf("%4d | %6.3f | %6.3f | %6.3f%n", i, x, x * x, grad);
        }

        System.out.printf("\n최종: x = %.4f (실제 최솟값: x = 0)%n", x);
    }

    private static void learningRateExperiment() {
        double[] learningRates = {0.01, 0.1, 0.5, 1.0, 1.5};

        System.out.println("f(x) = x² 최적화, 시작점 x = 10");
        System.out.println("\n학습률 | 10회 후 x값 | 수렴 여부");
        System.out.println("-------|------------|----------");

        for (double lr : learningRates) {
            double x = 10.0;

            for (int i = 0; i < 10; i++) {
                double grad = 2 * x;
                x = x - lr * grad;

                // 발산 체크
                if (Double.isNaN(x) || Math.abs(x) > 1e10) {
                    x = Double.POSITIVE_INFINITY;
                    break;
                }
            }

            String status;
            if (Double.isInfinite(x)) {
                status = "발산!";
            } else if (Math.abs(x) < 0.01) {
                status = "수렴";
            } else if (Math.abs(x) < 1) {
                status = "느린 수렴";
            } else {
                status = "진동";
            }

            System.out.printf("%5.2f  | %10.4f | %s%n", lr, x, status);
        }

        System.out.println("\n→ 학습률이 너무 크면 발산, 너무 작으면 느림");
        System.out.println("→ 적절한 학습률 선택이 중요 (보통 0.001 ~ 0.1)");
    }

    private static void twoDimensionalGD() {
        System.out.println("목표: f(x,y) = x² + y² 의 최솟값");
        System.out.println("그래디언트: ∇f = [2x, 2y]");

        double[] point = {5.0, 3.0};  // 시작점
        double lr = 0.1;

        System.out.printf("\n시작점: (%.1f, %.1f), f = %.1f%n",
            point[0], point[1], point[0]*point[0] + point[1]*point[1]);

        // 경사하강법 실행
        java.util.function.Function<double[], double[]> gradient =
            p -> new double[]{2 * p[0], 2 * p[1]};

        double[] result = GradientDescent.minimize(gradient, point, lr, 20);

        System.out.printf("최종점: (%.6f, %.6f)%n", result[0], result[1]);
        System.out.printf("f(최종) = %.8f%n", result[0]*result[0] + result[1]*result[1]);
        System.out.println("→ 원점 (0, 0)에 수렴");
    }

    private static void momentumDemo() {
        System.out.println("Momentum: 이전 업데이트 방향을 기억");
        System.out.println("velocity = β × velocity - lr × gradient");
        System.out.println("x = x + velocity");

        // 좁고 긴 골짜기 형태의 함수 (진동 발생 쉬움)
        // f(x,y) = 10x² + y²
        double[] point = {5.0, 5.0};

        System.out.println("\n일반 GD vs Momentum 비교");
        System.out.println("f(x,y) = 10x² + y² (좁고 긴 골짜기)");

        // 일반 GD
        java.util.function.Function<double[], double[]> gradient =
            p -> new double[]{20 * p[0], 2 * p[1]};

        double[] plainGD = GradientDescent.minimize(gradient, point.clone(), 0.05, 30);

        // Momentum
        double[] withMomentum = GradientDescent.minimizeWithMomentum(
            gradient, point.clone(), 0.05, 0.9, 30);

        System.out.println("\n30회 반복 후:");
        System.out.printf("  일반 GD:  (%.6f, %.6f)%n", plainGD[0], plainGD[1]);
        System.out.printf("  Momentum: (%.6f, %.6f)%n", withMomentum[0], withMomentum[1]);

        double lossPlain = 10*plainGD[0]*plainGD[0] + plainGD[1]*plainGD[1];
        double lossMomentum = 10*withMomentum[0]*withMomentum[0] + withMomentum[1]*withMomentum[1];

        System.out.printf("  일반 GD 손실:  %.8f%n", lossPlain);
        System.out.printf("  Momentum 손실: %.8f%n", lossMomentum);
        System.out.println("\n→ Momentum이 더 빨리 수렴 (진동 감소)");
    }

    private static void trainingVisualization() {
        System.out.println("f(x) = (x-3)² 최적화 과정");

        // 손실 함수: (x-3)²의 최솟값은 x=3
        java.util.function.Function<double[], Double> loss =
            p -> (p[0] - 3) * (p[0] - 3);
        java.util.function.Function<double[], double[]> gradient =
            p -> new double[]{2 * (p[0] - 3)};

        double[] start = {10.0};

        GradientDescent.TrainingHistory history =
            GradientDescent.minimizeWithHistory(loss, gradient, start, 0.1, 20);

        System.out.println("\n에폭 | x값    | 손실   | 시각화");
        System.out.println("-----|--------|--------|" + repeat("-", 30));

        for (int i = 0; i < history.losses.length; i += 2) {
            double x = history.positions[i][0];
            double l = history.losses[i];
            String bar = repeat("█", Math.min((int)(l), 25));

            System.out.printf("%4d | %6.3f | %6.3f | %s%n", i, x, l, bar);
        }

        System.out.printf("\n최종: x = %.4f (목표: 3.0)%n",
            history.positions[history.positions.length-1][0]);
    }

    private static String repeat(String s, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            sb.append(s);
        }
        return sb.toString();
    }
}
