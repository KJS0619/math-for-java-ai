package com.aimath.chapter7;

import com.aimath.core.Loss;
import com.aimath.core.Activation;
import com.aimath.core.Vector;

/**
 * 7장 예제: 손실 함수
 * MSE, Cross-Entropy, 손실과 그래디언트의 관계
 */
public class LossFunctionsDemo {

    public static void main(String[] args) {
        System.out.println("=== 7장: 손실 함수 ===\n");

        // 1. MSE (회귀용)
        System.out.println("1. MSE (Mean Squared Error) - 회귀");
        mseDemo();
        System.out.println();

        // 2. Binary Cross-Entropy (이진 분류용)
        System.out.println("2. Binary Cross-Entropy - 이진 분류");
        binaryCrossEntropyDemo();
        System.out.println();

        // 3. Categorical Cross-Entropy (다중 분류용)
        System.out.println("3. Cross-Entropy - 다중 클래스 분류");
        categoricalCrossEntropyDemo();
        System.out.println();

        // 4. 손실과 그래디언트
        System.out.println("4. 손실 → 그래디언트 → 가중치 업데이트");
        lossToGradientDemo();
        System.out.println();

        // 5. 원-핫 인코딩
        System.out.println("5. 원-핫 인코딩");
        oneHotDemo();
        System.out.println();

        // 6. AI 활용: 분류 학습 시뮬레이션
        System.out.println("=== AI 활용: 분류 학습 시뮬레이션 ===");
        classificationTrainingDemo();
    }

    private static void mseDemo() {
        System.out.println("MSE = (1/n) × Σ(예측 - 실제)²");
        System.out.println("→ 예측과 실제 차이의 제곱 평균\n");

        // 집값 예측 예시
        double[] predicted = {300, 450, 200, 500};  // 예측 집값 (만원)
        double[] actual = {320, 440, 210, 480};     // 실제 집값

        System.out.println("예측: " + Vector.toString(predicted));
        System.out.println("실제: " + Vector.toString(actual));

        double mse = Loss.mse(predicted, actual);
        System.out.printf("MSE = %.2f%n", mse);
        System.out.printf("RMSE = √MSE = %.2f (해석 가능한 단위)%n", Math.sqrt(mse));

        // 개별 오차
        System.out.println("\n각 샘플의 오차:");
        for (int i = 0; i < predicted.length; i++) {
            double error = predicted[i] - actual[i];
            System.out.printf("  샘플%d: 오차 %.0f, 제곱오차 %.0f%n",
                i+1, error, error * error);
        }
    }

    private static void binaryCrossEntropyDemo() {
        System.out.println("BCE = -[y·log(p) + (1-y)·log(1-p)]");
        System.out.println("→ 확률 예측의 정확도 측정\n");

        // 스팸 분류 예시
        System.out.println("스팸 분류 (y=1: 스팸, y=0: 정상):");
        System.out.println("예측확률 | 실제 | BCE 손실");
        System.out.println("---------|------|----------");

        double[][] cases = {
            {0.9, 1},  // 스팸을 높은 확률로 맞춤
            {0.1, 1},  // 스팸을 낮은 확률로 예측 (틀림)
            {0.1, 0},  // 정상을 낮은 확률로 맞춤
            {0.9, 0},  // 정상을 높은 확률로 예측 (틀림)
        };

        for (double[] c : cases) {
            double bce = Loss.binaryCrossEntropy(c[0], c[1]);
            String status = (c[0] > 0.5 && c[1] == 1) || (c[0] <= 0.5 && c[1] == 0)
                          ? "정답" : "오답";
            System.out.printf("   %.1f   |  %.0f  | %6.3f (%s)%n", c[0], c[1], bce, status);
        }

        System.out.println("\n→ 틀린 예측일수록 손실이 큼");
        System.out.println("→ 확률 예측이 극단적으로 틀리면 손실이 매우 큼");
    }

    private static void categoricalCrossEntropyDemo() {
        System.out.println("CE = -Σ y_true × log(y_pred)");
        System.out.println("→ 정답 클래스의 예측 확률이 높을수록 손실 낮음\n");

        // MNIST 숫자 분류 예시
        String[] classes = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
        int trueLabel = 3;  // 실제: 숫자 3

        // 좋은 예측
        double[] goodPred = {0.01, 0.02, 0.05, 0.80, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01};

        // 나쁜 예측
        double[] badPred = {0.15, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05};

        System.out.println("실제 레이블: " + trueLabel + " (숫자 3)");

        System.out.println("\n좋은 예측 (정답 클래스에 80% 확률):");
        double ceGood = Loss.crossEntropy(goodPred, trueLabel);
        System.out.printf("  P(3) = %.2f → CE = %.4f%n", goodPred[3], ceGood);

        System.out.println("\n나쁜 예측 (균등하게 분포):");
        double ceBad = Loss.crossEntropy(badPred, trueLabel);
        System.out.printf("  P(3) = %.2f → CE = %.4f%n", badPred[3], ceBad);

        System.out.println("\n→ 좋은 예측일수록 Cross-Entropy 낮음");
    }

    private static void lossToGradientDemo() {
        System.out.println("학습 흐름: 손실 계산 → 그래디언트 → 가중치 업데이트\n");

        // 예측과 실제
        double[] predicted = {0.7, 0.2, 0.1};  // softmax 출력
        double[] actual = {1.0, 0.0, 0.0};     // 원-핫 (정답: 클래스 0)

        System.out.println("예측 (softmax): " + Vector.toString(predicted));
        System.out.println("실제 (원-핫):   " + Vector.toString(actual));

        // 손실 계산
        double loss = Loss.crossEntropy(predicted, actual);
        System.out.printf("\nCross-Entropy 손실: %.4f%n", loss);

        // 그래디언트 계산 (softmax + CE의 그래디언트는 간단)
        double[] gradient = Loss.crossEntropyGradient(predicted, actual);
        System.out.println("그래디언트 (∂L/∂z): " + Vector.toString(gradient));

        System.out.println("\n해석:");
        System.out.println("  클래스 0: 그래디언트 음수 → 확률 높여야 함");
        System.out.println("  클래스 1,2: 그래디언트 양수 → 확률 낮춰야 함");

        // 가상의 가중치 업데이트
        double lr = 0.1;
        double[] newLogits = new double[3];
        for (int i = 0; i < 3; i++) {
            newLogits[i] = predicted[i] - lr * gradient[i];
        }
        System.out.println("\n업데이트 후 로짓: " + Vector.toString(newLogits));
        System.out.println("→ 클래스 0의 값이 증가");
    }

    private static void oneHotDemo() {
        System.out.println("레이블 → 원-핫 벡터 변환\n");

        int numClasses = 5;
        int[] labels = {0, 2, 4, 1};

        for (int label : labels) {
            double[] oneHot = Loss.oneHot(label, numClasses);
            System.out.printf("레이블 %d → %s%n", label, Vector.toString(oneHot));
        }

        System.out.println("\n→ Cross-Entropy 계산에 필요");
    }

    private static void classificationTrainingDemo() {
        System.out.println("3클래스 분류 학습 시뮬레이션");
        System.out.println("목표: 클래스 1을 정확히 예측\n");

        double[] logits = {1.0, 0.5, 0.5};  // 초기 로짓
        int trueLabel = 1;
        double lr = 0.5;

        System.out.println("에폭 | 로짓           | 확률           | 손실   | 예측");
        System.out.println("-----|----------------|----------------|--------|------");

        for (int epoch = 0; epoch < 8; epoch++) {
            // Softmax로 확률 변환
            double[] probs = Activation.softmax(logits);

            // 손실 계산
            double loss = Loss.crossEntropy(probs, trueLabel);

            // 예측 클래스
            int predicted = 0;
            for (int i = 1; i < probs.length; i++) {
                if (probs[i] > probs[predicted]) predicted = i;
            }
            String correct = predicted == trueLabel ? "O" : "X";

            System.out.printf("%4d | [%.2f,%.2f,%.2f] | [%.2f,%.2f,%.2f] | %.4f | %d %s%n",
                epoch,
                logits[0], logits[1], logits[2],
                probs[0], probs[1], probs[2],
                loss, predicted, correct);

            // 그래디언트 계산 및 업데이트
            double[] oneHot = Loss.oneHot(trueLabel, 3);
            double[] gradient = Loss.crossEntropyGradient(probs, oneHot);

            for (int i = 0; i < logits.length; i++) {
                logits[i] -= lr * gradient[i];
            }
        }

        System.out.println("\n→ 학습이 진행됨에 따라:");
        System.out.println("  - 클래스 1의 확률이 증가");
        System.out.println("  - Cross-Entropy 손실이 감소");
        System.out.println("  - 예측이 정답(1)으로 수렴");
    }
}
