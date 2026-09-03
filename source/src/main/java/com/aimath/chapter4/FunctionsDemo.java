package com.aimath.chapter4;

import com.aimath.core.Activation;
import com.aimath.core.Vector;

/**
 * 4장 예제: 함수와 활성화 함수
 * 선형/비선형 함수, Sigmoid, Tanh, ReLU, Softmax
 */
public class FunctionsDemo {

    public static void main(String[] args) {
        System.out.println("=== 4장: 함수와 활성화 함수 ===\n");

        // 1. 선형 함수 vs 비선형 함수
        System.out.println("1. 선형 vs 비선형 함수");
        linearVsNonlinear();
        System.out.println();

        // 2. Sigmoid 함수
        System.out.println("2. Sigmoid 함수: σ(x) = 1 / (1 + e^(-x))");
        sigmoidDemo();
        System.out.println();

        // 3. Tanh 함수
        System.out.println("3. Tanh 함수: 출력 범위 (-1, 1)");
        tanhDemo();
        System.out.println();

        // 4. ReLU 함수
        System.out.println("4. ReLU 함수: max(0, x)");
        reluDemo();
        System.out.println();

        // 5. Softmax 함수
        System.out.println("5. Softmax 함수: 확률 분포 출력");
        softmaxDemo();
        System.out.println();

        // 6. 활성화 함수 비교
        System.out.println("=== 활성화 함수 비교 ===");
        compareActivations();
        System.out.println();

        // 7. AI 활용: 이진 분류
        System.out.println("=== AI 활용: 이진 분류 (Sigmoid) ===");
        binaryClassificationDemo();
        System.out.println();

        // 8. AI 활용: 다중 클래스 분류
        System.out.println("=== AI 활용: 다중 클래스 분류 (Softmax) ===");
        multiClassDemo();
    }

    private static void linearVsNonlinear() {
        System.out.println("선형 함수: f(x) = 2x + 1");
        for (double x : new double[]{-2, -1, 0, 1, 2}) {
            System.out.printf("  f(%.0f) = %.1f%n", x, 2 * x + 1);
        }

        System.out.println("\n비선형 함수: f(x) = x²");
        for (double x : new double[]{-2, -1, 0, 1, 2}) {
            System.out.printf("  f(%.0f) = %.1f%n", x, x * x);
        }

        System.out.println("\n→ 신경망에 비선형 활성화가 필요한 이유:");
        System.out.println("  선형 함수의 합성은 여전히 선형 (복잡한 패턴 학습 불가)");
        System.out.println("  비선형 함수가 있어야 복잡한 결정 경계 표현 가능");
    }

    private static void sigmoidDemo() {
        System.out.println("입력 → Sigmoid 출력");
        double[] inputs = {-5, -2, -1, 0, 1, 2, 5};

        for (double x : inputs) {
            double y = Activation.sigmoid(x);
            String bar = repeat("█", (int) (y * 20));
            System.out.printf("  σ(%5.1f) = %.4f %s%n", x, y, bar);
        }

        System.out.println("\n특징:");
        System.out.println("  - 출력 범위: (0, 1)");
        System.out.println("  - x=0에서 출력 0.5");
        System.out.println("  - 확률 해석 가능 (이진 분류)");
        System.out.println("  - 단점: 그래디언트 소실 (양 끝에서)");
    }

    private static void tanhDemo() {
        System.out.println("입력 → Tanh 출력");
        double[] inputs = {-3, -2, -1, 0, 1, 2, 3};

        for (double x : inputs) {
            double y = Activation.tanh(x);
            System.out.printf("  tanh(%5.1f) = %7.4f%n", x, y);
        }

        System.out.println("\n특징:");
        System.out.println("  - 출력 범위: (-1, 1)");
        System.out.println("  - 중심이 0 (Sigmoid보다 학습에 유리)");
        System.out.println("  - RNN/LSTM에서 자주 사용");
    }

    private static void reluDemo() {
        System.out.println("입력 → ReLU 출력");
        double[] inputs = {-3, -1, 0, 1, 3, 5};

        for (double x : inputs) {
            double y = Activation.relu(x);
            String bar = repeat("█", (int) y);
            System.out.printf("  ReLU(%5.1f) = %.1f %s%n", x, y, bar);
        }

        System.out.println("\n특징:");
        System.out.println("  - 출력 범위: [0, ∞)");
        System.out.println("  - 계산이 매우 빠름 (현대 딥러닝 기본)");
        System.out.println("  - 그래디언트 소실 완화");
        System.out.println("  - 단점: Dying ReLU (음수에서 학습 안됨)");

        System.out.println("\nLeaky ReLU로 해결:");
        for (double x : new double[]{-3, -1, 0, 1, 3}) {
            double y = Activation.leakyRelu(x, 0.1);
            System.out.printf("  LeakyReLU(%.1f, α=0.1) = %.2f%n", x, y);
        }
    }

    private static void softmaxDemo() {
        System.out.println("로짓(점수) → 확률 변환");

        double[] logits = {2.0, 1.0, 0.1};
        double[] probs = Activation.softmax(logits);

        System.out.println("로짓: " + Vector.toString(logits));
        System.out.println("확률: " + Vector.toString(probs));

        double sum = 0;
        for (double p : probs) sum += p;
        System.out.printf("확률의 합: %.4f (항상 1)%n", sum);

        System.out.println("\n특징:");
        System.out.println("  - 출력의 합 = 1 (확률 분포)");
        System.out.println("  - 가장 큰 로짓 → 가장 높은 확률");
        System.out.println("  - 다중 클래스 분류의 출력층");
    }

    private static void compareActivations() {
        double x = 2.5;
        System.out.println("x = " + x + "일 때:");
        System.out.printf("  Sigmoid:  %.4f (0~1)%n", Activation.sigmoid(x));
        System.out.printf("  Tanh:     %.4f (-1~1)%n", Activation.tanh(x));
        System.out.printf("  ReLU:     %.4f (0~∞)%n", Activation.relu(x));

        System.out.println("\n도함수 (역전파에 사용):");
        System.out.printf("  σ'(%.1f) = %.4f%n", x, Activation.sigmoidDerivative(x));
        System.out.printf("  tanh'(%.1f) = %.4f%n", x, Activation.tanhDerivative(x));
        System.out.printf("  ReLU'(%.1f) = %.1f%n", x, Activation.reluDerivative(x));
    }

    private static void binaryClassificationDemo() {
        System.out.println("스팸 메일 분류 (이진 분류)");

        // 모델이 출력한 로짓
        double[] logits = {-2.5, 0.3, 1.8, 3.5};
        String[] labels = {"확실히 정상", "아마 정상", "아마 스팸", "확실히 스팸"};

        System.out.println("\n로짓 → Sigmoid → 확률 → 판정");
        for (int i = 0; i < logits.length; i++) {
            double prob = Activation.sigmoid(logits[i]);
            String decision = prob > 0.5 ? "스팸" : "정상";
            System.out.printf("  %.1f → %.2f → %s (%s)%n",
                logits[i], prob, decision, labels[i]);
        }

        System.out.println("\n임계값(threshold) = 0.5가 기본");
    }

    private static void multiClassDemo() {
        System.out.println("이미지 분류: 개 vs 고양이 vs 새");

        // 3개 클래스에 대한 모델 출력
        double[] logits = {3.2, 1.5, 0.8};
        double[] probs = Activation.softmax(logits);

        String[] classes = {"개", "고양이", "새"};

        System.out.println("\n로짓 → Softmax → 확률");
        for (int i = 0; i < classes.length; i++) {
            String bar = repeat("█", (int) (probs[i] * 30));
            System.out.printf("  %s: %.1f → %.1f%% %s%n",
                classes[i], logits[i], probs[i] * 100, bar);
        }

        // 예측 클래스 찾기
        int predicted = 0;
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > probs[predicted]) predicted = i;
        }
        System.out.printf("\n예측: %s (%.1f%% 확률)%n", classes[predicted], probs[predicted] * 100);
    }

    private static String repeat(String s, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            sb.append(s);
        }
        return sb.toString();
    }
}
