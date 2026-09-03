package com.aimath.chapter8;

import com.aimath.core.Matrix;
import com.aimath.core.Vector;
import com.aimath.core.Activation;
import com.aimath.core.Loss;

import java.util.Random;

/**
 * 8장 예제: MNIST 손글씨 분류기
 * 784 → 128 → 64 → 10 구조의 신경망
 *
 * 이 예제는 합성 데이터로 동작하며, 실제 MNIST는 MnistLoader 사용
 */
public class MnistClassifier {

    // 네트워크 구조: 784 → 128 → 64 → 10
    private static final int INPUT_SIZE = 784;
    private static final int HIDDEN1_SIZE = 128;
    private static final int HIDDEN2_SIZE = 64;
    private static final int OUTPUT_SIZE = 10;

    // 가중치와 편향
    private double[][] W1, W2, W3;  // 가중치
    private double[] b1, b2, b3;     // 편향

    // 학습률
    private double learningRate = 0.01;

    // 난수 생성기
    private Random random = new Random(42);

    public MnistClassifier() {
        initializeWeights();
    }

    /**
     * 가중치 초기화 (He 초기화 사용)
     */
    private void initializeWeights() {
        // He 초기화: sqrt(2/n) 스케일
        double scale1 = Math.sqrt(2.0 / INPUT_SIZE);
        double scale2 = Math.sqrt(2.0 / HIDDEN1_SIZE);
        double scale3 = Math.sqrt(2.0 / HIDDEN2_SIZE);

        W1 = randomMatrix(HIDDEN1_SIZE, INPUT_SIZE, scale1);
        W2 = randomMatrix(HIDDEN2_SIZE, HIDDEN1_SIZE, scale2);
        W3 = randomMatrix(OUTPUT_SIZE, HIDDEN2_SIZE, scale3);

        b1 = new double[HIDDEN1_SIZE];
        b2 = new double[HIDDEN2_SIZE];
        b3 = new double[OUTPUT_SIZE];

        System.out.println("가중치 초기화 완료:");
        System.out.printf("  W1: (%d × %d), b1: (%d)%n", W1.length, W1[0].length, b1.length);
        System.out.printf("  W2: (%d × %d), b2: (%d)%n", W2.length, W2[0].length, b2.length);
        System.out.printf("  W3: (%d × %d), b3: (%d)%n", W3.length, W3[0].length, b3.length);
    }

    /**
     * 순전파 (Forward Pass)
     */
    public double[] forward(double[] x) {
        // Layer 1: Linear + ReLU
        double[] z1 = Vector.add(Matrix.multiplyVector(W1, x), b1);
        double[] a1 = Activation.relu(z1);

        // Layer 2: Linear + ReLU
        double[] z2 = Vector.add(Matrix.multiplyVector(W2, a1), b2);
        double[] a2 = Activation.relu(z2);

        // Layer 3: Linear + Softmax
        double[] z3 = Vector.add(Matrix.multiplyVector(W3, a2), b3);
        double[] output = Activation.softmax(z3);

        return output;
    }

    /**
     * 순전파 + 중간 값 저장 (역전파용)
     */
    private class ForwardCache {
        double[] x, z1, a1, z2, a2, z3, output;
    }

    private ForwardCache forwardWithCache(double[] x) {
        ForwardCache cache = new ForwardCache();
        cache.x = x;

        cache.z1 = Vector.add(Matrix.multiplyVector(W1, x), b1);
        cache.a1 = Activation.relu(cache.z1);

        cache.z2 = Vector.add(Matrix.multiplyVector(W2, cache.a1), b2);
        cache.a2 = Activation.relu(cache.z2);

        cache.z3 = Vector.add(Matrix.multiplyVector(W3, cache.a2), b3);
        cache.output = Activation.softmax(cache.z3);

        return cache;
    }

    /**
     * 역전파 + 가중치 업데이트
     */
    private double backward(ForwardCache cache, int label) {
        // 손실 계산
        double loss = Loss.crossEntropy(cache.output, label);

        // 원-핫 인코딩
        double[] oneHot = Loss.oneHot(label, OUTPUT_SIZE);

        // 출력층 그래디언트: dL/dz3 = output - oneHot
        double[] dz3 = Loss.crossEntropyGradient(cache.output, oneHot);

        // W3, b3 그래디언트
        double[][] dW3 = outerProduct(dz3, cache.a2);
        double[] db3 = dz3;

        // Layer 2 역전파
        double[] da2 = matrixTransposeVector(W3, dz3);
        double[] dz2 = elementwiseMultiply(da2, reluDerivative(cache.z2));
        double[][] dW2 = outerProduct(dz2, cache.a1);
        double[] db2 = dz2;

        // Layer 1 역전파
        double[] da1 = matrixTransposeVector(W2, dz2);
        double[] dz1 = elementwiseMultiply(da1, reluDerivative(cache.z1));
        double[][] dW1 = outerProduct(dz1, cache.x);
        double[] db1 = dz1;

        // 가중치 업데이트 (SGD)
        updateWeights(W3, dW3);
        updateBias(b3, db3);
        updateWeights(W2, dW2);
        updateBias(b2, db2);
        updateWeights(W1, dW1);
        updateBias(b1, db1);

        return loss;
    }

    /**
     * 단일 샘플 학습
     */
    public double trainStep(double[] x, int label) {
        ForwardCache cache = forwardWithCache(x);
        return backward(cache, label);
    }

    /**
     * 예측
     */
    public int predict(double[] x) {
        double[] output = forward(x);
        int predicted = 0;
        for (int i = 1; i < output.length; i++) {
            if (output[i] > output[predicted]) {
                predicted = i;
            }
        }
        return predicted;
    }

    /**
     * 정확도 계산
     */
    public double accuracy(double[][] images, int[] labels) {
        int correct = 0;
        for (int i = 0; i < images.length; i++) {
            if (predict(images[i]) == labels[i]) {
                correct++;
            }
        }
        return (double) correct / images.length;
    }

    // ===== 유틸리티 메서드 =====

    private double[][] randomMatrix(int rows, int cols, double scale) {
        double[][] m = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                m[i][j] = random.nextGaussian() * scale;
            }
        }
        return m;
    }

    private double[][] outerProduct(double[] a, double[] b) {
        double[][] result = new double[a.length][b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                result[i][j] = a[i] * b[j];
            }
        }
        return result;
    }

    private double[] matrixTransposeVector(double[][] W, double[] v) {
        // W^T × v
        double[] result = new double[W[0].length];
        for (int j = 0; j < W[0].length; j++) {
            for (int i = 0; i < W.length; i++) {
                result[j] += W[i][j] * v[i];
            }
        }
        return result;
    }

    private double[] elementwiseMultiply(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * b[i];
        }
        return result;
    }

    private double[] reluDerivative(double[] z) {
        double[] result = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            result[i] = z[i] > 0 ? 1.0 : 0.0;
        }
        return result;
    }

    private void updateWeights(double[][] W, double[][] dW) {
        for (int i = 0; i < W.length; i++) {
            for (int j = 0; j < W[0].length; j++) {
                W[i][j] -= learningRate * dW[i][j];
            }
        }
    }

    private void updateBias(double[] b, double[] db) {
        for (int i = 0; i < b.length; i++) {
            b[i] -= learningRate * db[i];
        }
    }

    // ===== 메인 =====

    public static void main(String[] args) {
        System.out.println("=== 8장: MNIST 손글씨 분류기 ===\n");

        // 1. 네트워크 구조 설명
        System.out.println("네트워크 구조:");
        System.out.println("  입력: 28×28 = 784 픽셀");
        System.out.println("  은닉층1: 128 뉴런 + ReLU");
        System.out.println("  은닉층2: 64 뉴런 + ReLU");
        System.out.println("  출력: 10 클래스 + Softmax");
        System.out.println();

        // 2. 분류기 생성
        MnistClassifier classifier = new MnistClassifier();
        System.out.println();

        // 3. 합성 데이터 생성 (실제로는 MnistLoader 사용)
        System.out.println("합성 데이터 생성 (시연용)...");
        SyntheticData data = generateSyntheticData(1000, 200);
        System.out.printf("  훈련: %d 샘플, 테스트: %d 샘플%n",
            data.trainImages.length, data.testImages.length);
        System.out.println();

        // 4. 학습
        System.out.println("학습 시작...");
        System.out.println("에폭 | 평균 손실 | 테스트 정확도");
        System.out.println("-----|----------|-------------");

        int epochs = 10;
        for (int epoch = 1; epoch <= epochs; epoch++) {
            double totalLoss = 0;

            // 미니배치 학습 (배치 크기 = 1 for simplicity)
            int[] indices = shuffleIndices(data.trainImages.length);
            for (int idx : indices) {
                totalLoss += classifier.trainStep(data.trainImages[idx], data.trainLabels[idx]);
            }

            double avgLoss = totalLoss / data.trainImages.length;
            double testAcc = classifier.accuracy(data.testImages, data.testLabels);

            System.out.printf("%4d | %8.4f | %6.2f%%%n", epoch, avgLoss, testAcc * 100);
        }

        // 5. 최종 평가
        System.out.println();
        System.out.println("=== 최종 평가 ===");
        double finalAcc = classifier.accuracy(data.testImages, data.testLabels);
        System.out.printf("테스트 정확도: %.2f%%%n", finalAcc * 100);

        // 6. 샘플 예측
        System.out.println();
        System.out.println("=== 샘플 예측 ===");
        for (int i = 0; i < 5; i++) {
            int predicted = classifier.predict(data.testImages[i]);
            int actual = data.testLabels[i];
            double[] probs = classifier.forward(data.testImages[i]);
            String status = predicted == actual ? "O" : "X";
            System.out.printf("샘플 %d: 예측=%d, 실제=%d, 확률=%.2f%% [%s]%n",
                i, predicted, actual, probs[predicted] * 100, status);
        }

        System.out.println();
        System.out.println("=== 학습 완료! ===");
        System.out.println("실제 MNIST 데이터로 학습하면 97%+ 정확도 달성 가능");
    }

    // ===== 합성 데이터 =====

    static class SyntheticData {
        double[][] trainImages, testImages;
        int[] trainLabels, testLabels;
    }

    /**
     * 각 숫자에 대한 간단한 패턴 생성
     * 실제 MNIST 대신 시연용
     */
    private static SyntheticData generateSyntheticData(int trainSize, int testSize) {
        Random rand = new Random(123);
        SyntheticData data = new SyntheticData();

        data.trainImages = new double[trainSize][INPUT_SIZE];
        data.trainLabels = new int[trainSize];
        data.testImages = new double[testSize][INPUT_SIZE];
        data.testLabels = new int[testSize];

        // 각 숫자에 대한 "패턴" 생성 (고유한 특징)
        double[][] patterns = new double[10][INPUT_SIZE];
        for (int digit = 0; digit < 10; digit++) {
            // 각 숫자는 특정 영역에 집중된 패턴
            for (int i = 0; i < INPUT_SIZE; i++) {
                // digit에 따라 다른 위치에 강한 값
                int row = i / 28;
                int col = i % 28;

                double center = (digit + 1) * 2.5;
                double dist = Math.sqrt((row - center) * (row - center) +
                                       (col - center) * (col - center));
                patterns[digit][i] = Math.exp(-dist / 10.0);
            }
        }

        // 훈련 데이터 생성 (패턴 + 노이즈)
        for (int i = 0; i < trainSize; i++) {
            int label = rand.nextInt(10);
            data.trainLabels[i] = label;
            for (int j = 0; j < INPUT_SIZE; j++) {
                data.trainImages[i][j] = Math.max(0, Math.min(1,
                    patterns[label][j] + rand.nextGaussian() * 0.1));
            }
        }

        // 테스트 데이터
        for (int i = 0; i < testSize; i++) {
            int label = rand.nextInt(10);
            data.testLabels[i] = label;
            for (int j = 0; j < INPUT_SIZE; j++) {
                data.testImages[i][j] = Math.max(0, Math.min(1,
                    patterns[label][j] + rand.nextGaussian() * 0.1));
            }
        }

        return data;
    }

    private static int[] shuffleIndices(int n) {
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        Random rand = new Random();
        for (int i = n - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        return indices;
    }
}
