package com.aimath.core;

/**
 * 손실 함수 모음
 * 모델의 예측이 얼마나 틀렸는지 측정
 */
public class Loss {

    /**
     * MSE (Mean Squared Error): 평균 제곱 오차
     * 회귀 문제에 사용
     * L = (1/n) * Σ(y_pred - y_true)²
     */
    public static double mse(double[] predicted, double[] actual) {
        if (predicted.length != actual.length) {
            throw new IllegalArgumentException("배열 길이가 다릅니다");
        }
        double sum = 0;
        for (int i = 0; i < predicted.length; i++) {
            double diff = predicted[i] - actual[i];
            sum += diff * diff;
        }
        return sum / predicted.length;
    }

    /**
     * MSE 그래디언트: dL/d(y_pred) = 2 * (y_pred - y_true) / n
     */
    public static double[] mseGradient(double[] predicted, double[] actual) {
        double[] gradient = new double[predicted.length];
        double scale = 2.0 / predicted.length;
        for (int i = 0; i < predicted.length; i++) {
            gradient[i] = scale * (predicted[i] - actual[i]);
        }
        return gradient;
    }

    /**
     * Binary Cross-Entropy: 이진 분류용
     * L = -[y*log(p) + (1-y)*log(1-p)]
     */
    public static double binaryCrossEntropy(double predicted, double actual) {
        // 수치 안정성을 위한 클리핑
        predicted = Math.max(1e-15, Math.min(1 - 1e-15, predicted));
        return -(actual * Math.log(predicted) + (1 - actual) * Math.log(1 - predicted));
    }

    /**
     * Cross-Entropy: 다중 클래스 분류용
     * L = -Σ y_true * log(y_pred)
     *
     * @param predicted softmax 출력 (확률 분포)
     * @param actual 원-핫 인코딩된 정답
     */
    public static double crossEntropy(double[] predicted, double[] actual) {
        double sum = 0;
        for (int i = 0; i < predicted.length; i++) {
            if (actual[i] > 0) {
                // 수치 안정성을 위한 클리핑
                double p = Math.max(1e-15, predicted[i]);
                sum -= actual[i] * Math.log(p);
            }
        }
        return sum;
    }

    /**
     * Cross-Entropy (정수 레이블 버전)
     *
     * @param predicted softmax 출력
     * @param label 정답 클래스 인덱스 (0, 1, 2, ...)
     */
    public static double crossEntropy(double[] predicted, int label) {
        double p = Math.max(1e-15, predicted[label]);
        return -Math.log(p);
    }

    /**
     * Cross-Entropy 그래디언트 (softmax + cross-entropy 결합)
     * 매우 간단한 형태: y_pred - y_true
     */
    public static double[] crossEntropyGradient(double[] predicted, double[] actual) {
        double[] gradient = new double[predicted.length];
        for (int i = 0; i < predicted.length; i++) {
            gradient[i] = predicted[i] - actual[i];
        }
        return gradient;
    }

    /**
     * 정답 레이블을 원-핫 인코딩으로 변환
     */
    public static double[] oneHot(int label, int numClasses) {
        double[] result = new double[numClasses];
        result[label] = 1.0;
        return result;
    }
}
