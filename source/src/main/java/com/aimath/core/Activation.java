package com.aimath.core;

/**
 * 활성화 함수 모음
 * 신경망에 비선형성을 추가하는 함수들
 */
public class Activation {

    /**
     * Sigmoid: σ(x) = 1 / (1 + e^(-x))
     * 출력 범위: (0, 1)
     * 용도: 이진 분류의 출력층
     */
    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static double[] sigmoid(double[] v) {
        double[] result = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            result[i] = sigmoid(v[i]);
        }
        return result;
    }

    /**
     * Sigmoid 도함수: σ'(x) = σ(x) * (1 - σ(x))
     * 역전파에 사용
     */
    public static double sigmoidDerivative(double x) {
        double s = sigmoid(x);
        return s * (1 - s);
    }

    /**
     * Tanh: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
     * 출력 범위: (-1, 1)
     * Sigmoid보다 중심이 0이라 학습에 유리
     */
    public static double tanh(double x) {
        return Math.tanh(x);
    }

    public static double[] tanh(double[] v) {
        double[] result = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            result[i] = Math.tanh(v[i]);
        }
        return result;
    }

    /**
     * Tanh 도함수: tanh'(x) = 1 - tanh²(x)
     */
    public static double tanhDerivative(double x) {
        double t = Math.tanh(x);
        return 1 - t * t;
    }

    /**
     * ReLU: max(0, x)
     * 현대 신경망에서 가장 많이 사용
     * 계산이 빠르고 그래디언트 소실 문제 완화
     */
    public static double relu(double x) {
        return Math.max(0, x);
    }

    public static double[] relu(double[] v) {
        double[] result = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            result[i] = relu(v[i]);
        }
        return result;
    }

    /**
     * ReLU 도함수: 1 if x > 0, else 0
     */
    public static double reluDerivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }

    /**
     * Leaky ReLU: x if x > 0, else alpha * x
     * ReLU의 "dying ReLU" 문제 해결
     */
    public static double leakyRelu(double x, double alpha) {
        return x > 0 ? x : alpha * x;
    }

    public static double leakyRelu(double x) {
        return leakyRelu(x, 0.01);
    }

    /**
     * Softmax: e^(xi) / Σe^(xj)
     * 다중 클래스 분류의 출력층
     * 출력의 합이 1이 되는 확률 분포
     */
    public static double[] softmax(double[] v) {
        // 수치 안정성을 위해 최댓값을 빼줌
        double max = Double.NEGATIVE_INFINITY;
        for (double x : v) {
            if (x > max) max = x;
        }

        double[] exp = new double[v.length];
        double sum = 0;
        for (int i = 0; i < v.length; i++) {
            exp[i] = Math.exp(v[i] - max);
            sum += exp[i];
        }

        double[] result = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            result[i] = exp[i] / sum;
        }
        return result;
    }
}
