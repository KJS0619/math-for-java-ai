package com.aimath.core;

/**
 * 행렬 연산을 위한 유틸리티 클래스
 * 신경망의 순전파/역전파에 필수적인 행렬 연산들을 구현
 */
public class Matrix {

    /**
     * 행렬 덧셈: A + B
     */
    public static double[][] add(double[][] a, double[][] b) {
        int rows = a.length;
        int cols = a[0].length;
        if (b.length != rows || b[0].length != cols) {
            throw new IllegalArgumentException("행렬 크기가 다릅니다");
        }
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        return result;
    }

    /**
     * 스칼라 곱: scalar * A
     */
    public static double[][] scale(double[][] a, double scalar) {
        int rows = a.length;
        int cols = a[0].length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = a[i][j] * scalar;
            }
        }
        return result;
    }

    /**
     * 행렬 곱셈: A × B
     * A: (m × n), B: (n × p) → 결과: (m × p)
     * 신경망 순전파의 핵심 연산
     */
    public static double[][] multiply(double[][] a, double[][] b) {
        int m = a.length;
        int n = a[0].length;
        int p = b[0].length;
        if (b.length != n) {
            throw new IllegalArgumentException(
                "행렬 곱셈 불가: A의 열 수(" + n + ")와 B의 행 수(" + b.length + ")가 다릅니다"
            );
        }
        double[][] result = new double[m][p];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    /**
     * 행렬-벡터 곱셈: A × v
     * A: (m × n), v: (n) → 결과: (m)
     */
    public static double[] multiplyVector(double[][] a, double[] v) {
        int m = a.length;
        int n = a[0].length;
        if (v.length != n) {
            throw new IllegalArgumentException(
                "곱셈 불가: 행렬 열 수(" + n + ")와 벡터 길이(" + v.length + ")가 다릅니다"
            );
        }
        double[] result = new double[m];
        for (int i = 0; i < m; i++) {
            double sum = 0;
            for (int j = 0; j < n; j++) {
                sum += a[i][j] * v[j];
            }
            result[i] = sum;
        }
        return result;
    }

    /**
     * 전치 행렬: A^T
     */
    public static double[][] transpose(double[][] a) {
        int rows = a.length;
        int cols = a[0].length;
        double[][] result = new double[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = a[i][j];
            }
        }
        return result;
    }

    /**
     * 단위 행렬 생성: I
     */
    public static double[][] identity(int n) {
        double[][] result = new double[n][n];
        for (int i = 0; i < n; i++) {
            result[i][i] = 1.0;
        }
        return result;
    }

    /**
     * 영행렬 생성
     */
    public static double[][] zeros(int rows, int cols) {
        return new double[rows][cols];
    }

    /**
     * 랜덤 행렬 생성 (가중치 초기화용)
     */
    public static double[][] random(int rows, int cols, double scale) {
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = (Math.random() - 0.5) * 2 * scale;
            }
        }
        return result;
    }

    /**
     * 행렬 형태 반환: (rows, cols)
     */
    public static int[] shape(double[][] a) {
        return new int[]{a.length, a[0].length};
    }

    /**
     * 행렬을 문자열로 출력
     */
    public static String toString(double[][] a) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < a.length; i++) {
            sb.append("[");
            for (int j = 0; j < a[i].length; j++) {
                sb.append(String.format("%8.4f", a[i][j]));
                if (j < a[i].length - 1) sb.append(", ");
            }
            sb.append("]\n");
        }
        return sb.toString();
    }
}
