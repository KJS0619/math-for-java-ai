package com.aimath.core;

/**
 * 벡터 연산을 위한 유틸리티 클래스
 * AI 수학의 기초가 되는 벡터 연산들을 Java로 구현
 */
public class Vector {

    /**
     * 벡터 덧셈: a + b
     */
    public static double[] add(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("벡터 길이가 다릅니다: " + a.length + " vs " + b.length);
        }
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    /**
     * 스칼라 곱: scalar * v
     */
    public static double[] scale(double[] v, double scalar) {
        double[] result = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            result[i] = v[i] * scalar;
        }
        return result;
    }

    /**
     * 내적 (Dot Product): a · b
     * 두 벡터의 유사도를 측정하는 기본 연산
     */
    public static double dot(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("벡터 길이가 다릅니다: " + a.length + " vs " + b.length);
        }
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    /**
     * L2 노름 (유클리드 거리): ||v||
     * 벡터의 크기를 계산
     */
    public static double norm(double[] v) {
        double sumOfSquares = 0;
        for (double x : v) {
            sumOfSquares += x * x;
        }
        return Math.sqrt(sumOfSquares);
    }

    /**
     * L1 노름 (맨해튼 거리): |v|
     */
    public static double normL1(double[] v) {
        double sum = 0;
        for (double x : v) {
            sum += Math.abs(x);
        }
        return sum;
    }

    /**
     * 정규화: v / ||v||
     * 벡터를 단위 벡터로 변환
     */
    public static double[] normalize(double[] v) {
        double n = norm(v);
        if (n == 0) {
            throw new IllegalArgumentException("영벡터는 정규화할 수 없습니다");
        }
        return scale(v, 1.0 / n);
    }

    /**
     * 코사인 유사도: (a · b) / (||a|| * ||b||)
     * -1 ~ 1 사이의 값, 1에 가까울수록 유사
     */
    public static double cosineSimilarity(double[] a, double[] b) {
        double dotProduct = dot(a, b);
        double normA = norm(a);
        double normB = norm(b);
        if (normA == 0 || normB == 0) {
            throw new IllegalArgumentException("영벡터의 코사인 유사도는 정의되지 않습니다");
        }
        return dotProduct / (normA * normB);
    }

    /**
     * 벡터를 문자열로 출력
     */
    public static String toString(double[] v) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            sb.append(String.format("%.4f", v[i]));
            if (i < v.length - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }
}
