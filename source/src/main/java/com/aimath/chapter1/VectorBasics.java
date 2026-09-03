package com.aimath.chapter1;

import com.aimath.core.Vector;

/**
 * 1장 예제: 벡터의 기초
 * 벡터가 무엇이고, AI에서 어떻게 사용되는지 학습
 */
public class VectorBasics {

    public static void main(String[] args) {
        System.out.println("=== 1장: 벡터의 기초 ===\n");

        // 1. 벡터 생성
        System.out.println("1. 벡터 생성");
        double[] v1 = {1.0, 2.0, 3.0};
        double[] v2 = {4.0, 5.0, 6.0};
        System.out.println("v1 = " + Vector.toString(v1));
        System.out.println("v2 = " + Vector.toString(v2));
        System.out.println();

        // 2. 벡터 덧셈
        System.out.println("2. 벡터 덧셈: v1 + v2");
        double[] sum = Vector.add(v1, v2);
        System.out.println("결과 = " + Vector.toString(sum));
        System.out.println();

        // 3. 스칼라 곱
        System.out.println("3. 스칼라 곱: 2 * v1");
        double[] scaled = Vector.scale(v1, 2.0);
        System.out.println("결과 = " + Vector.toString(scaled));
        System.out.println();

        // 4. 내적 (Dot Product)
        System.out.println("4. 내적: v1 · v2");
        double dot = Vector.dot(v1, v2);
        System.out.println("결과 = " + dot);
        System.out.println("해석: 두 벡터가 얼마나 같은 방향인지 나타냄");
        System.out.println();

        // 5. 벡터의 크기 (Norm)
        System.out.println("5. 벡터의 크기: ||v1||");
        double norm = Vector.norm(v1);
        System.out.println("결과 = " + norm);
        System.out.println();

        // 6. 정규화 (Normalization)
        System.out.println("6. 정규화: v1 / ||v1||");
        double[] normalized = Vector.normalize(v1);
        System.out.println("결과 = " + Vector.toString(normalized));
        System.out.println("크기 확인 = " + Vector.norm(normalized) + " (1이어야 함)");
        System.out.println();

        // 7. 코사인 유사도
        System.out.println("7. 코사인 유사도");
        double similarity = Vector.cosineSimilarity(v1, v2);
        System.out.println("cos(v1, v2) = " + similarity);
        System.out.println("해석: 1에 가까울수록 유사, 0이면 직교, -1이면 반대");
        System.out.println();

        // AI 연결: 임베딩 유사도
        System.out.println("=== AI 연결: 단어 임베딩 유사도 ===");
        double[] king = {0.5, 0.8, 0.2};    // 가상의 king 임베딩
        double[] queen = {0.6, 0.7, 0.3};   // 가상의 queen 임베딩
        double[] apple = {-0.3, 0.1, 0.9};  // 가상의 apple 임베딩

        System.out.println("king-queen 유사도: " +
            String.format("%.4f", Vector.cosineSimilarity(king, queen)));
        System.out.println("king-apple 유사도: " +
            String.format("%.4f", Vector.cosineSimilarity(king, apple)));
        System.out.println("→ king과 queen이 더 유사함 (의미적으로 관련)");
    }
}
