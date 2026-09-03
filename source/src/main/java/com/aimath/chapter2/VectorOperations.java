package com.aimath.chapter2;

import com.aimath.core.Vector;

/**
 * 2장 예제: 벡터 연산
 * 내적, 노름, 정규화, 코사인 유사도의 AI 활용
 */
public class VectorOperations {

    public static void main(String[] args) {
        System.out.println("=== 2장: 벡터 연산 ===\n");

        // 1. 내적 (Dot Product) - 뉴런 계산의 핵심
        System.out.println("1. 내적 - 뉴런 계산의 핵심");
        double[] inputs = {0.5, 0.3, 0.8};
        double[] weights = {0.2, 0.4, 0.1};
        double bias = 0.1;

        double z = Vector.dot(inputs, weights) + bias;
        System.out.println("입력: " + Vector.toString(inputs));
        System.out.println("가중치: " + Vector.toString(weights));
        System.out.println("편향: " + bias);
        System.out.println("뉴런 출력 (z = dot(inputs, weights) + bias): " + z);
        System.out.println();

        // 2. 노름 (Norm) - 벡터의 크기
        System.out.println("2. 벡터의 크기 (Norm)");
        double[] v = {3.0, 4.0};
        System.out.println("v = " + Vector.toString(v));
        System.out.println("L2 노름 (유클리드): " + Vector.norm(v));
        System.out.println("L1 노름 (맨해튼): " + Vector.normL1(v));
        System.out.println("해석: L2는 직선 거리, L1은 격자 거리");
        System.out.println();

        // 3. 정규화 - 단위 벡터 만들기
        System.out.println("3. 정규화 - 단위 벡터");
        double[] normalized = Vector.normalize(v);
        System.out.println("원본: " + Vector.toString(v) + ", 크기: " + Vector.norm(v));
        System.out.println("정규화: " + Vector.toString(normalized) + ", 크기: " + Vector.norm(normalized));
        System.out.println();

        // 4. 코사인 유사도 - 방향의 유사성
        System.out.println("4. 코사인 유사도");
        double[] a = {1.0, 2.0, 3.0};
        double[] b = {2.0, 4.0, 6.0};  // a의 2배 (같은 방향)
        double[] c = {-1.0, -2.0, -3.0};  // a의 반대 방향

        System.out.println("a = " + Vector.toString(a));
        System.out.println("b = " + Vector.toString(b) + " (a의 2배)");
        System.out.println("c = " + Vector.toString(c) + " (a의 반대)");
        System.out.printf("cos(a, b) = %.4f (같은 방향)%n", Vector.cosineSimilarity(a, b));
        System.out.printf("cos(a, c) = %.4f (반대 방향)%n", Vector.cosineSimilarity(a, c));
        System.out.println();

        // 5. AI 활용: 추천 시스템
        System.out.println("=== AI 활용: 추천 시스템 ===");
        recommendationDemo();
        System.out.println();

        // 6. AI 활용: 단어 임베딩
        System.out.println("=== AI 활용: 단어 임베딩 유사도 ===");
        wordEmbeddingDemo();
        System.out.println();

        // 7. AI 활용: 그래디언트 클리핑
        System.out.println("=== AI 활용: 그래디언트 클리핑 ===");
        gradientClippingDemo();
    }

    /**
     * 추천 시스템 예제
     * 사용자와 아이템 간의 코사인 유사도로 추천
     */
    private static void recommendationDemo() {
        // 사용자 선호도 벡터 (장르: 액션, 로맨스, SF, 코미디)
        double[] user = {0.8, 0.2, 0.7, 0.3};

        // 영화 벡터
        double[] movie1 = {0.9, 0.1, 0.8, 0.2};  // 액션 SF 영화
        double[] movie2 = {0.1, 0.9, 0.2, 0.7};  // 로맨스 코미디
        double[] movie3 = {0.7, 0.3, 0.6, 0.4};  // 혼합 장르

        System.out.println("사용자 선호: " + Vector.toString(user));
        System.out.println("  (액션=0.8, 로맨스=0.2, SF=0.7, 코미디=0.3)");
        System.out.println();

        double sim1 = Vector.cosineSimilarity(user, movie1);
        double sim2 = Vector.cosineSimilarity(user, movie2);
        double sim3 = Vector.cosineSimilarity(user, movie3);

        System.out.printf("영화1 (액션SF) 유사도: %.4f%n", sim1);
        System.out.printf("영화2 (로맨스코미디) 유사도: %.4f%n", sim2);
        System.out.printf("영화3 (혼합) 유사도: %.4f%n", sim3);

        // 가장 높은 유사도 찾기
        String recommended = sim1 >= sim2 && sim1 >= sim3 ? "영화1" :
                            sim2 >= sim1 && sim2 >= sim3 ? "영화2" : "영화3";
        System.out.println("→ 추천: " + recommended);
    }

    /**
     * 단어 임베딩 예제
     * Word2Vec 스타일의 단어 관계
     */
    private static void wordEmbeddingDemo() {
        // 가상의 단어 임베딩 (실제로는 수백 차원)
        double[] king = {0.5, 0.8, 0.2, -0.3};
        double[] queen = {0.6, 0.7, 0.3, -0.2};
        double[] man = {0.4, 0.1, 0.5, 0.9};
        double[] woman = {0.5, 0.0, 0.6, 0.8};

        System.out.println("King - Man + Woman ≈ ?");

        // king - man + woman 계산
        double[] result = new double[4];
        for (int i = 0; i < 4; i++) {
            result[i] = king[i] - man[i] + woman[i];
        }

        System.out.println("계산 결과: " + Vector.toString(result));
        System.out.println("Queen 벡터: " + Vector.toString(queen));
        System.out.printf("결과와 Queen 유사도: %.4f%n", Vector.cosineSimilarity(result, queen));
        System.out.println("→ King - Man + Woman ≈ Queen (높은 유사도!)");
    }

    /**
     * 그래디언트 클리핑 예제
     * 그래디언트가 너무 크면 잘라서 학습 안정화
     */
    private static void gradientClippingDemo() {
        // 매우 큰 그래디언트 (폭발 상황)
        double[] gradient = {100.0, -200.0, 150.0};
        double maxNorm = 5.0;

        System.out.println("원본 그래디언트: " + Vector.toString(gradient));
        System.out.printf("그래디언트 크기: %.4f%n", Vector.norm(gradient));
        System.out.println("최대 허용 크기: " + maxNorm);

        // 클리핑
        double gradNorm = Vector.norm(gradient);
        double[] clipped;
        if (gradNorm > maxNorm) {
            clipped = Vector.scale(gradient, maxNorm / gradNorm);
            System.out.println("→ 클리핑 수행!");
        } else {
            clipped = gradient.clone();
            System.out.println("→ 클리핑 불필요");
        }

        System.out.println("클리핑 후: " + Vector.toString(clipped));
        System.out.printf("클리핑 후 크기: %.4f%n", Vector.norm(clipped));
    }
}
