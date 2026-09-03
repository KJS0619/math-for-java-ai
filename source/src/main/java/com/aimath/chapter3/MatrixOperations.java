package com.aimath.chapter3;

import com.aimath.core.Matrix;
import com.aimath.core.Vector;

/**
 * 3장 예제: 행렬 연산
 * 행렬 곱셈, 전치, 신경망 순전파 시뮬레이션
 */
public class MatrixOperations {

    public static void main(String[] args) {
        System.out.println("=== 3장: 행렬 연산 ===\n");

        // 1. 행렬 기본 연산
        System.out.println("1. 행렬 덧셈과 스칼라 곱");
        double[][] A = {{1, 2}, {3, 4}};
        double[][] B = {{5, 6}, {7, 8}};

        System.out.println("A:");
        System.out.print(Matrix.toString(A));
        System.out.println("B:");
        System.out.print(Matrix.toString(B));

        double[][] sum = Matrix.add(A, B);
        System.out.println("A + B:");
        System.out.print(Matrix.toString(sum));

        double[][] scaled = Matrix.scale(A, 0.5);
        System.out.println("0.5 * A:");
        System.out.print(Matrix.toString(scaled));

        // 2. 행렬 곱셈
        System.out.println("2. 행렬 곱셈 (2x3) × (3x2) = (2x2)");
        double[][] M1 = {
            {1, 2, 3},
            {4, 5, 6}
        };
        double[][] M2 = {
            {7, 8},
            {9, 10},
            {11, 12}
        };

        System.out.println("M1 (2×3):");
        System.out.print(Matrix.toString(M1));
        System.out.println("M2 (3×2):");
        System.out.print(Matrix.toString(M2));

        double[][] product = Matrix.multiply(M1, M2);
        System.out.println("M1 × M2:");
        System.out.print(Matrix.toString(product));
        System.out.println("C[0][0] = 1×7 + 2×9 + 3×11 = 58");
        System.out.println();

        // 3. 행렬-벡터 곱셈
        System.out.println("3. 행렬-벡터 곱셈");
        double[][] W = {
            {0.2, 0.4, 0.1},
            {0.3, 0.5, 0.2}
        };
        double[] x = {1.0, 2.0, 3.0};

        System.out.println("가중치 W (2×3):");
        System.out.print(Matrix.toString(W));
        System.out.println("입력 x: " + Vector.toString(x));

        double[] y = Matrix.multiplyVector(W, x);
        System.out.println("W × x = " + Vector.toString(y));
        System.out.println();

        // 4. 전치 행렬
        System.out.println("4. 전치 행렬");
        double[][] original = {
            {1, 2, 3},
            {4, 5, 6}
        };
        System.out.println("원본 (2×3):");
        System.out.print(Matrix.toString(original));

        double[][] transposed = Matrix.transpose(original);
        System.out.println("전치 (3×2):");
        System.out.print(Matrix.toString(transposed));
        System.out.println("행과 열이 바뀜: A[i][j] → A^T[j][i]");
        System.out.println();

        // 5. AI 활용: 신경망 순전파
        System.out.println("=== AI 활용: 신경망 순전파 ===");
        forwardPassDemo();
        System.out.println();

        // 6. AI 활용: 배치 처리
        System.out.println("=== AI 활용: 배치 처리 ===");
        batchProcessingDemo();
    }

    /**
     * 신경망 순전파 데모
     * 입력층(3) → 은닉층(4) → 출력층(2)
     */
    private static void forwardPassDemo() {
        // 입력: 3차원
        double[] input = {0.5, 0.8, 0.3};

        // 첫 번째 층 가중치 (4×3) + 편향
        double[][] W1 = {
            {0.1, 0.2, 0.3},
            {0.4, 0.5, 0.6},
            {0.7, 0.8, 0.9},
            {0.2, 0.3, 0.4}
        };
        double[] b1 = {0.1, 0.1, 0.1, 0.1};

        // 두 번째 층 가중치 (2×4) + 편향
        double[][] W2 = {
            {0.1, 0.2, 0.3, 0.4},
            {0.5, 0.6, 0.7, 0.8}
        };
        double[] b2 = {0.1, 0.1};

        System.out.println("입력: " + Vector.toString(input));
        System.out.println();

        // 순전파 1: z1 = W1 × input + b1
        System.out.println("Layer 1: z = W1 × input + b1");
        double[] z1 = Matrix.multiplyVector(W1, input);
        double[] h1 = Vector.add(z1, b1);
        System.out.println("z1 = " + Vector.toString(z1));
        System.out.println("h1 (+ bias) = " + Vector.toString(h1));

        // ReLU 활성화
        double[] a1 = relu(h1);
        System.out.println("a1 (ReLU) = " + Vector.toString(a1));
        System.out.println();

        // 순전파 2: z2 = W2 × a1 + b2
        System.out.println("Layer 2: z = W2 × a1 + b2");
        double[] z2 = Matrix.multiplyVector(W2, a1);
        double[] output = Vector.add(z2, b2);
        System.out.println("z2 = " + Vector.toString(z2));
        System.out.println("output = " + Vector.toString(output));
    }

    /**
     * 배치 처리 데모
     * 여러 샘플을 한번에 행렬 곱셈으로 처리
     */
    private static void batchProcessingDemo() {
        // 배치 입력: 3개 샘플, 각 2차원
        double[][] batch = {
            {1.0, 2.0},
            {3.0, 4.0},
            {5.0, 6.0}
        };

        // 가중치: 입력 2차원 → 출력 3차원
        double[][] W = {
            {0.1, 0.2},
            {0.3, 0.4},
            {0.5, 0.6}
        };

        System.out.println("배치 입력 (3 샘플 × 2 특성):");
        System.out.print(Matrix.toString(batch));

        System.out.println("가중치 (3 출력 × 2 입력):");
        System.out.print(Matrix.toString(W));

        // 배치 처리: (3×2) × (2×3)^T = (3×3)
        // 실제로는 batch × W^T
        double[][] Wt = Matrix.transpose(W);
        double[][] output = Matrix.multiply(batch, Wt);

        System.out.println("배치 출력 (3 샘플 × 3 출력):");
        System.out.print(Matrix.toString(output));
        System.out.println("→ 3개 샘플이 동시에 처리됨 (GPU 병렬화에 유리)");
    }

    /**
     * ReLU 활성화 함수
     */
    private static double[] relu(double[] v) {
        double[] result = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            result[i] = Math.max(0, v[i]);
        }
        return result;
    }
}
