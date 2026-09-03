package com.aimath.chapter5;

import com.aimath.core.Activation;

/**
 * 5장 예제: 미분
 * 수치 미분, 해석적 미분, 편미분, 그래디언트
 */
public class DifferentiationDemo {

    public static void main(String[] args) {
        System.out.println("=== 5장: 미분 ===\n");

        // 1. 도함수의 기하학적 의미
        System.out.println("1. 도함수 = 접선의 기울기");
        derivativeMeaning();
        System.out.println();

        // 2. 수치 미분
        System.out.println("2. 수치 미분 (Numerical Differentiation)");
        numericalDiff();
        System.out.println();

        // 3. 활성화 함수의 도함수
        System.out.println("3. 활성화 함수의 도함수");
        activationDerivatives();
        System.out.println();

        // 4. 연쇄 법칙 (Chain Rule)
        System.out.println("4. 연쇄 법칙 - 역전파의 핵심");
        chainRuleDemo();
        System.out.println();

        // 5. 편미분
        System.out.println("5. 편미분 (Partial Derivative)");
        partialDerivativeDemo();
        System.out.println();

        // 6. 그래디언트
        System.out.println("6. 그래디언트 = 편미분의 벡터");
        gradientDemo();
        System.out.println();

        // 7. AI 활용: 역전파 시뮬레이션
        System.out.println("=== AI 활용: 역전파 시뮬레이션 ===");
        backpropDemo();
    }

    private static void derivativeMeaning() {
        System.out.println("f(x) = x² 의 도함수 f'(x) = 2x");
        System.out.println("\n각 점에서의 기울기:");
        for (double x : new double[]{-2, -1, 0, 1, 2}) {
            double slope = 2 * x;  // f'(x) = 2x
            String direction = slope > 0 ? "↗ 증가" : slope < 0 ? "↘ 감소" : "→ 극값";
            System.out.printf("  x=%.0f: 기울기=%.0f (%s)%n", x, slope, direction);
        }
        System.out.println("\n→ 기울기가 0인 곳이 최솟값 (x=0)");
    }

    private static void numericalDiff() {
        System.out.println("수치 미분: f'(x) ≈ [f(x+h) - f(x-h)] / 2h");
        System.out.println("h가 작을수록 정확 (보통 h=1e-5)");

        // f(x) = x² 를 수치 미분
        double x = 3.0;
        double h = 1e-5;

        double f_plus = x + h;
        double f_minus = x - h;
        double numerical = ((f_plus * f_plus) - (f_minus * f_minus)) / (2 * h);
        double analytical = 2 * x;  // 해석적 해

        System.out.printf("\nf(x) = x² at x=%.1f%n", x);
        System.out.printf("  수치 미분: %.6f%n", numerical);
        System.out.printf("  해석적 해: %.6f%n", analytical);
        System.out.printf("  오차: %.10f%n", Math.abs(numerical - analytical));
    }

    private static void activationDerivatives() {
        double x = 0.5;
        System.out.printf("x = %.1f 에서 각 활성화 함수의 도함수:%n%n", x);

        // Sigmoid
        double sigY = Activation.sigmoid(x);
        double sigDeriv = Activation.sigmoidDerivative(x);
        System.out.println("Sigmoid:");
        System.out.printf("  σ(%.1f) = %.4f%n", x, sigY);
        System.out.printf("  σ'(%.1f) = σ(x)(1-σ(x)) = %.4f%n", x, sigDeriv);

        // Tanh
        double tanhY = Activation.tanh(x);
        double tanhDeriv = Activation.tanhDerivative(x);
        System.out.println("\nTanh:");
        System.out.printf("  tanh(%.1f) = %.4f%n", x, tanhY);
        System.out.printf("  tanh'(%.1f) = 1 - tanh²(x) = %.4f%n", x, tanhDeriv);

        // ReLU
        double reluY = Activation.relu(x);
        double reluDeriv = Activation.reluDerivative(x);
        System.out.println("\nReLU:");
        System.out.printf("  ReLU(%.1f) = %.4f%n", x, reluY);
        System.out.printf("  ReLU'(%.1f) = %.1f (x>0이면 1, 아니면 0)%n", x, reluDeriv);
    }

    private static void chainRuleDemo() {
        System.out.println("합성 함수의 미분: [f(g(x))]' = f'(g(x)) · g'(x)");
        System.out.println("\n예: y = σ(2x + 1)");
        System.out.println("  g(x) = 2x + 1  →  g'(x) = 2");
        System.out.println("  f(u) = σ(u)    →  f'(u) = σ(u)(1-σ(u))");
        System.out.println("  dy/dx = σ'(2x+1) · 2");

        double x = 1.0;
        double u = 2 * x + 1;  // g(x)
        double g_prime = 2.0;
        double f_prime = Activation.sigmoidDerivative(u);
        double dy_dx = f_prime * g_prime;

        System.out.printf("\nx=%.1f 에서:%n", x);
        System.out.printf("  u = 2(%.1f) + 1 = %.1f%n", x, u);
        System.out.printf("  σ'(%.1f) = %.4f%n", u, f_prime);
        System.out.printf("  dy/dx = %.4f × %.1f = %.4f%n", f_prime, g_prime, dy_dx);

        System.out.println("\n→ 역전파에서 그래디언트가 체인처럼 연결됨");
    }

    private static void partialDerivativeDemo() {
        System.out.println("f(x, y) = x² + 2xy + y²");
        System.out.println("\n편미분:");
        System.out.println("  ∂f/∂x = 2x + 2y  (y를 상수 취급)");
        System.out.println("  ∂f/∂y = 2x + 2y  (x를 상수 취급)");

        double x = 1.0, y = 2.0;
        System.out.printf("\n(x, y) = (%.1f, %.1f) 에서:%n", x, y);
        System.out.printf("  ∂f/∂x = 2(%.1f) + 2(%.1f) = %.1f%n", x, y, 2*x + 2*y);
        System.out.printf("  ∂f/∂y = 2(%.1f) + 2(%.1f) = %.1f%n", x, y, 2*x + 2*y);

        System.out.println("\n→ 각 가중치가 손실에 미치는 영향 계산에 사용");
    }

    private static void gradientDemo() {
        System.out.println("그래디언트 = 모든 편미분을 모은 벡터");
        System.out.println("∇f = [∂f/∂x, ∂f/∂y, ∂f/∂z, ...]");

        System.out.println("\n예: f(w1, w2) = w1² + w2² (볼록 함수)");
        System.out.println("  ∇f = [2w1, 2w2]");

        double w1 = 3.0, w2 = 4.0;
        double[] gradient = {2 * w1, 2 * w2};

        System.out.printf("\n(w1, w2) = (%.1f, %.1f) 에서:%n", w1, w2);
        System.out.printf("  ∇f = [%.1f, %.1f]%n", gradient[0], gradient[1]);

        // 그래디언트 방향 = 가장 가파르게 증가하는 방향
        double norm = Math.sqrt(gradient[0]*gradient[0] + gradient[1]*gradient[1]);
        System.out.printf("  |∇f| = %.1f (그래디언트 크기)%n", norm);

        System.out.println("\n→ 그래디언트의 반대 방향으로 이동 = 경사 하강");
    }

    private static void backpropDemo() {
        System.out.println("간단한 뉴런의 역전파");
        System.out.println("y = σ(w·x + b), 손실 L = (y - target)²");

        // 순전파
        double x = 2.0;
        double w = 0.5;
        double b = 0.1;
        double target = 1.0;

        double z = w * x + b;
        double y = Activation.sigmoid(z);
        double loss = (y - target) * (y - target);

        System.out.println("\n순전파:");
        System.out.printf("  z = w·x + b = %.1f×%.1f + %.1f = %.2f%n", w, x, b, z);
        System.out.printf("  y = σ(z) = %.4f%n", y);
        System.out.printf("  L = (y - target)² = (%.4f - %.1f)² = %.4f%n", y, target, loss);

        // 역전파 (체인 룰)
        System.out.println("\n역전파 (연쇄 법칙):");

        double dL_dy = 2 * (y - target);
        System.out.printf("  dL/dy = 2(y - target) = %.4f%n", dL_dy);

        double dy_dz = Activation.sigmoidDerivative(z);
        System.out.printf("  dy/dz = σ'(z) = %.4f%n", dy_dz);

        double dz_dw = x;
        double dz_db = 1.0;
        System.out.printf("  dz/dw = x = %.1f%n", dz_dw);
        System.out.printf("  dz/db = 1%n");

        double dL_dw = dL_dy * dy_dz * dz_dw;
        double dL_db = dL_dy * dy_dz * dz_db;

        System.out.println("\n최종 그래디언트:");
        System.out.printf("  dL/dw = %.4f × %.4f × %.1f = %.6f%n",
            dL_dy, dy_dz, dz_dw, dL_dw);
        System.out.printf("  dL/db = %.4f × %.4f × %.1f = %.6f%n",
            dL_dy, dy_dz, dz_db, dL_db);

        // 가중치 업데이트
        double lr = 0.1;
        double w_new = w - lr * dL_dw;
        double b_new = b - lr * dL_db;

        System.out.println("\n가중치 업데이트 (lr=0.1):");
        System.out.printf("  w: %.4f → %.4f%n", w, w_new);
        System.out.printf("  b: %.4f → %.4f%n", b, b_new);
    }
}
