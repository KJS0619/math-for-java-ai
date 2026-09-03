# 4장. 함수와 그래프 — 입력과 출력의 관계

## 4.1 함수란 무엇인가

### 수학의 함수 vs Java 메서드

함수는 **입력을 받아 출력을 내보내는 규칙**입니다.

$$f(x) = x^2$$

Java 개발자에게 이것은 메서드와 같습니다:

```java
// 수학 함수: f(x) = x²
public static double f(double x) {
    return x * x;
}

// 사용
double y = f(3);  // 9
```

### 함수의 핵심 특성

**1. 결정론적 (Deterministic)**
```java
// 같은 입력 → 항상 같은 출력
f(3) == 9  // 항상 참
f(3) == 9  // 몇 번을 호출해도 9
```

**2. 순수 함수 (Pure Function)**
```java
// 좋은 예: 외부 상태에 의존하지 않음
public static double square(double x) {
    return x * x;
}

// 나쁜 예: 외부 상태에 의존
private static int counter = 0;
public static int impureFunction(int x) {
    counter++;  // 부작용!
    return x + counter;
}
```

### 다변수 함수

입력이 여러 개인 함수:

$$f(x, y) = x^2 + y^2$$

```java
public static double f(double x, double y) {
    return x * x + y * y;
}

// 또는 벡터 입력
public static double f(double[] inputs) {
    double sum = 0;
    for (double x : inputs) {
        sum += x * x;
    }
    return sum;
}
```

---

## 4.2 선형 함수: y = ax + b

### 선형 함수의 정의

$$f(x) = ax + b$$

- **a**: 기울기 (slope) - 변화율
- **b**: y절편 (intercept) - x=0일 때의 값

```java
public class LinearFunction {
    private double a;  // 기울기
    private double b;  // 절편

    public LinearFunction(double a, double b) {
        this.a = a;
        this.b = b;
    }

    public double apply(double x) {
        return a * x + b;
    }
}

// 사용
LinearFunction f = new LinearFunction(2, 3);  // f(x) = 2x + 3
double y = f.apply(5);  // 2*5 + 3 = 13
```

### 다차원 선형 함수

$$f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b = \sum_{i} w_i x_i + b$$

```java
public static double linearMultiDim(double[] w, double[] x, double b) {
    return Vector.dot(w, x) + b;
}

// 이것이 바로 선형 회귀!
double[] weights = {0.5, 0.3, 0.2};
double[] features = {10, 20, 30};
double bias = 1.0;

double prediction = linearMultiDim(weights, features, bias);
// 0.5*10 + 0.3*20 + 0.2*30 + 1.0 = 5 + 6 + 6 + 1 = 18
```

### 선형 함수의 한계

선형 함수만으로는 **XOR 문제**를 풀 수 없습니다:

```
입력 (x1, x2) → 출력
(0, 0) → 0
(0, 1) → 1
(1, 0) → 1
(1, 1) → 0

이것을 직선 하나로 분리할 수 없음!
```

---

## 4.3 지수와 로그 함수

활성화 함수를 이해하려면 먼저 **지수 함수**와 **로그 함수**를 알아야 합니다.

### 지수 함수: $e^x$

**오일러 상수 e**

$$e \approx 2.71828...$$

$e$는 자연 상수라고 불리며, 원주율 $\pi$처럼 수학에서 자연스럽게 등장하는 상수입니다.

```java
// Java에서 e
System.out.println(Math.E);  // 2.718281828459045
```

**$e$의 정의 (복리 이자에서 유래)**

1원을 연이율 100%로 n번 복리 계산하면:

$$\left(1 + \frac{1}{n}\right)^n$$

$n$을 무한대로 보내면 $e$가 됩니다:

```java
// n이 커질수록 e에 가까워짐
for (int n : new int[]{1, 10, 100, 1000, 10000, 100000}) {
    double result = Math.pow(1 + 1.0/n, n);
    System.out.printf("n=%6d: %.10f%n", n, result);
}
// n=     1: 2.0000000000
// n=    10: 2.5937424601
// n=   100: 2.7048138294
// n=  1000: 2.7169239322
// n= 10000: 2.7181459268
// n=100000: 2.7182682372  ← e에 수렴
```

**지수 함수의 특징**

```java
public static double exp(double x) {
    return Math.exp(x);  // e^x
}

// 기본 성질
exp(0);   // 1.0 (e^0 = 1)
exp(1);   // 2.718... (e^1 = e)
exp(2);   // 7.389... (e^2)
exp(-1);  // 0.368... (e^-1 = 1/e)

// 핵심 성질: 곱셈 → 덧셈
exp(a) * exp(b) == exp(a + b);  // e^a × e^b = e^(a+b)
```

**지수 함수 그래프**

```java
// 지수 함수는 항상 양수, 급격히 증가
for (double x = -3; x <= 3; x += 0.5) {
    double y = Math.exp(x);
    int bars = (int)(y * 2);
    System.out.printf("x=%5.1f: %8.4f |%s%n", x, y, "█".repeat(Math.min(bars, 50)));
}
// x= -3.0:   0.0498 |
// x= -2.0:   0.1353 |
// x= -1.0:   0.3679 |
// x=  0.0:   1.0000 |██
// x=  1.0:   2.7183 |█████
// x=  2.0:   7.3891 |██████████████
// x=  3.0:  20.0855 |████████████████████████████████████████
```

### 로그 함수: $\ln x$

로그는 **지수의 역함수**입니다.

$$y = e^x \quad \Leftrightarrow \quad x = \ln y$$

"$e$를 몇 제곱하면 $x$가 되는가?"

```java
public static double ln(double x) {
    return Math.log(x);  // 자연로그 (밑이 e)
}

// 기본 값
ln(1);        // 0.0 (e^0 = 1이므로)
ln(Math.E);   // 1.0 (e^1 = e이므로)
ln(Math.E * Math.E);  // 2.0 (e^2)
ln(0.5);      // -0.693... (음수!)

// 정의역: x > 0 (양수만 가능)
ln(0);        // -Infinity
ln(-1);       // NaN (정의 안됨)
```

**로그의 핵심 성질: 곱셈을 덧셈으로**

```java
double a = 10, b = 5;

// 곱셈 → 덧셈
ln(a * b) == ln(a) + ln(b);  // ln(50) = ln(10) + ln(5)

// 나눗셈 → 뺄셈
ln(a / b) == ln(a) - ln(b);  // ln(2) = ln(10) - ln(5)

// 거듭제곱 → 곱셈
ln(Math.pow(a, 3)) == 3 * ln(a);  // ln(1000) = 3 × ln(10)
```

이 성질이 왜 중요할까요? AI에서 확률의 곱을 다룰 때:

```java
// 확률의 곱 (언더플로우 위험!)
double p1 = 0.001, p2 = 0.002, p3 = 0.0005;
double product = p1 * p2 * p3;  // 1e-12 → 매우 작은 수, 언더플로우!

// 로그 확률의 합 (안전!)
double logSum = ln(p1) + ln(p2) + ln(p3);  // -6.9 + -6.2 + -7.6 = -20.7
// 필요하면 다시 exp(logSum)으로 복원
```

### 왜 AI에서 $e$를 쓸까?

**이유 1: 미분이 자기 자신**

$$\frac{d}{dx}e^x = e^x$$

지수 함수를 미분해도 형태가 변하지 않습니다!

```java
// e^x의 미분값 = e^x 그 자체
double x = 2.0;
double expX = Math.exp(x);        // 7.389...
double derivative = Math.exp(x);  // 미분값도 7.389... (같다!)
```

이 성질 덕분에 경사하강법에서 그래디언트 계산이 깔끔해집니다.

**이유 2: 양수 보장**

$e^x > 0$ (항상 양수)

확률은 항상 0 이상이어야 하므로, 어떤 값이든 $e^x$를 취하면 양수가 됩니다.

```java
// 아무 실수나 넣어도 양수가 나옴
exp(-100);  // 3.7e-44 (아주 작지만 양수)
exp(0);     // 1.0
exp(100);   // 2.7e+43 (아주 크지만 양수)
```

**이유 3: Softmax에서 확률 분포 생성**

```java
// 점수(logits)를 확률로 변환
double[] logits = {2.0, 1.0, 0.1};

// exp를 취하면 모두 양수
double[] exps = {exp(2.0), exp(1.0), exp(0.1)};
// = {7.39, 2.72, 1.11}

// 합으로 나누면 확률 (합=1)
double sum = 7.39 + 2.72 + 1.11;  // 11.22
double[] probs = {7.39/sum, 2.72/sum, 1.11/sum};
// = {0.659, 0.242, 0.099}  ← 확률 분포!
```

### Sigmoid와 지수 함수

Sigmoid 함수는 지수 함수로 만들어집니다:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

```java
public static double sigmoid(double x) {
    return 1.0 / (1.0 + Math.exp(-x));
}

// x가 크면 → e^(-x) ≈ 0 → 1/(1+0) = 1
// x가 0이면 → e^0 = 1 → 1/(1+1) = 0.5
// x가 작으면 → e^(-x) ≈ ∞ → 1/(1+∞) = 0

sigmoid(10);   // 0.9999...  ≈ 1
sigmoid(0);    // 0.5
sigmoid(-10);  // 0.0000...  ≈ 0
```

### Cross-Entropy와 로그

손실 함수에서 로그를 쓰는 이유:

```java
// 정답이 1인데 예측이 0.9면?
// → 잘한 것 → 손실 작아야 함
-ln(0.9);   // 0.105 (작은 손실)

// 정답이 1인데 예측이 0.1이면?
// → 못한 것 → 손실 커야 함
-ln(0.1);   // 2.303 (큰 손실)

// 정답이 1인데 예측이 0.0001이면?
// → 완전 틀림 → 손실 매우 커야 함
-ln(0.0001);  // 9.21 (매우 큰 손실)
```

로그의 특성상 확률이 작을수록 손실이 급격히 커집니다 → 틀린 예측에 큰 페널티!

### 지수/로그 함수 정리

```java
public class ExpLogFunctions {
    // 지수 함수
    public static double exp(double x) {
        return Math.exp(x);
    }

    // 자연로그
    public static double ln(double x) {
        return Math.log(x);
    }

    // 역함수 관계 확인
    public static void verifyInverse() {
        double x = 3.0;
        System.out.println(ln(exp(x)));  // 3.0 (원래대로)
        System.out.println(exp(ln(x)));  // 3.0 (원래대로)
    }
}
```

| 함수 | 정의역 | 치역 | 핵심 성질 |
|-----|-------|------|----------|
| $e^x$ | 모든 실수 | $(0, \infty)$ | 미분 = 자기 자신 |
| $\ln x$ | $(0, \infty)$ | 모든 실수 | 곱셈 → 덧셈 |

---

## 4.4 비선형 함수의 필요성

### 왜 비선형이 필요한가?

**선형 함수의 합성은 여전히 선형**:

$$f(x) = 2x + 1$$
$$g(x) = 3x + 2$$
$$g(f(x)) = 3(2x + 1) + 2 = 6x + 5$$

아무리 많은 선형 레이어를 쌓아도 결국 하나의 선형 함수!

```java
// 레이어 1: y1 = W1 * x + b1
// 레이어 2: y2 = W2 * y1 + b2
//         = W2 * (W1 * x + b1) + b2
//         = (W2 * W1) * x + (W2 * b1 + b2)
//         = W' * x + b'  ← 결국 선형!
```

### 비선형 활성화 함수 추가

```java
// 비선형 함수를 사이에 넣으면:
// y1 = σ(W1 * x + b1)  ← 비선형!
// y2 = σ(W2 * y1 + b2)
// 이제 더 복잡한 패턴 학습 가능
```

---

## 4.5 활성화 함수들

### Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

출력 범위: (0, 1) - 확률로 해석 가능

```java
public static double sigmoid(double x) {
    return 1.0 / (1.0 + Math.exp(-x));
}

// 특징
sigmoid(0);    // 0.5
sigmoid(10);   // ≈ 1.0
sigmoid(-10);  // ≈ 0.0
```

**용도**: 이진 분류의 출력층

**단점**:
- 그래디언트 소실 (vanishing gradient)
- 출력이 0 중심이 아님

### Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

출력 범위: (-1, 1)

```java
public static double tanh(double x) {
    return Math.tanh(x);
}

// 특징
tanh(0);   // 0 (중심이 0!)
tanh(2);   // ≈ 0.96
tanh(-2);  // ≈ -0.96
```

**장점**: Sigmoid보다 중심이 0이라 학습에 유리

### ReLU (Rectified Linear Unit)

$$\text{ReLU}(x) = \max(0, x)$$

```java
public static double relu(double x) {
    return Math.max(0, x);
}

// 특징
relu(-5);  // 0
relu(0);   // 0
relu(5);   // 5
```

**장점**:
- 계산이 빠름
- 그래디언트 소실 완화
- 현대 딥러닝의 기본 활성화 함수

**단점**: Dying ReLU (음수 입력에서 영원히 0)

### Leaky ReLU

$$\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$$

```java
public static double leakyRelu(double x, double alpha) {
    return x > 0 ? x : alpha * x;
}

// 보통 alpha = 0.01
leakyRelu(-5, 0.01);  // -0.05 (완전히 죽지 않음)
```

### Softmax

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$

출력의 합이 1이 되는 확률 분포로 변환:

```java
public static double[] softmax(double[] x) {
    // 수치 안정성을 위해 최대값 빼기
    double max = Arrays.stream(x).max().getAsDouble();

    double[] exp = new double[x.length];
    double sum = 0;
    for (int i = 0; i < x.length; i++) {
        exp[i] = Math.exp(x[i] - max);
        sum += exp[i];
    }

    double[] result = new double[x.length];
    for (int i = 0; i < x.length; i++) {
        result[i] = exp[i] / sum;
    }
    return result;
}

// 사용 예
double[] logits = {2.0, 1.0, 0.1};
double[] probs = softmax(logits);
// [0.659, 0.242, 0.099] - 합이 1.0
```

**용도**: 다중 클래스 분류의 출력층

### 활성화 함수 비교

| 함수 | 범위 | 장점 | 단점 | 용도 |
|-----|-----|-----|-----|-----|
| Sigmoid | (0,1) | 확률 해석 | 그래디언트 소실 | 이진 분류 출력 |
| Tanh | (-1,1) | 0 중심 | 그래디언트 소실 | 순환 신경망 |
| ReLU | [0,∞) | 빠름, 소실 완화 | Dying ReLU | 은닉층 기본 |
| Softmax | (0,1), 합=1 | 확률 분포 | - | 다중 분류 출력 |

---

## 4.6 함수 합성

### 합성 함수

두 함수를 연결하여 새로운 함수를 만듭니다:

$$(g \circ f)(x) = g(f(x))$$

```java
// f(x) = 2x + 1
// g(x) = x²
// (g ∘ f)(x) = (2x + 1)²

public static double f(double x) { return 2 * x + 1; }
public static double g(double x) { return x * x; }

public static double composed(double x) {
    return g(f(x));  // 먼저 f, 그 다음 g
}

composed(3);  // f(3)=7, g(7)=49
```

### Java 함수형 인터페이스로 합성

```java
import java.util.function.DoubleUnaryOperator;

public class FunctionComposition {
    public static void main(String[] args) {
        DoubleUnaryOperator f = x -> 2 * x + 1;
        DoubleUnaryOperator g = x -> x * x;

        // g(f(x))
        DoubleUnaryOperator gof = f.andThen(g);
        System.out.println(gof.applyAsDouble(3));  // 49

        // f(g(x))
        DoubleUnaryOperator fog = f.compose(g);
        System.out.println(fog.applyAsDouble(3));  // 2*9+1 = 19
    }
}
```

---

## 4.7 [AI 연결] 신경망 = 함수들의 합성

### 신경망의 수학적 표현

신경망은 함수의 합성입니다:

$$\mathbf{y} = f_3(f_2(f_1(\mathbf{x})))$$

각 레이어 함수:
$$f_i(\mathbf{x}) = \sigma(\mathbf{W}_i \mathbf{x} + \mathbf{b}_i)$$

```java
public class NeuralNetwork {
    private double[][] W1, W2, W3;
    private double[] b1, b2, b3;

    public double[] forward(double[] x) {
        // f1: 선형 변환 + 활성화
        double[] z1 = Vector.add(Matrix.multiplyVector(W1, x), b1);
        double[] a1 = Activation.relu(z1);

        // f2: 선형 변환 + 활성화
        double[] z2 = Vector.add(Matrix.multiplyVector(W2, a1), b2);
        double[] a2 = Activation.relu(z2);

        // f3: 선형 변환 + softmax
        double[] z3 = Vector.add(Matrix.multiplyVector(W3, a2), b3);
        double[] output = Activation.softmax(z3);

        return output;
    }
}
```

### 범용 근사 정리 (Universal Approximation Theorem)

**충분히 넓은 은닉층을 가진 신경망은 어떤 연속 함수든 근사할 수 있다.**

```java
// 이론적으로, 이 신경망은 어떤 함수든 흉내낼 수 있음
// (충분히 큰 hiddenSize와 적절한 학습이 있다면)

double[][] W1 = new double[hiddenSize][inputSize];
double[] b1 = new double[hiddenSize];
double[][] W2 = new double[outputSize][hiddenSize];
double[] b2 = new double[outputSize];
```

### 활성화 함수 선택 가이드

```java
public class LayerBuilder {
    public static double[] hiddenLayer(double[] input, double[][] W, double[] b) {
        double[] z = Vector.add(Matrix.multiplyVector(W, input), b);
        return Activation.relu(z);  // 은닉층: ReLU
    }

    public static double[] outputLayerClassification(double[] input, double[][] W, double[] b) {
        double[] z = Vector.add(Matrix.multiplyVector(W, input), b);
        return Activation.softmax(z);  // 분류: Softmax
    }

    public static double[] outputLayerRegression(double[] input, double[][] W, double[] b) {
        // 회귀: 활성화 없음 (선형)
        return Vector.add(Matrix.multiplyVector(W, input), b);
    }

    public static double[] outputLayerBinary(double[] input, double[][] W, double[] b) {
        double[] z = Vector.add(Matrix.multiplyVector(W, input), b);
        return Activation.sigmoid(z);  // 이진 분류: Sigmoid
    }
}
```

---

## 연습 문제

### 1. 활성화 함수 구현

다음 활성화 함수를 구현하세요:

```java
// ELU (Exponential Linear Unit)
// f(x) = x if x > 0, else α(e^x - 1)
public static double elu(double x, double alpha) {
    // 구현하세요
}

// GELU (Gaussian Error Linear Unit) - GPT에서 사용
// f(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
public static double gelu(double x) {
    // 구현하세요
}
```

### 2. 함수 시각화

Sigmoid, Tanh, ReLU 함수를 -5~5 범위에서 출력하세요:

```java
public static void plotFunctions() {
    for (double x = -5; x <= 5; x += 0.5) {
        System.out.printf("x=%.1f: sig=%.3f, tanh=%.3f, relu=%.3f%n",
            x, sigmoid(x), tanh(x), relu(x));
    }
}
```

### 3. 간단한 신경망

```java
// 2-3-2 신경망 (입력2, 은닉3, 출력2)
// XOR 문제를 풀 수 있는지 테스트
public class XORNetwork {
    // 가중치 초기화, forward 구현
    // 테스트: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0
}
```

---

## 정리

| 개념 | 설명 | AI 활용 |
|-----|------|--------|
| 선형 함수 | y = ax + b | 선형 회귀, 레이어 기본 |
| 비선형 함수 | 곡선 관계 | 활성화 함수 |
| Sigmoid | (0,1) 출력 | 이진 분류 출력 |
| ReLU | max(0,x) | 은닉층 기본 |
| Softmax | 확률 분포 | 다중 분류 출력 |
| 함수 합성 | g(f(x)) | 신경망 = 합성 |

**핵심 포인트**:
1. 신경망 = 선형 변환 + 비선형 활성화의 합성
2. 비선형 활성화 없이는 복잡한 패턴 학습 불가
3. 은닉층은 ReLU, 출력층은 문제에 따라 선택

---

## 다음 장 예고

5장에서는 미분을 배웁니다:
- 순간 변화율과 도함수
- 연쇄 법칙 (Chain Rule) - 역전파의 핵심
- 편미분과 그래디언트
