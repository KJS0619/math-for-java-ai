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

## 4.3 비선형 함수의 필요성

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

## 4.4 활성화 함수들

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

## 4.5 함수 합성

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

## 4.6 [AI 연결] 신경망 = 함수들의 합성

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
