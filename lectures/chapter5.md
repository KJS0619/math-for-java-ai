# 5장. 미분 — 변화를 측정하다

## 5.1 순간 변화율의 개념

### 평균 변화율

두 점 사이의 기울기입니다.

$$\frac{\Delta y}{\Delta x} = \frac{f(x + h) - f(x)}{h}$$

```java
public static double averageRateOfChange(
    DoubleUnaryOperator f, double x, double h) {
    return (f.applyAsDouble(x + h) - f.applyAsDouble(x)) / h;
}

// f(x) = x² 에서 x=2, h=1
DoubleUnaryOperator f = x -> x * x;
double avg = averageRateOfChange(f, 2, 1);
// (f(3) - f(2)) / 1 = (9 - 4) / 1 = 5
```

### 순간 변화율 (미분)

h를 0에 가깝게 줄이면 **순간 변화율**이 됩니다.

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

```java
// 수치 미분: h를 아주 작게
public static double numericalDerivative(
    DoubleUnaryOperator f, double x) {
    double h = 1e-7;  // 0.0000001
    return (f.applyAsDouble(x + h) - f.applyAsDouble(x)) / h;
}

// f(x) = x² 에서 x=2
double derivative = numericalDerivative(x -> x * x, 2);
// ≈ 4.0 (해석적 해: f'(x) = 2x, f'(2) = 4)
```

### 중앙 차분법 (더 정확)

```java
// 중앙 차분: (f(x+h) - f(x-h)) / 2h
public static double centralDifference(
    DoubleUnaryOperator f, double x) {
    double h = 1e-5;
    return (f.applyAsDouble(x + h) - f.applyAsDouble(x - h)) / (2 * h);
}
```

---

## 5.2 도함수 (Derivative)

### 도함수란?

원래 함수의 **기울기를 나타내는 새로운 함수**입니다.

- $f(x) = x^2$ → $f'(x) = 2x$
- $f(x) = x^3$ → $f'(x) = 3x^2$

### 기본 미분 공식

| 함수 | 도함수 |
|-----|-------|
| $c$ (상수) | $0$ |
| $x$ | $1$ |
| $x^n$ | $n \cdot x^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln(x)$ | $1/x$ |
| $\sin(x)$ | $\cos(x)$ |
| $\cos(x)$ | $-\sin(x)$ |

```java
// 해석적 미분 (손으로 계산한 결과 코딩)
public class Derivatives {
    // f(x) = x² → f'(x) = 2x
    public static double squareDerivative(double x) {
        return 2 * x;
    }

    // f(x) = x³ → f'(x) = 3x²
    public static double cubeDerivative(double x) {
        return 3 * x * x;
    }

    // f(x) = e^x → f'(x) = e^x
    public static double expDerivative(double x) {
        return Math.exp(x);
    }
}
```

### 미분 법칙

**덧셈 법칙**: $(f + g)' = f' + g'$

**곱셈 법칙**: $(fg)' = f'g + fg'$

**나눗셈 법칙**: $(f/g)' = (f'g - fg') / g^2$

```java
// 예: f(x) = x² + 3x + 2
// f'(x) = 2x + 3

public static double polyDerivative(double x) {
    // d/dx (x² + 3x + 2) = 2x + 3
    return 2 * x + 3;
}
```

---

## 5.3 활성화 함수의 도함수

### Sigmoid의 도함수

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
$$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$

```java
public static double sigmoidDerivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

// 특징: 최대값이 0.25 (x=0에서)
// → 그래디언트가 항상 작아짐 (소실 문제)
```

### Tanh의 도함수

$$\tanh'(x) = 1 - \tanh^2(x)$$

```java
public static double tanhDerivative(double x) {
    double t = Math.tanh(x);
    return 1 - t * t;
}
```

### ReLU의 도함수

$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

```java
public static double reluDerivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

// 특징: 도함수가 0 또는 1 (단순!)
// → 그래디언트가 그대로 전파됨 (소실 완화)
```

---

## 5.4 연쇄 법칙 (Chain Rule)

### 연쇄 법칙이란?

합성 함수의 미분 법칙입니다.

$$\frac{d}{dx}[g(f(x))] = g'(f(x)) \cdot f'(x)$$

### 직관적 이해

```
x → [f] → f(x) → [g] → g(f(x))

x가 조금 변하면:
- f(x)는 f'(x)만큼 변하고
- 그 변화가 g를 통과하면 g'(f(x))만큼 증폭됨
- 총 변화: f'(x) × g'(f(x))
```

### Java 구현

```java
// y = (x² + 1)³ 의 미분
// y = g(f(x)), where f(x) = x² + 1, g(u) = u³

// f'(x) = 2x
// g'(u) = 3u²

// dy/dx = g'(f(x)) * f'(x)
//       = 3(x² + 1)² * 2x
//       = 6x(x² + 1)²

public static double chainRuleExample(double x) {
    double f = x * x + 1;      // f(x)
    double fPrime = 2 * x;     // f'(x)
    double gPrime = 3 * f * f; // g'(f(x)) = 3(f(x))²
    return gPrime * fPrime;    // 연쇄 법칙
}
```

### 신경망에서의 연쇄 법칙

```
입력 x → [W1, b1] → z1 → [ReLU] → a1 → [W2, b2] → z2 → [Softmax] → y → [Loss] → L

역전파: 오른쪽에서 왼쪽으로 그래디언트 전파
dL/dW1 = dL/dy × dy/dz2 × dz2/da1 × da1/dz1 × dz1/dW1
         ↑       ↑        ↑         ↑         ↑
      연쇄 법칙의 연속 적용!
```

---

## 5.5 편미분 (Partial Derivative)

### 편미분이란?

다변수 함수에서 **하나의 변수만** 미분하고 나머지는 상수 취급합니다.

$$f(x, y) = x^2 + xy + y^2$$

$$\frac{\partial f}{\partial x} = 2x + y$$

$$\frac{\partial f}{\partial y} = x + 2y$$

```java
// f(x, y) = x² + xy + y²

// ∂f/∂x = 2x + y (y는 상수 취급)
public static double partialX(double x, double y) {
    return 2 * x + y;
}

// ∂f/∂y = x + 2y (x는 상수 취급)
public static double partialY(double x, double y) {
    return x + 2 * y;
}

// 수치적 편미분
public static double numericalPartialX(
    BiFunction<Double, Double, Double> f, double x, double y) {
    double h = 1e-5;
    return (f.apply(x + h, y) - f.apply(x - h, y)) / (2 * h);
}
```

### AI에서의 편미분

손실 함수 L이 가중치 W1, W2, ..., Wn에 의존할 때:

$$L(W_1, W_2, ..., W_n)$$

각 가중치에 대한 편미분:
$$\frac{\partial L}{\partial W_1}, \frac{\partial L}{\partial W_2}, ..., \frac{\partial L}{\partial W_n}$$

이것들이 **그래디언트**!

---

## 5.6 그래디언트 (Gradient)

### 그래디언트란?

모든 편미분을 모아 놓은 벡터입니다.

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

```java
// f(x, y) = x² + y² 의 그래디언트
// ∇f = [2x, 2y]

public static double[] gradient(double x, double y) {
    return new double[] { 2 * x, 2 * y };
}

// 일반적인 수치 그래디언트
public static double[] numericalGradient(
    Function<double[], Double> f, double[] point) {

    double h = 1e-5;
    int n = point.length;
    double[] grad = new double[n];

    for (int i = 0; i < n; i++) {
        double[] pointPlus = point.clone();
        double[] pointMinus = point.clone();
        pointPlus[i] += h;
        pointMinus[i] -= h;

        grad[i] = (f.apply(pointPlus) - f.apply(pointMinus)) / (2 * h);
    }

    return grad;
}
```

### 그래디언트의 기하학적 의미

**그래디언트는 함수가 가장 가파르게 증가하는 방향**을 가리킵니다.

```
함수 f(x,y) = x² + y² (볼록한 그릇 모양)

     ↗ ∇f (가장 가파른 상승 방향)
    /
   •
  /
 ↙ -∇f (가장 가파른 하강 방향 = 최솟값 향함)
```

**결론**: 손실을 줄이려면 **-∇L 방향**으로 가중치를 업데이트!

---

## 5.7 [AI 연결] 역전파의 수학적 기초

### 역전파란?

연쇄 법칙을 사용하여 **각 가중치에 대한 손실의 그래디언트**를 계산하는 알고리즘입니다.

### 간단한 예제

```
x → [×w] → z → [σ] → a → [Loss] → L

순전파:
z = w * x
a = σ(z)
L = (a - y)²

역전파:
dL/da = 2(a - y)
da/dz = σ'(z) = σ(z)(1 - σ(z))
dz/dw = x

dL/dw = dL/da × da/dz × dz/dw
      = 2(a - y) × σ(z)(1 - σ(z)) × x
```

```java
public class SimpleBackprop {
    private double w = 0.5;  // 학습할 가중치
    private double learningRate = 0.1;

    public double forward(double x) {
        double z = w * x;
        double a = sigmoid(z);
        return a;
    }

    public void backward(double x, double y) {
        // 순전파
        double z = w * x;
        double a = sigmoid(z);
        double loss = (a - y) * (a - y);

        // 역전파 (연쇄 법칙)
        double dL_da = 2 * (a - y);
        double da_dz = a * (1 - a);  // sigmoid 도함수
        double dz_dw = x;

        double dL_dw = dL_da * da_dz * dz_dw;

        // 가중치 업데이트
        w = w - learningRate * dL_dw;

        System.out.printf("Loss: %.4f, dL/dw: %.4f, w: %.4f%n",
            loss, dL_dw, w);
    }
}
```

### 역전파의 효율성

- **순전파**: 왼쪽에서 오른쪽으로 한 번
- **역전파**: 오른쪽에서 왼쪽으로 한 번
- **모든 가중치의 그래디언트**를 O(n)에 계산 (n = 파라미터 수)

개별적으로 수치 미분하면 O(n²)이므로 역전파가 훨씬 효율적!

---

## 연습 문제

### 1. 도함수 계산

다음 함수의 도함수를 손으로 구하고 Java로 구현하세요:

- $f(x) = 3x^4 - 2x^2 + x - 5$
- $g(x) = e^{2x}$ (연쇄 법칙 사용)
- $h(x) = \ln(x^2 + 1)$ (연쇄 법칙 사용)

### 2. 수치 미분 검증

해석적 미분과 수치 미분 결과가 일치하는지 확인하세요:

```java
// Sigmoid의 도함수 검증
double x = 0.5;
double analytical = sigmoidDerivative(x);
double numerical = centralDifference(Activation::sigmoid, x);
System.out.println(Math.abs(analytical - numerical) < 1e-6);  // true
```

### 3. 그래디언트 계산

$$f(x, y, z) = x^2y + y^2z + z^2x$$

이 함수의 그래디언트를 구하고 (1, 2, 3)에서 계산하세요.

### 4. 간단한 역전파

2층 신경망에서 역전파를 구현하세요:
```java
// x → [W1] → h → [ReLU] → a → [W2] → y
// 손실: MSE
```

---

## 정리

| 개념 | 수식 | AI 활용 |
|-----|------|--------|
| 도함수 | $f'(x) = \lim_{h→0} \frac{f(x+h)-f(x)}{h}$ | 변화율 측정 |
| 연쇄 법칙 | $(g∘f)' = g'(f) \cdot f'$ | 역전파의 핵심 |
| 편미분 | $\frac{\partial f}{\partial x_i}$ | 개별 가중치 영향 |
| 그래디언트 | $\nabla f$ | 손실 최소화 방향 |

**핵심 포인트**:
1. 미분 = 변화율 측정
2. 연쇄 법칙으로 복잡한 합성 함수도 미분 가능
3. 그래디언트의 반대 방향 = 손실 감소 방향

---

## 다음 장 예고

6장에서는 경사하강법을 배웁니다:
- 그래디언트를 사용한 최적화
- 학습률의 중요성
- SGD, Momentum, Adam
