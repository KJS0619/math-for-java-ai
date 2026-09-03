# 3장. 행렬 연산 — 데이터 변환의 핵심

## 3.1 행렬 덧셈과 스칼라 곱

### 행렬 덧셈

같은 Shape의 행렬끼리 더할 수 있습니다. 같은 위치의 원소끼리 더합니다.

```java
public static double[][] add(double[][] A, double[][] B) {
    int rows = A.length;
    int cols = A[0].length;

    if (B.length != rows || B[0].length != cols) {
        throw new IllegalArgumentException("행렬 크기가 다릅니다");
    }

    double[][] result = new double[rows][cols];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
    return result;
}

// 사용 예
double[][] A = {{1, 2}, {3, 4}};
double[][] B = {{5, 6}, {7, 8}};
double[][] C = add(A, B);  // {{6, 8}, {10, 12}}
```

### 스칼라 곱

행렬의 모든 원소에 같은 스칼라를 곱합니다.

```java
public static double[][] scale(double[][] A, double scalar) {
    int rows = A.length;
    int cols = A[0].length;

    double[][] result = new double[rows][cols];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = A[i][j] * scalar;
        }
    }
    return result;
}

// 사용 예: 가중치에 학습률 곱하기
double[][] gradients = {{0.1, 0.2}, {0.3, 0.4}};
double learningRate = 0.01;
double[][] scaledGrad = scale(gradients, learningRate);
```

---

## 3.2 행렬 곱셈

### 행렬 곱셈의 정의

행렬 A (m×n)과 행렬 B (n×p)의 곱 C = A×B는 (m×p) 행렬입니다.

$$C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}$$

**중요**: A의 열 수 = B의 행 수 여야 곱셈 가능!

```
A: (m × n)  ×  B: (n × p)  =  C: (m × p)
      ↑____________↑
       이 둘이 같아야 함
```

### Java 구현

```java
public static double[][] multiply(double[][] A, double[][] B) {
    int m = A.length;      // A의 행 수
    int n = A[0].length;   // A의 열 수
    int p = B[0].length;   // B의 열 수

    if (B.length != n) {
        throw new IllegalArgumentException(
            "곱셈 불가: A의 열 수(" + n + ") ≠ B의 행 수(" + B.length + ")"
        );
    }

    double[][] C = new double[m][p];

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    return C;
}
```

### 행렬 곱셈 시각화

```
A (2×3)         B (3×2)         C (2×2)
[a b c]         [g h]           [ag+bi+ck  ah+bj+cl]
[d e f]    ×    [i j]     =     [dg+ei+fk  dh+ej+fl]
                [k l]
```

C[0][0]을 구하려면:
- A의 0번 행: [a, b, c]
- B의 0번 열: [g, i, k]
- 내적: a×g + b×i + c×k

### 계산 예시

```java
double[][] A = {
    {1, 2, 3},
    {4, 5, 6}
};  // (2×3)

double[][] B = {
    {7, 8},
    {9, 10},
    {11, 12}
};  // (3×2)

double[][] C = multiply(A, B);
// C[0][0] = 1×7 + 2×9 + 3×11 = 7 + 18 + 33 = 58
// C[0][1] = 1×8 + 2×10 + 3×12 = 8 + 20 + 36 = 64
// C[1][0] = 4×7 + 5×9 + 6×11 = 28 + 45 + 66 = 139
// C[1][1] = 4×8 + 5×10 + 6×12 = 32 + 50 + 72 = 154

// 결과: {{58, 64}, {139, 154}}
```

---

## 3.3 행렬-벡터 곱셈

### 정의

행렬 A (m×n)과 벡터 v (n차원)의 곱은 m차원 벡터입니다.

```java
public static double[] multiplyVector(double[][] A, double[] v) {
    int m = A.length;
    int n = A[0].length;

    if (v.length != n) {
        throw new IllegalArgumentException("차원 불일치");
    }

    double[] result = new double[m];
    for (int i = 0; i < m; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            sum += A[i][j] * v[j];
        }
        result[i] = sum;
    }
    return result;
}
```

### 신경망에서의 활용

```java
// 신경망 한 레이어의 계산
// output = W × input + bias

double[] input = {0.5, 0.3, 0.8};     // 3차원 입력
double[][] W = {                       // (2×3) 가중치
    {0.1, 0.2, 0.3},
    {0.4, 0.5, 0.6}
};
double[] bias = {0.1, 0.2};           // 2차원 편향

// 행렬-벡터 곱: (2×3) × (3) = (2)
double[] z = multiplyVector(W, input);  // [0.38, 0.83]

// 편향 더하기
double[] output = Vector.add(z, bias);  // [0.48, 1.03]

// 활성화 함수 적용
double[] activated = Activation.relu(output);
```

---

## 3.4 전치 (Transpose)

### 전치란?

행과 열을 바꾸는 연산입니다. A가 (m×n)이면 A^T는 (n×m)입니다.

$$A^T_{ij} = A_{ji}$$

```java
public static double[][] transpose(double[][] A) {
    int rows = A.length;
    int cols = A[0].length;

    double[][] T = new double[cols][rows];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            T[j][i] = A[i][j];
        }
    }
    return T;
}

// 사용 예
double[][] A = {
    {1, 2, 3},
    {4, 5, 6}
};  // (2×3)

double[][] AT = transpose(A);
// {{1, 4},
//  {2, 5},
//  {3, 6}}  // (3×2)
```

### AI에서 전치의 활용

**1. 역전파에서 가중치 전치**

```java
// 순전파: y = Wx
// 역전파: dL/dx = W^T × dL/dy

double[][] W = {...};        // (outputDim × inputDim)
double[] dLdy = {...};       // 출력의 그래디언트

double[][] WT = transpose(W);
double[] dLdx = multiplyVector(WT, dLdy);
```

**2. 배치 처리에서 차원 맞추기**

```java
// 데이터 배치: (배치크기 × 피처수)
// 가중치: (피처수 × 출력크기)

// 배치 처리: X × W
// X: (32 × 784), W: (784 × 128) → 결과: (32 × 128)
```

---

## 3.5 항등 행렬과 역행렬

### 항등 행렬 (Identity Matrix)

대각선이 모두 1이고 나머지가 0인 정사각 행렬입니다.

```java
public static double[][] identity(int n) {
    double[][] I = new double[n][n];
    for (int i = 0; i < n; i++) {
        I[i][i] = 1.0;
    }
    return I;
}

// 3×3 항등 행렬
// {{1, 0, 0},
//  {0, 1, 0},
//  {0, 0, 1}}
```

**성질**: 어떤 행렬 A에 대해 A × I = I × A = A

### 역행렬 (Inverse Matrix)

행렬 A의 역행렬 A^(-1)은 다음을 만족합니다:

$$A \times A^{-1} = A^{-1} \times A = I$$

```java
// 2×2 행렬의 역행렬 (간단한 경우만)
public static double[][] inverse2x2(double[][] A) {
    double a = A[0][0], b = A[0][1];
    double c = A[1][0], d = A[1][1];

    double det = a * d - b * c;  // 행렬식

    if (Math.abs(det) < 1e-10) {
        throw new IllegalArgumentException("역행렬이 존재하지 않습니다");
    }

    return new double[][] {
        { d / det, -b / det},
        {-c / det,  a / det}
    };
}
```

> **참고**: 일반적인 역행렬 계산은 복잡합니다. 실무에서는 NumPy나 라이브러리를 사용합니다.

---

## 3.6 행렬 곱셈의 기하학적 의미: 선형 변환

### 선형 변환이란?

행렬 곱셈은 공간의 **변환**으로 이해할 수 있습니다.

```java
// 2D 점을 변환
double[] point = {1, 0};  // x축 위의 점

double[][] transform = {
    {2, 0},
    {0, 2}
};

double[] newPoint = multiplyVector(transform, point);
// [2, 0] - 2배 확대!
```

### 주요 변환 행렬

**1. 스케일링 (확대/축소)**

```java
// x축 2배, y축 3배 확대
double[][] scale = {
    {2, 0},
    {0, 3}
};
```

**2. 회전**

```java
// θ 각도만큼 회전
double theta = Math.PI / 4;  // 45도
double[][] rotation = {
    {Math.cos(theta), -Math.sin(theta)},
    {Math.sin(theta),  Math.cos(theta)}
};
```

**3. 반사**

```java
// x축 기준 반사 (위아래 뒤집기)
double[][] reflectX = {
    {1,  0},
    {0, -1}
};

// y축 기준 반사 (좌우 뒤집기)
double[][] reflectY = {
    {-1, 0},
    { 0, 1}
};
```

### AI에서의 의미

신경망의 가중치 행렬은 **입력 공간을 출력 공간으로 변환**합니다:

```java
// 784차원(이미지) → 128차원(특징)
// 이 변환을 학습하는 것이 신경망 훈련!

double[][] W = new double[784][128];  // 학습되는 변환 행렬
double[] image = new double[784];     // 입력 이미지

double[] features = multiplyVector(W, image);
// 784차원 공간의 점이 128차원 공간의 점으로 변환됨
```

---

## 3.7 [AI 연결] 신경망의 순전파 = 행렬 곱셈의 연속

### 순전파 공식

단순한 신경망의 순전파:

$$\mathbf{h} = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$$
$$\mathbf{y} = \sigma(\mathbf{W}_2 \mathbf{h} + \mathbf{b}_2)$$

```java
public class SimpleNeuralNetwork {
    private double[][] W1;  // (hiddenSize × inputSize)
    private double[] b1;    // (hiddenSize)
    private double[][] W2;  // (outputSize × hiddenSize)
    private double[] b2;    // (outputSize)

    public double[] forward(double[] x) {
        // 첫 번째 레이어
        double[] z1 = Matrix.multiplyVector(W1, x);
        z1 = Vector.add(z1, b1);
        double[] h = Activation.relu(z1);

        // 두 번째 레이어
        double[] z2 = Matrix.multiplyVector(W2, h);
        z2 = Vector.add(z2, b2);
        double[] y = Activation.softmax(z2);

        return y;
    }
}
```

### 배치 처리

여러 샘플을 한 번에 처리하면 효율적입니다:

```java
// 개별 처리 (느림)
for (int i = 0; i < batchSize; i++) {
    outputs[i] = forward(inputs[i]);
}

// 배치 처리 (빠름) - 행렬 곱셈 한 번!
// inputs: (batchSize × inputSize)
// W: (inputSize × outputSize)
// outputs: (batchSize × outputSize)
double[][] outputs = Matrix.multiply(inputs, W);
```

### 전체 파이프라인

```
입력 이미지 (32, 784)
        ↓
    × W1 (784, 256) + b1
        ↓
    ReLU 활성화
        ↓
    (32, 256) 은닉층
        ↓
    × W2 (256, 128) + b2
        ↓
    ReLU 활성화
        ↓
    (32, 128) 은닉층
        ↓
    × W3 (128, 10) + b3
        ↓
    Softmax
        ↓
    출력 확률 (32, 10)
```

---

## 연습 문제

### 1. 행렬 곱셈 Shape

다음 연산의 결과 Shape을 구하세요:
- A (64, 784) × B (784, 256) = ?
- C (256, 128) × D (128, 10) = ?
- E (32, 64) × F (128, 32) = ? (가능/불가능?)

### 2. 전치 구현 확인

```java
// A × A^T는 항상 대칭 행렬인가?
// 테스트 코드를 작성하여 확인하세요.
```

### 3. 간단한 신경망

```java
// 2→3→2 신경망 구현
// 입력: 2차원, 은닉층: 3차원, 출력: 2차원
public class TinyNetwork {
    // 가중치와 편향을 초기화하고
    // forward() 메서드를 구현하세요
}
```

---

## 정리

| 연산 | Shape 변화 | AI 활용 |
|-----|-----------|--------|
| 행렬 덧셈 | 같음 | 편향 추가, 잔차 연결 |
| 스칼라 곱 | 같음 | 학습률 적용 |
| 행렬 곱셈 | (m,n)×(n,p)→(m,p) | 레이어 계산, 배치 처리 |
| 전치 | (m,n)→(n,m) | 역전파, 차원 맞춤 |
| 역행렬 | 같음 | 최소제곱법, 정규방정식 |

**핵심 포인트**:
1. 행렬 곱셈 = 신경망의 핵심 연산
2. Shape을 항상 추적하라 (디버깅의 핵심)
3. 배치 처리로 효율적인 계산

---

## 다음 장 예고

4장에서는 함수와 활성화 함수를 배웁니다:
- 왜 비선형 함수가 필요한가
- Sigmoid, ReLU, Softmax
- 함수 합성과 신경망
