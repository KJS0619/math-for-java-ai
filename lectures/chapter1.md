# 1장. 스칼라, 벡터, 행렬 — 데이터의 그릇

## 1.1 스칼라: 단일 값

### 스칼라란?

스칼라(Scalar)는 가장 단순한 수학적 객체입니다. 하나의 숫자.

```java
// 스칼라의 예
double temperature = 36.5;    // 온도
int count = 42;               // 개수
double probability = 0.85;    // 확률
```

자바 개발자에게 스칼라는 익숙합니다. `int`, `double`, `float` 같은 기본 타입이 모두 스칼라입니다.

### AI에서 스칼라가 쓰이는 곳

- **손실 값(Loss)**: 모델이 얼마나 틀렸는지 하나의 숫자로 표현
- **학습률(Learning Rate)**: 0.001 같은 하이퍼파라미터
- **확률**: 분류 결과의 신뢰도 (예: 고양이일 확률 0.92)

```java
double loss = 0.0234;        // 이번 배치의 손실
double learningRate = 0.001; // 학습률
double confidence = 0.92;    // 예측 신뢰도
```

스칼라는 단순하지만, 벡터와 행렬 연산의 결과가 스칼라인 경우가 많습니다 (예: 내적, 손실 함수).

---

## 1.2 벡터: 순서 있는 숫자 모음

### 벡터란?

벡터(Vector)는 **순서가 있는 숫자들의 모음**입니다.

```java
// 벡터의 예
double[] v = {1.0, 2.0, 3.0};  // 3차원 벡터
```

수학에서는 이렇게 표기합니다:

$$\mathbf{v} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$$

### Java 배열로 벡터 표현

```java
public class VectorDemo {
    public static void main(String[] args) {
        // 3차원 벡터
        double[] position = {10.0, 20.0, 30.0};  // x, y, z 좌표

        // 784차원 벡터 (MNIST 이미지)
        double[] image = new double[784];  // 28x28 픽셀을 펼친 것

        // 벡터의 차원 = 배열의 길이
        System.out.println("position 차원: " + position.length);  // 3
        System.out.println("image 차원: " + image.length);        // 784
    }
}
```

### 벡터의 기하학적 의미

2차원 벡터 `[3, 4]`를 시각화하면:

```
    y
    ↑
  4 +       • (3,4)
    |      /
    |     /
    |    /
    |   /
    +---+---+---→ x
    0   1   2   3
```

벡터는 **원점에서 특정 점으로 향하는 화살표**로 이해할 수 있습니다.

### AI에서 벡터가 쓰이는 곳

**1. 데이터 한 건 = 벡터 하나**

```java
// 사용자 정보를 벡터로 표현
// [나이, 키(cm), 몸무게(kg), 연소득(만원)]
double[] user1 = {30, 175, 70, 5000};
double[] user2 = {25, 162, 55, 4200};
```

**2. 단어 임베딩**

자연어 처리에서 단어를 벡터로 표현합니다:

```java
// 가상의 단어 임베딩 (실제로는 수백 차원)
double[] king = {0.5, 0.8, 0.2, -0.3};
double[] queen = {0.6, 0.7, 0.3, -0.2};
double[] man = {0.4, 0.1, 0.5, 0.9};
double[] woman = {0.5, 0.0, 0.6, 0.8};

// 유명한 공식: king - man + woman ≈ queen
```

**3. 신경망 레이어의 출력**

```java
// 신경망의 각 레이어 출력은 벡터
double[] input = new double[784];    // 입력층: 784차원
double[] hidden = new double[128];   // 은닉층: 128차원
double[] output = new double[10];    // 출력층: 10차원 (0~9 분류)
```

### 벡터 생성 유틸리티

```java
public class VectorUtils {

    // 영벡터 생성
    public static double[] zeros(int n) {
        return new double[n];
    }

    // 모든 원소가 1인 벡터
    public static double[] ones(int n) {
        double[] v = new double[n];
        Arrays.fill(v, 1.0);
        return v;
    }

    // 랜덤 벡터 (가중치 초기화용)
    public static double[] random(int n) {
        double[] v = new double[n];
        Random rand = new Random();
        for (int i = 0; i < n; i++) {
            v[i] = rand.nextGaussian() * 0.01;  // 작은 값으로 초기화
        }
        return v;
    }

    // 벡터 출력
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
```

---

## 1.3 행렬: 2차원 숫자 배열

### 행렬이란?

행렬(Matrix)은 **2차원으로 배열된 숫자들**입니다.

```java
// 2x3 행렬 (2행 3열)
double[][] matrix = {
    {1, 2, 3},
    {4, 5, 6}
};
```

수학에서는 이렇게 표기합니다:

$$\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}$$

### 행렬의 형태(Shape)

행렬의 형태는 **(행의 수, 열의 수)**로 표현합니다.

```java
double[][] A = {
    {1, 2, 3},     // 0번 행
    {4, 5, 6}      // 1번 행
};
// 열: 0  1  2

int rows = A.length;        // 2 (행의 수)
int cols = A[0].length;     // 3 (열의 수)
System.out.println("Shape: (" + rows + ", " + cols + ")");  // (2, 3)
```

### 행렬의 원소 접근

```java
double[][] A = {
    {1, 2, 3},
    {4, 5, 6}
};

// A[행][열] - 0부터 시작
double element = A[0][2];  // 첫 번째 행, 세 번째 열 → 3
A[1][0] = 10;              // 두 번째 행, 첫 번째 열을 10으로 변경
```

### AI에서 행렬이 쓰이는 곳

**1. 데이터 배치 = 행렬**

여러 데이터를 한 번에 처리할 때 행렬로 묶습니다:

```java
// 4명의 사용자 데이터 (배치 크기 = 4)
// 각 사용자: [나이, 키, 몸무게]
double[][] batch = {
    {30, 175, 70},   // 사용자 1
    {25, 162, 55},   // 사용자 2
    {35, 180, 85},   // 사용자 3
    {28, 168, 62}    // 사용자 4
};
// Shape: (4, 3) = (배치 크기, 피처 수)
```

**2. 이미지 = 행렬**

흑백 이미지는 2D 행렬입니다:

```java
// 28x28 MNIST 이미지
double[][] image = new double[28][28];

// 픽셀 값: 0(검은색) ~ 255(흰색)
// 또는 정규화: 0.0 ~ 1.0
```

**3. 가중치 = 행렬**

신경망의 레이어 간 연결을 행렬로 표현합니다:

```java
// 입력: 784차원 → 출력: 128차원
// 가중치 행렬 Shape: (784, 128)
double[][] weights = new double[784][128];

// 각 원소 W[i][j]는
// "입력 i번 뉴런 → 출력 j번 뉴런" 연결 강도
```

### 행렬 생성 유틸리티

```java
public class MatrixUtils {

    // 영행렬
    public static double[][] zeros(int rows, int cols) {
        return new double[rows][cols];
    }

    // 단위행렬 (대각선만 1)
    public static double[][] identity(int n) {
        double[][] I = new double[n][n];
        for (int i = 0; i < n; i++) {
            I[i][i] = 1.0;
        }
        return I;
    }

    // 랜덤 행렬 (Xavier 초기화)
    public static double[][] randomXavier(int rows, int cols) {
        double[][] M = new double[rows][cols];
        double scale = Math.sqrt(2.0 / (rows + cols));
        Random rand = new Random();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                M[i][j] = rand.nextGaussian() * scale;
            }
        }
        return M;
    }

    // 형태 반환
    public static int[] shape(double[][] M) {
        return new int[]{M.length, M[0].length};
    }

    // 행렬 출력
    public static void print(double[][] M) {
        for (double[] row : M) {
            System.out.print("[");
            for (int j = 0; j < row.length; j++) {
                System.out.printf("%8.4f", row[j]);
                if (j < row.length - 1) System.out.print(", ");
            }
            System.out.println("]");
        }
    }
}
```

---

## 1.4 텐서: N차원으로 확장

### 텐서란?

텐서(Tensor)는 **N차원 배열**의 일반화입니다.

| 차원 | 이름 | Java 표현 | 예시 |
|-----|------|----------|------|
| 0차원 | 스칼라 | `double` | 온도: 36.5 |
| 1차원 | 벡터 | `double[]` | 좌표: [x, y, z] |
| 2차원 | 행렬 | `double[][]` | 흑백 이미지, 데이터 배치 |
| 3차원 | 3D 텐서 | `double[][][]` | 컬러 이미지, 시계열 배치 |
| 4차원 | 4D 텐서 | `double[][][][]` | 이미지 배치 |

### 3D 텐서: 컬러 이미지

컬러 이미지는 (높이, 너비, 채널) 3차원입니다:

```java
// 28x28 컬러 이미지 (RGB 3채널)
double[][][] colorImage = new double[28][28][3];

// colorImage[y][x][c]
// y: 행(높이), x: 열(너비), c: 채널(0=R, 1=G, 2=B)
colorImage[0][0][0] = 0.9;  // (0,0) 픽셀의 빨강 값
colorImage[0][0][1] = 0.1;  // (0,0) 픽셀의 초록 값
colorImage[0][0][2] = 0.2;  // (0,0) 픽셀의 파랑 값
```

### 4D 텐서: 이미지 배치

딥러닝에서 이미지를 배치로 처리할 때:

```java
// 32장의 28x28 컬러 이미지
// Shape: (배치, 높이, 너비, 채널) = (32, 28, 28, 3)
double[][][][] imageBatch = new double[32][28][28][3];

// 또는 PyTorch 스타일: (배치, 채널, 높이, 너비)
double[][][][] imageBatchPyTorch = new double[32][3][28][28];
```

### Java에서 다차원 배열의 한계

Java의 다차원 배열은 실제로 "배열의 배열"입니다:

```java
// 이것은 진짜 2D 배열이 아니라
// "double[] 배열"의 배열
double[][] matrix = new double[3][4];

// 각 행이 다른 길이일 수 있음 (비정형 배열)
double[][] jagged = {
    {1, 2},
    {3, 4, 5, 6},
    {7}
};
```

실제 AI 작업에서는 NumPy나 PyTorch의 진짜 다차원 배열을 사용합니다. 이 책에서 Java로 구현하는 것은 개념 이해를 위함입니다.

---

## 1.5 NumPy ndarray로 전환

### Java 배열 → NumPy 변환

Java에서 배운 개념을 Python/NumPy로 옮겨봅시다.

**스칼라**
```python
import numpy as np

# 스칼라
loss = 0.0234
learning_rate = 0.001
```

**벡터**
```python
# Java: double[] v = {1, 2, 3};
v = np.array([1, 2, 3])

print(v.shape)  # (3,)
print(v.ndim)   # 1 (1차원)
```

**행렬**
```python
# Java:
# double[][] A = {
#     {1, 2, 3},
#     {4, 5, 6}
# };

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(A.shape)  # (2, 3)
print(A.ndim)   # 2 (2차원)
```

**텐서**
```python
# 3D 텐서: (2, 3, 4)
tensor = np.zeros((2, 3, 4))
print(tensor.shape)  # (2, 3, 4)
print(tensor.ndim)   # 3
```

### NumPy의 장점

```python
# 1. 벡터화 연산 (반복문 없이!)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Java에서는 반복문 필요
# Python에서는 한 줄
c = a + b  # [5, 7, 9]
d = a * b  # [4, 10, 18] (원소별 곱셈)
e = a @ b  # 32 (내적)

# 2. 브로드캐스팅
matrix = np.array([[1, 2], [3, 4]])
scalar = 10
result = matrix + scalar  # 모든 원소에 10 더함
# [[11, 12], [13, 14]]

# 3. Shape 변환
flat = np.arange(12)        # [0,1,2,...,11]
matrix = flat.reshape(3, 4) # (3, 4) 행렬로 변환
vector = matrix.flatten()   # 다시 1차원으로
```

---

## 1.6 [AI 연결] 신경망의 입력/출력/가중치는 모두 텐서

### 신경망 = 텐서 변환기

신경망은 텐서를 입력받아 텐서를 출력하는 함수입니다:

```
입력 텐서 → [신경망] → 출력 텐서
```

### MNIST 예제로 보는 텐서 흐름

```java
// 1. 입력: 이미지 배치
// Shape: (배치크기, 784) - 28*28=784
double[][] input = new double[32][784];

// 2. 첫 번째 레이어 가중치
// Shape: (784, 128) - 784차원 → 128차원
double[][] W1 = new double[784][128];
double[] b1 = new double[128];  // 편향

// 3. 첫 번째 레이어 출력
// Shape: (32, 128)
// hidden = input × W1 + b1
double[][] hidden = new double[32][128];

// 4. 두 번째 레이어 가중치
// Shape: (128, 10) - 128차원 → 10차원
double[][] W2 = new double[128][10];
double[] b2 = new double[10];

// 5. 출력
// Shape: (32, 10) - 32개 샘플 × 10개 클래스
double[][] output = new double[32][10];
```

### Shape 변환의 중요성

```java
// 틀린 Shape으로 연산하면 에러!
double[][] A = new double[32][784];  // (32, 784)
double[][] B = new double[784][128]; // (784, 128)

// A × B 가능: 결과 (32, 128)
// A의 열 수(784) == B의 행 수(784)

double[][] C = new double[32][100];  // (32, 100)
// A × C 불가능!
// A의 열 수(784) != C의 행 수(32)
```

### PyTorch에서의 텐서

```python
import torch

# 텐서 생성
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(x.shape)  # torch.Size([2, 2])

# GPU로 이동
if torch.cuda.is_available():
    x = x.cuda()

# 자동 미분 활성화
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 연산 후 역전파
y = x.sum()
y.backward()
print(x.grad)  # 그래디언트
```

---

## 연습 문제

### 1. Shape 맞추기

다음 행렬 곱셈이 가능한지, 가능하다면 결과 Shape은?

```
A: (64, 256)
B: (256, 128)
C: (128, 10)

A × B = ?
(A × B) × C = ?
```

### 2. Java로 구현하기

다음 기능을 Java로 구현하세요:

```java
// 행렬의 특정 열 추출
public static double[] getColumn(double[][] M, int col);

// 행렬의 특정 행 추출
public static double[] getRow(double[][] M, int row);

// 두 행렬이 같은 Shape인지 확인
public static boolean sameShape(double[][] A, double[][] B);
```

### 3. 실습: 이미지 → 벡터 변환

MNIST 이미지(28x28)를 784차원 벡터로 변환하는 함수:

```java
// 2D 이미지 → 1D 벡터
public static double[] flatten(double[][] image) {
    // 구현하세요
}

// 1D 벡터 → 2D 이미지
public static double[][] reshape(double[] vector, int rows, int cols) {
    // 구현하세요
}
```

---

## 정리

| 개념 | 차원 | Java 타입 | AI 예시 |
|-----|-----|----------|--------|
| 스칼라 | 0D | `double` | 손실, 학습률 |
| 벡터 | 1D | `double[]` | 데이터 1건, 임베딩 |
| 행렬 | 2D | `double[][]` | 데이터 배치, 가중치 |
| 텐서 | ND | `double[]...[]` | 이미지 배치 |

**핵심 포인트**:
1. AI의 모든 데이터는 텐서로 표현된다
2. Shape을 항상 의식하라 (연산 가능 여부 결정)
3. Java 배열로 개념을 이해하고, NumPy/PyTorch로 실무 적용

---

## 다음 장 예고

2장에서는 벡터에 대한 연산을 배웁니다:
- 벡터 덧셈과 스칼라 곱
- 내적 (Dot Product) - AI의 핵심 연산
- 벡터의 크기 (Norm)
- 코사인 유사도 - 추천 시스템의 기초
