# 0장. 오리엔테이션

## 0.1 왜 개발자에게 수학이 필요한가

### 개발자의 흔한 질문

"저는 10년차 자바 개발자입니다. 스프링 부트로 REST API 만들고, JPA로 데이터베이스 연동하고, 쿠버네티스에 배포합니다. 그런데 AI를 배우려니까 갑자기 수학이 나옵니다. 정말 필요한가요?"

필요합니다. 하지만 모든 수학이 필요한 건 아닙니다.

### 수학 없이 AI를 쓸 수 있지 않나요?

맞습니다. scikit-learn이나 PyTorch로 모델을 학습시키는 건 수학 없이도 가능합니다.

```python
# 수학 몰라도 작동하는 코드
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X_test)
```

그런데 문제가 생기면 어떻게 할까요?

- 모델 정확도가 60%에서 안 올라갑니다
- 학습이 발산합니다 (loss가 NaN)
- 어떤 하이퍼파라미터를 조정해야 할지 모릅니다

이때 수학을 모르면 "이것저것 바꿔보기"만 할 수 있습니다. 수학을 알면 "왜 이런 현상이 발생하는지" 이해하고 체계적으로 해결할 수 있습니다.

### 개발자에게 필요한 수학의 범위

AI에 필요한 수학은 크게 세 영역입니다:

| 영역 | 핵심 내용 | AI에서의 역할 |
|------|----------|--------------|
| 선형대수 | 벡터, 행렬 연산 | 데이터 표현, 모델 계산 |
| 미적분 | 미분, 편미분 | 모델 학습 (경사하강법) |
| 확률/통계 | 분포, 추정, 검정 | 모델 평가, 불확실성 |

이 책(Vol.1)에서는 선형대수와 미적분을 다룹니다. 확률/통계는 Vol.2에서 다룹니다.

### 수학자처럼 증명할 필요는 없습니다

우리의 목표는 수학자가 아니라 AI 엔지니어입니다.

- ❌ 정리를 증명하는 것
- ❌ 공식을 암기하는 것
- ✅ 개념을 직관적으로 이해하는 것
- ✅ 코드로 구현할 수 있는 것
- ✅ AI에서 어떻게 쓰이는지 아는 것

---

## 0.2 이 책의 학습 방식: 코드로 이해하는 수학

### Java 개발자의 장점

여러분은 이미 강력한 무기를 가지고 있습니다: **코드로 생각하는 능력**.

수학 공식을 보면 추상적으로 느껴집니다:

$$\mathbf{c} = \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i$$

하지만 Java 코드로 보면 명확합니다:

```java
public static double dotProduct(double[] a, double[] b) {
    double sum = 0;
    for (int i = 0; i < a.length; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
```

이게 내적(dot product)입니다. 두 배열의 같은 위치 원소를 곱해서 더한 것.

### 학습 사이클

이 책의 모든 개념은 다음 사이클로 학습합니다:

```
1. 개념 소개 (한 문장 정의)
      ↓
2. Java로 구현 (동작 이해)
      ↓
3. 시각화 (기하학적 의미)
      ↓
4. AI 연결 (실제 사용처)
      ↓
5. NumPy/PyTorch 전환 (실무 코드)
```

### 예시: 벡터의 크기

**1. 개념**: 벡터의 크기(norm)는 원점에서 벡터 끝점까지의 거리

**2. Java 구현**:
```java
public static double norm(double[] v) {
    double sumOfSquares = 0;
    for (double x : v) {
        sumOfSquares += x * x;
    }
    return Math.sqrt(sumOfSquares);
}
```

**3. 시각화**: 2D 벡터 [3, 4]의 크기는 5 (피타고라스 정리)

**4. AI 연결**: 벡터 정규화할 때 사용. 임베딩 벡터를 단위 벡터로 만들 때.

**5. NumPy 전환**:
```python
import numpy as np
norm = np.linalg.norm(v)
```

---

## 0.3 AI에서 수학이 쓰이는 곳 (전체 지도)

### 신경망의 구조

신경망은 결국 수학 연산의 조합입니다:

```
입력 데이터 (벡터/행렬)
      ↓
[가중치 곱셈] ← 행렬 곱셈 (선형대수)
      ↓
[활성화 함수] ← 비선형 함수 (미적분)
      ↓
    반복...
      ↓
[손실 계산] ← 손실 함수 (미적분)
      ↓
[역전파] ← 편미분, 연쇄법칙 (미적분)
      ↓
[가중치 업데이트] ← 경사하강법 (최적화)
```

### 각 장에서 배우는 내용의 위치

| 장 | 내용 | 신경망에서의 위치 |
|----|------|------------------|
| 1장 | 벡터, 행렬, 텐서 | 모든 데이터 표현 |
| 2장 | 벡터 연산, 내적 | 유사도, 뉴런 계산 |
| 3장 | 행렬 곱셈 | 순전파의 핵심 |
| 4장 | 함수, 활성화 함수 | 비선형성 추가 |
| 5장 | 미분, 편미분 | 역전파의 기초 |
| 6장 | 경사하강법 | 학습 알고리즘 |
| 7장 | 손실 함수 | 오차 측정 |
| 8장 | 실전 통합 | 전체 파이프라인 |

---

## 0.4 준비물

### 필수

- **Java 17+**: 예제 코드 실행용
- **IDE**: IntelliJ IDEA 또는 VS Code
- **Python 3.11+**: NumPy/PyTorch 실습용
- **Jupyter Notebook**: 시각화 및 실험용

### Java 프로젝트 설정

```xml
<!-- pom.xml -->
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>ai-math</artifactId>
    <version>1.0.0</version>
    <properties>
        <maven.compiler.source>17</maven.compiler.source>
        <maven.compiler.target>17</maven.compiler.target>
    </properties>
</project>
```

### Python 환경 설정

```bash
# 가상환경 생성
python -m venv venv

# 활성화 (Windows)
venv\Scripts\activate

# 패키지 설치
pip install numpy matplotlib jupyter torch
```

### 확인 코드

Java:
```java
public class EnvCheck {
    public static void main(String[] args) {
        System.out.println("Java: " + System.getProperty("java.version"));
        double[] v = {1, 2, 3};
        System.out.println("Vector created: length = " + v.length);
    }
}
```

Python:
```python
import numpy as np
import torch

print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
v = np.array([1, 2, 3])
print(f"Vector created: {v}")
```

---

## 0.5 이 책에서 만드는 것

### 최종 목표: MNIST 손글씨 분류기

8장에서 우리는 손글씨 숫자(0-9)를 인식하는 신경망을 **바닥부터** 만듭니다.

```
입력: 28x28 픽셀 이미지 (784차원 벡터)
      ↓
[첫 번째 레이어] 784 → 128
      ↓
[ReLU 활성화]
      ↓
[두 번째 레이어] 128 → 10
      ↓
[Softmax]
      ↓
출력: 0~9 각 숫자일 확률
```

### Java로 먼저, 그 다음 PyTorch

```java
// Java로 직접 구현
public class SimpleNN {
    private double[][] weights1; // 784 x 128
    private double[][] weights2; // 128 x 10

    public double[] forward(double[] input) {
        double[] hidden = matmul(input, weights1);
        hidden = relu(hidden);
        double[] output = matmul(hidden, weights2);
        return softmax(output);
    }

    public void backward(double[] input, int label) {
        // 역전파 구현
    }
}
```

```python
# PyTorch로 같은 모델
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

Java 버전을 먼저 만들면 PyTorch가 무엇을 추상화하고 있는지 정확히 이해할 수 있습니다.

---

## 다음 장 예고

1장에서는 AI의 가장 기본적인 데이터 구조인 **벡터, 행렬, 텐서**를 다룹니다. Java 배열과 비교하면서 시작합니다.

```java
// 다음 장에서 다룰 내용 미리보기
double[] vector = {1, 2, 3};           // 1차원: 벡터
double[][] matrix = {{1, 2}, {3, 4}};  // 2차원: 행렬
double[][][] tensor = {...};            // 3차원: 텐서
```
