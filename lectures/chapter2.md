# 2장. 벡터 연산 — AI의 기본 동작

## 2.1 벡터 덧셈과 스칼라 곱

### 벡터 덧셈

같은 차원의 벡터끼리 더할 수 있습니다. 같은 위치의 원소끼리 더합니다.

$$\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 \\ a_2 \\ a_3 \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \\ a_3 + b_3 \end{bmatrix}$$

```java
public static double[] add(double[] a, double[] b) {
    if (a.length != b.length) {
        throw new IllegalArgumentException("벡터 차원이 다릅니다");
    }
    double[] result = new double[a.length];
    for (int i = 0; i < a.length; i++) {
        result[i] = a[i] + b[i];
    }
    return result;
}

// 사용 예
double[] a = {1, 2, 3};
double[] b = {4, 5, 6};
double[] c = add(a, b);  // [5, 7, 9]
```

### 기하학적 의미

벡터 덧셈은 두 화살표를 이어 붙이는 것입니다:

```
        b
    --------→
   /          \
  /   a + b    \
 ↗              ↘
a                결과
```

### 스칼라 곱 (Scalar Multiplication)

벡터에 스칼라를 곱하면 모든 원소에 같은 값이 곱해집니다:

$$c \cdot \mathbf{v} = c \cdot \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix} = \begin{bmatrix} c \cdot v_1 \\ c \cdot v_2 \\ c \cdot v_3 \end{bmatrix}$$

```java
public static double[] scale(double[] v, double scalar) {
    double[] result = new double[v.length];
    for (int i = 0; i < v.length; i++) {
        result[i] = v[i] * scalar;
    }
    return result;
}

// 사용 예
double[] v = {1, 2, 3};
double[] doubled = scale(v, 2.0);   // [2, 4, 6]
double[] halved = scale(v, 0.5);    // [0.5, 1, 1.5]
double[] negated = scale(v, -1.0);  // [-1, -2, -3]
```

### 기하학적 의미

- `scale(v, 2)`: 같은 방향으로 2배 길어짐
- `scale(v, 0.5)`: 같은 방향으로 절반 길이
- `scale(v, -1)`: 반대 방향

### AI에서의 활용

**1. 가중치 업데이트 (경사하강법)**

```java
// weights = weights - learning_rate * gradient
double learningRate = 0.01;
double[] weights = {0.5, 0.3, 0.2};
double[] gradient = {0.1, -0.2, 0.05};

// gradient에 learning_rate 곱하기
double[] scaledGradient = scale(gradient, learningRate);

// weights에서 빼기
weights = subtract(weights, scaledGradient);
```

**2. 편향(Bias) 더하기**

```java
// output = input + bias (브로드캐스팅)
double[] layerOutput = {0.5, 0.3, 0.8};
double[] bias = {0.1, 0.1, 0.1};
double[] withBias = add(layerOutput, bias);
```

---

## 2.2 내적 (Dot Product)

### 내적이란?

내적은 두 벡터를 하나의 스칼라로 만드는 연산입니다.

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$$

```java
public static double dot(double[] a, double[] b) {
    if (a.length != b.length) {
        throw new IllegalArgumentException("벡터 차원이 다릅니다");
    }
    double sum = 0;
    for (int i = 0; i < a.length; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// 사용 예
double[] a = {1, 2, 3};
double[] b = {4, 5, 6};
double result = dot(a, b);  // 1*4 + 2*5 + 3*6 = 32
```

### 기하학적 의미

내적은 두 벡터가 **얼마나 같은 방향을 향하는지** 측정합니다.

$$\mathbf{a} \cdot \mathbf{b} = |\mathbf{a}| |\mathbf{b}| \cos\theta$$

- θ = 0° (같은 방향): 내적 > 0 (최대)
- θ = 90° (직교): 내적 = 0
- θ = 180° (반대 방향): 내적 < 0 (최소)

```
같은 방향          직교              반대 방향
   a→               a→                 a→
   b→                ↑b                 ←b
 내적 > 0         내적 = 0           내적 < 0
```

### AI에서 내적이 핵심인 이유

**1. 뉴런의 계산 = 내적**

```java
// 뉴런 하나의 계산
// output = dot(inputs, weights) + bias
double[] inputs = {0.5, 0.3, 0.8};   // 입력값
double[] weights = {0.2, 0.4, 0.1};  // 가중치
double bias = 0.1;

double z = dot(inputs, weights) + bias;
// z = 0.5*0.2 + 0.3*0.4 + 0.8*0.1 + 0.1 = 0.40

double output = sigmoid(z);  // 활성화 함수 적용
```

**2. 유사도 측정**

```java
// 두 문서/단어/상품의 유사도
double[] docA = {0.8, 0.2, 0.5};  // 문서 A의 벡터
double[] docB = {0.7, 0.3, 0.6};  // 문서 B의 벡터

double similarity = dot(docA, docB);  // 유사도
```

**3. 어텐션 메커니즘 (Transformer)**

```java
// Query와 Key의 내적 → 어텐션 점수
double[] query = {...};  // 현재 단어
double[] key = {...};    // 다른 단어

double attentionScore = dot(query, key) / Math.sqrt(dimension);
```

---

## 2.3 벡터의 크기 (Norm)

### L2 노름 (유클리드 거리)

가장 일반적인 벡터 크기 측정법입니다.

$$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2} = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

```java
public static double norm(double[] v) {
    double sumOfSquares = 0;
    for (double x : v) {
        sumOfSquares += x * x;
    }
    return Math.sqrt(sumOfSquares);
}

// 사용 예
double[] v = {3, 4};
double length = norm(v);  // √(9 + 16) = √25 = 5
```

### 기하학적 의미

2D 벡터 [3, 4]의 L2 노름은 피타고라스 정리:

```
    y
    ↑
  4 +       • (3,4)
    |      /|
    |     / |
    |  5 /  | 4
    |   /   |
    +--+----+→ x
    0  3
```

### L1 노름 (맨해튼 거리)

각 원소 절댓값의 합입니다.

$$\|\mathbf{v}\|_1 = \sum_{i=1}^{n} |v_i|$$

```java
public static double normL1(double[] v) {
    double sum = 0;
    for (double x : v) {
        sum += Math.abs(x);
    }
    return sum;
}

// 사용 예
double[] v = {3, -4};
double l1 = normL1(v);  // |3| + |-4| = 7
double l2 = norm(v);    // √(9 + 16) = 5
```

### AI에서 노름의 활용

**1. 정규화 (Regularization)**

```java
// L2 정규화: 가중치가 너무 커지는 것 방지
double l2Penalty = 0.01 * norm(weights);  // 손실에 추가
```

**2. 그래디언트 클리핑**

```java
// 그래디언트가 너무 크면 폭발 방지
double maxNorm = 1.0;
double gradNorm = norm(gradient);
if (gradNorm > maxNorm) {
    gradient = scale(gradient, maxNorm / gradNorm);
}
```

---

## 2.4 정규화 (Normalization)

### 단위 벡터 만들기

벡터를 그 크기로 나누면 길이가 1인 **단위 벡터**가 됩니다.

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

```java
public static double[] normalize(double[] v) {
    double n = norm(v);
    if (n == 0) {
        throw new IllegalArgumentException("영벡터는 정규화 불가");
    }
    return scale(v, 1.0 / n);
}

// 사용 예
double[] v = {3, 4};
double[] unit = normalize(v);  // [0.6, 0.8]

// 확인: 크기가 1인가?
System.out.println(norm(unit));  // 1.0
```

### 왜 정규화하는가?

**1. 공정한 비교**

```java
// 크기가 다른 벡터들
double[] short_ = {1, 2};      // 크기: √5 ≈ 2.24
double[] long_ = {100, 200};   // 크기: √50000 ≈ 224

// 내적: 방향보다 크기에 영향받음
double rawDot = dot(short_, long_);  // 500

// 정규화 후 내적: 순수하게 방향만 비교
double[] shortNorm = normalize(short_);
double[] longNorm = normalize(long_);
double normDot = dot(shortNorm, longNorm);  // 1.0 (같은 방향)
```

**2. 임베딩 저장 효율**

```java
// 정규화된 벡터는 크기 정보를 버리고 방향만 저장
// 메모리 절약 + 검색 속도 향상
double[] embedding = computeEmbedding(text);
double[] normalized = normalize(embedding);
database.store(normalized);
```

---

## 2.5 코사인 유사도

### 공식

두 벡터 사이의 각도의 코사인 값입니다.

$$\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$$

```java
public static double cosineSimilarity(double[] a, double[] b) {
    double dotProduct = dot(a, b);
    double normA = norm(a);
    double normB = norm(b);

    if (normA == 0 || normB == 0) {
        throw new IllegalArgumentException("영벡터의 유사도 계산 불가");
    }

    return dotProduct / (normA * normB);
}
```

### 값의 범위와 해석

| 코사인 유사도 | 각도 | 의미 |
|-------------|------|------|
| 1.0 | 0° | 완전히 같은 방향 |
| 0.5 | 60° | 어느 정도 유사 |
| 0.0 | 90° | 관계없음 (직교) |
| -0.5 | 120° | 어느 정도 반대 |
| -1.0 | 180° | 완전히 반대 방향 |

### AI에서의 활용: 추천 시스템

```java
public class RecommendationDemo {
    public static void main(String[] args) {
        // 사용자-아이템 임베딩
        double[] user = {0.8, 0.3, 0.5, 0.2};

        double[] movie1 = {0.7, 0.4, 0.6, 0.1};  // 액션 영화
        double[] movie2 = {0.2, 0.9, 0.1, 0.8};  // 로맨스 영화
        double[] movie3 = {0.75, 0.35, 0.55, 0.15};  // 액션 영화 2

        // 유사도 계산
        System.out.printf("Movie1 유사도: %.4f%n", cosineSimilarity(user, movie1));
        System.out.printf("Movie2 유사도: %.4f%n", cosineSimilarity(user, movie2));
        System.out.printf("Movie3 유사도: %.4f%n", cosineSimilarity(user, movie3));

        // 출력:
        // Movie1 유사도: 0.9823
        // Movie2 유사도: 0.5124
        // Movie3 유사도: 0.9967  ← 가장 추천!
    }
}
```

### 코사인 유사도 vs 유클리드 거리

```java
// 같은 방향, 다른 크기
double[] a = {1, 2, 3};
double[] b = {2, 4, 6};  // a의 2배

// 코사인 유사도: 1.0 (완전히 유사)
System.out.println(cosineSimilarity(a, b));  // 1.0

// 유클리드 거리: √14 ≈ 3.74 (꽤 멀다고 판단)
System.out.println(euclideanDistance(a, b));  // 3.74

// 결론: 방향만 중요하면 코사인 유사도, 크기도 중요하면 유클리드 거리
```

---

## 2.6 [AI 연결] 임베딩과 유사도 검색

### 임베딩이란?

임베딩(Embedding)은 단어, 문장, 이미지 등을 **고정 길이 벡터**로 변환한 것입니다.

```java
// 개념적인 예시
String[] words = {"king", "queen", "man", "woman"};

// 각 단어를 4차원 벡터로 임베딩
double[][] embeddings = {
    {0.5, 0.8, 0.2, -0.3},   // king
    {0.6, 0.7, 0.3, -0.2},   // queen
    {0.4, 0.1, 0.5, 0.9},    // man
    {0.5, 0.0, 0.6, 0.8}     // woman
};
```

### Word2Vec의 유명한 예시

```java
// king - man + woman ≈ queen

double[] king = {0.5, 0.8, 0.2, -0.3};
double[] man = {0.4, 0.1, 0.5, 0.9};
double[] woman = {0.5, 0.0, 0.6, 0.8};
double[] queen = {0.6, 0.7, 0.3, -0.2};

// king - man + woman 계산
double[] result = add(subtract(king, man), woman);
// result ≈ [0.6, 0.7, 0.3, -0.4]

// queen과 유사도
double similarity = cosineSimilarity(result, queen);
// 높은 유사도! → king - man + woman ≈ queen
```

### 벡터 검색 (Vector Search)

```java
public class VectorSearch {
    private double[][] database;  // N개의 벡터

    // 가장 유사한 K개 찾기
    public int[] findTopK(double[] query, int k) {
        double[] similarities = new double[database.length];

        for (int i = 0; i < database.length; i++) {
            similarities[i] = cosineSimilarity(query, database[i]);
        }

        // 상위 K개 인덱스 반환 (정렬)
        return topKIndices(similarities, k);
    }
}

// 사용 예: 질문에 가장 관련 있는 문서 찾기
String question = "자바에서 배열 정렬하는 방법";
double[] queryEmbedding = embed(question);
int[] topDocs = vectorSearch.findTopK(queryEmbedding, 5);
```

### 실제 응용: RAG (Retrieval-Augmented Generation)

```
1. 질문 임베딩: "Java 정렬 방법" → [0.3, 0.8, ...]

2. 문서 DB에서 유사 문서 검색 (코사인 유사도)
   - doc1: "Arrays.sort() 사용법" → 유사도 0.92
   - doc2: "Python 정렬" → 유사도 0.31
   - doc3: "Java Collections.sort()" → 유사도 0.89

3. 상위 문서를 LLM에 컨텍스트로 전달

4. LLM이 문서 기반으로 답변 생성
```

---

## 연습 문제

### 1. 내적 계산

다음 벡터의 내적을 손으로 계산하세요:
- a = [2, 3, 4]
- b = [1, -1, 2]

### 2. 코사인 유사도 해석

세 문서의 임베딩이 다음과 같을 때:
```java
double[] docA = {1, 0, 0};
double[] docB = {0, 1, 0};
double[] docC = {0.7, 0.7, 0};
```

각 쌍의 코사인 유사도를 계산하고 해석하세요.

### 3. 정규화 구현

다음 함수를 완성하세요:

```java
// 배치 정규화: 각 벡터를 정규화
public static double[][] normalizeBatch(double[][] batch) {
    // 구현하세요
}
```

### 4. 최근접 이웃 찾기

```java
// 주어진 query에 가장 가까운 벡터의 인덱스 반환
public static int findNearest(double[] query, double[][] database) {
    // 구현하세요 (코사인 유사도 사용)
}
```

---

## 정리

| 연산 | 수식 | 결과 | AI 활용 |
|-----|------|------|--------|
| 덧셈 | a + b | 벡터 | 편향 추가, 가중치 업데이트 |
| 스칼라곱 | c · v | 벡터 | 학습률 적용 |
| 내적 | a · b | 스칼라 | 뉴런 계산, 유사도 |
| L2 노름 | ‖v‖ | 스칼라 | 정규화, 클리핑 |
| 정규화 | v / ‖v‖ | 단위벡터 | 임베딩 저장 |
| 코사인 유사도 | (a·b) / (‖a‖‖b‖) | [-1, 1] | 추천, 검색 |

**핵심 포인트**:
1. 내적은 AI의 가장 기본적인 연산 (뉴런 = 내적 + 활성화)
2. 코사인 유사도로 "비슷함"을 측정
3. 정규화로 크기 영향 제거, 방향만 비교

---

## 다음 장 예고

3장에서는 행렬 연산을 배웁니다:
- 행렬 곱셈 — 신경망 순전파의 핵심
- 전치 행렬
- 선형 변환의 기하학적 의미
