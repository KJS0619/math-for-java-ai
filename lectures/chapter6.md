# 6장. 경사하강법 — 학습의 엔진

## 6.1 최적화 문제

### AI 학습 = 최적화

신경망 학습의 목표는 **손실 함수를 최소화하는 가중치**를 찾는 것입니다.

$$\mathbf{W}^* = \arg\min_{\mathbf{W}} L(\mathbf{W})$$

```java
// 우리가 풀고 싶은 문제
// weights = findOptimalWeights(lossFunction);
// 여기서 lossFunction(weights)가 최소가 되는 weights를 찾기
```

### 왜 경사하강법인가?

**방법 1: 모든 경우를 시도** - 불가능
```java
// 가중치가 1000개, 각각 100가지 값을 시도하면
// 100^1000 가지 조합 = 우주 나이보다 긴 시간 필요
```

**방법 2: 수학적 해석해** - 대부분 불가능
```java
// 미분 = 0인 점 찾기
// dL/dW = 0 을 풀면 됨
// 하지만 신경망 손실 함수는 너무 복잡해서 해석적 해 없음
```

**방법 3: 경사하강법** - 실용적!
```java
// 현재 위치에서 가장 가파른 내리막 방향으로 조금씩 이동
// 반복하면 최솟값(또는 그 근처)에 도달
```

---

## 6.2 경사하강법의 직관

### 산을 내려가는 비유

```
눈을 가리고 산 정상에 서 있다면, 어떻게 내려갈까?

1. 발로 주변을 더듬어 가장 가파른 내리막 방향을 찾는다
2. 그 방향으로 한 걸음 내딛는다
3. 1-2를 반복한다

이것이 경사하강법!
```

### 수학적 표현

$$W_{new} = W_{old} - \eta \cdot \nabla L(W_{old})$$

- $W_{old}$: 현재 가중치
- $\nabla L$: 손실 함수의 그래디언트 (가장 가파른 상승 방향)
- $-\nabla L$: 가장 가파른 하강 방향
- $\eta$: 학습률 (걸음 크기)
- $W_{new}$: 업데이트된 가중치

```java
public class GradientDescent {
    private double learningRate;

    public GradientDescent(double learningRate) {
        this.learningRate = learningRate;
    }

    // 한 스텝 업데이트
    public double[] step(double[] weights, double[] gradients) {
        double[] newWeights = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            newWeights[i] = weights[i] - learningRate * gradients[i];
        }
        return newWeights;
    }
}
```

---

## 6.3 학습률 (Learning Rate)

### 학습률이란?

한 번에 얼마나 이동할지 결정하는 **하이퍼파라미터**입니다.

```java
// η = 0.01 (작은 학습률)
// W_new = W_old - 0.01 * gradient
// → 조금씩 이동, 안정적이지만 느림

// η = 1.0 (큰 학습률)
// W_new = W_old - 1.0 * gradient
// → 크게 이동, 빠르지만 불안정
```

### 학습률에 따른 행동

```
손실 함수 그래프 (U자 모양):

     Loss
      │     ●  (시작점)
      │    /
      │   /
      │  /
      │ /
      └─●─────────── Weights
       최솟값

학습률이 적절할 때:  ● → ● → ● → ● (목표에 수렴)
학습률이 너무 작을 때: ● → ● → ● → ... (너무 느림)
학습률이 너무 클 때:  ● → ← → ← → (진동하거나 발산)
```

### 학습률 실험

```java
public class LearningRateExperiment {
    // 간단한 함수: f(x) = x² (최솟값: x=0)
    public static double loss(double x) {
        return x * x;
    }

    public static double gradient(double x) {
        return 2 * x;  // f'(x) = 2x
    }

    public static void experiment(double learningRate, int steps) {
        double x = 10.0;  // 시작점

        System.out.printf("학습률 = %.2f%n", learningRate);
        for (int i = 0; i < steps; i++) {
            double grad = gradient(x);
            x = x - learningRate * grad;
            System.out.printf("  Step %d: x=%.4f, loss=%.4f%n",
                i + 1, x, loss(x));
        }
    }

    public static void main(String[] args) {
        // 적절한 학습률
        experiment(0.1, 10);
        // 출력: x가 0에 빠르게 수렴

        // 너무 작은 학습률
        experiment(0.01, 10);
        // 출력: x가 천천히 0에 접근

        // 너무 큰 학습률
        experiment(1.0, 10);
        // 출력: x가 -10, 10, -10... 진동
    }
}
```

### 학습률 선택 가이드

| 학습률 | 일반적 범위 | 특징 |
|-------|-----------|------|
| 매우 작음 | 0.00001 | 안정적, 매우 느림 |
| 작음 | 0.0001 ~ 0.001 | 안정적, 느림 |
| 보통 | 0.001 ~ 0.01 | **일반적으로 시작하는 값** |
| 큼 | 0.1 | 빠르지만 불안정할 수 있음 |
| 매우 큼 | 1.0+ | 대부분 발산 |

---

## 6.4 배치 경사하강법 (Batch GD)

### 정의

**전체 데이터셋**을 사용하여 그래디언트를 계산합니다.

$$\nabla L = \frac{1}{N} \sum_{i=1}^{N} \nabla L_i$$

```java
public class BatchGradientDescent {
    private double learningRate;

    public void train(double[][] X, double[] y, double[] weights, int epochs) {
        int n = X.length;

        for (int epoch = 0; epoch < epochs; epoch++) {
            // 전체 데이터에 대한 그래디언트 계산
            double[] totalGradient = new double[weights.length];

            for (int i = 0; i < n; i++) {
                double[] grad = computeGradient(X[i], y[i], weights);
                for (int j = 0; j < weights.length; j++) {
                    totalGradient[j] += grad[j];
                }
            }

            // 평균 그래디언트로 업데이트
            for (int j = 0; j < weights.length; j++) {
                totalGradient[j] /= n;
                weights[j] -= learningRate * totalGradient[j];
            }

            System.out.printf("Epoch %d: Loss = %.4f%n",
                epoch, computeLoss(X, y, weights));
        }
    }
}
```

### 장단점

**장점**:
- 안정적인 수렴
- 전체 데이터의 정확한 그래디언트

**단점**:
- 데이터가 많으면 매우 느림
- 메모리에 전체 데이터 로드 필요
- 지역 최솟값에 갇힐 수 있음

---

## 6.5 확률적 경사하강법 (SGD)

### 정의

**하나의 샘플**만 사용하여 그래디언트를 계산합니다.

```java
public class StochasticGradientDescent {
    private double learningRate;
    private Random random = new Random();

    public void train(double[][] X, double[] y, double[] weights, int epochs) {
        int n = X.length;

        for (int epoch = 0; epoch < epochs; epoch++) {
            // 데이터 셔플 (중요!)
            int[] indices = shuffle(n);

            for (int idx : indices) {
                // 하나의 샘플로 그래디언트 계산
                double[] grad = computeGradient(X[idx], y[idx], weights);

                // 즉시 업데이트
                for (int j = 0; j < weights.length; j++) {
                    weights[j] -= learningRate * grad[j];
                }
            }

            System.out.printf("Epoch %d: Loss = %.4f%n",
                epoch, computeLoss(X, y, weights));
        }
    }

    private int[] shuffle(int n) {
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        return indices;
    }
}
```

### 장단점

**장점**:
- 빠른 업데이트
- 지역 최솟값 탈출 가능 (노이즈가 도움)
- 메모리 효율적

**단점**:
- 불안정한 수렴 (진동)
- 노이즈가 많은 그래디언트

### SGD의 경로

```
Batch GD:        SGD:
  ●                ●
   \                \
    \                ↘
     \              ↙
      \            ↘
       ●          ↙
                   ●
(직선)        (지그재그)
```

---

## 6.6 미니배치 경사하강법

### 정의

**작은 배치**를 사용하여 그래디언트를 계산합니다. Batch GD와 SGD의 절충안입니다.

```java
public class MiniBatchGradientDescent {
    private double learningRate;
    private int batchSize;

    public MiniBatchGradientDescent(double learningRate, int batchSize) {
        this.learningRate = learningRate;
        this.batchSize = batchSize;
    }

    public void train(double[][] X, double[] y, double[] weights, int epochs) {
        int n = X.length;
        int numBatches = (n + batchSize - 1) / batchSize;

        for (int epoch = 0; epoch < epochs; epoch++) {
            int[] indices = shuffle(n);

            for (int batch = 0; batch < numBatches; batch++) {
                int start = batch * batchSize;
                int end = Math.min(start + batchSize, n);

                // 미니배치의 그래디언트 평균
                double[] batchGradient = new double[weights.length];

                for (int i = start; i < end; i++) {
                    int idx = indices[i];
                    double[] grad = computeGradient(X[idx], y[idx], weights);
                    for (int j = 0; j < weights.length; j++) {
                        batchGradient[j] += grad[j];
                    }
                }

                // 업데이트
                int currentBatchSize = end - start;
                for (int j = 0; j < weights.length; j++) {
                    batchGradient[j] /= currentBatchSize;
                    weights[j] -= learningRate * batchGradient[j];
                }
            }

            System.out.printf("Epoch %d: Loss = %.4f%n",
                epoch, computeLoss(X, y, weights));
        }
    }
}
```

### 배치 크기 선택

| 배치 크기 | 특징 | 일반적 사용 |
|----------|------|-----------|
| 1 | SGD, 많은 노이즈 | 거의 사용 안 함 |
| 16~32 | 작은 배치, GPU 활용 | 메모리 제한적 |
| **64~256** | **균형 잡힌 선택** | **가장 일반적** |
| 512+ | 큰 배치, 안정적 | 대규모 분산 학습 |

### 비교 정리

```java
// Batch GD: 전체 데이터 한 번에
// 업데이트 횟수 per epoch: 1회
for (all samples) { gradient += compute(); }
weights -= lr * gradient / n;

// SGD: 샘플 하나씩
// 업데이트 횟수 per epoch: n회
for (each sample) {
    gradient = compute();
    weights -= lr * gradient;
}

// Mini-batch: 배치 단위
// 업데이트 횟수 per epoch: n/batch_size회
for (each batch) {
    gradient = average(batch);
    weights -= lr * gradient;
}
```

---

## 6.7 모멘텀 (Momentum)

### 문제: 진동과 느린 수렴

```
좁은 골짜기에서의 SGD:

     ↗↘
    ↗  ↘
   ↗    ↘
  ↗      ↘
 ●────────● 목표

좌우로 진동하면서 천천히 전진
```

### 해결: 관성 추가

물리의 관성처럼, **이전 이동 방향을 기억**합니다.

$$v_t = \beta v_{t-1} + \nabla L$$
$$W_t = W_{t-1} - \eta v_t$$

```java
public class MomentumOptimizer {
    private double learningRate;
    private double beta;  // 모멘텀 계수 (보통 0.9)
    private double[] velocity;

    public MomentumOptimizer(double learningRate, double beta, int size) {
        this.learningRate = learningRate;
        this.beta = beta;
        this.velocity = new double[size];  // 0으로 초기화
    }

    public void step(double[] weights, double[] gradients) {
        for (int i = 0; i < weights.length; i++) {
            // 속도 업데이트: 이전 속도 + 현재 그래디언트
            velocity[i] = beta * velocity[i] + gradients[i];

            // 가중치 업데이트
            weights[i] -= learningRate * velocity[i];
        }
    }
}
```

### 모멘텀의 효과

```
모멘텀 적용 후:

     →→→
    →→→→
   →→→→→
  →→→→→→
 ●────────● 목표

진동이 상쇄되고 일관된 방향으로 가속
```

**비유**: 공이 언덕을 굴러 내려갈 때 속도가 붙는 것처럼!

---

## 6.8 Adam 옵티마이저

### Adam = Adaptive Moment Estimation

**모멘텀 + 학습률 자동 조절**을 결합한 가장 인기 있는 옵티마이저입니다.

### 핵심 아이디어

1. **1차 모멘트 (m)**: 그래디언트의 평균 (방향)
2. **2차 모멘트 (v)**: 그래디언트 제곱의 평균 (크기)

```java
public class AdamOptimizer {
    private double learningRate;  // 보통 0.001
    private double beta1;         // 보통 0.9
    private double beta2;         // 보통 0.999
    private double epsilon;       // 보통 1e-8

    private double[] m;  // 1차 모멘트
    private double[] v;  // 2차 모멘트
    private int t;       // 타임스텝

    public AdamOptimizer(double lr, int size) {
        this.learningRate = lr;
        this.beta1 = 0.9;
        this.beta2 = 0.999;
        this.epsilon = 1e-8;
        this.m = new double[size];
        this.v = new double[size];
        this.t = 0;
    }

    public void step(double[] weights, double[] gradients) {
        t++;

        for (int i = 0; i < weights.length; i++) {
            // 1차 모멘트 업데이트 (그래디언트 평균)
            m[i] = beta1 * m[i] + (1 - beta1) * gradients[i];

            // 2차 모멘트 업데이트 (그래디언트 제곱 평균)
            v[i] = beta2 * v[i] + (1 - beta2) * gradients[i] * gradients[i];

            // 편향 보정 (초기값 0으로 인한 편향 제거)
            double mHat = m[i] / (1 - Math.pow(beta1, t));
            double vHat = v[i] / (1 - Math.pow(beta2, t));

            // 가중치 업데이트
            weights[i] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
        }
    }
}
```

### Adam의 장점

1. **학습률 자동 조절**: 그래디언트가 큰 파라미터는 작게, 작은 파라미터는 크게
2. **모멘텀 효과**: 일관된 방향으로 가속
3. **희소한 그래디언트 처리**: NLP, 추천 시스템에 유리

### 옵티마이저 비교

| 옵티마이저 | 특징 | 추천 사용처 |
|-----------|------|-----------|
| SGD | 단순, 튜닝 필요 | 연구, 세밀한 제어 |
| SGD + Momentum | 빠른 수렴 | 컴퓨터 비전 |
| Adam | 적응적 학습률 | **기본 선택**, NLP |
| AdamW | Adam + Weight Decay | 트랜스포머 |

---

## 6.9 학습률 스케줄링

### 왜 필요한가?

초기에는 큰 학습률로 빠르게, 후반에는 작은 학습률로 정밀하게 수렴

### Step Decay

```java
public class StepDecayScheduler {
    private double initialLR;
    private double decayFactor;
    private int decayEvery;

    public StepDecayScheduler(double initialLR, double decayFactor, int decayEvery) {
        this.initialLR = initialLR;
        this.decayFactor = decayFactor;  // 보통 0.1 (10분의 1)
        this.decayEvery = decayEvery;    // 몇 에폭마다
    }

    public double getLearningRate(int epoch) {
        int numDecays = epoch / decayEvery;
        return initialLR * Math.pow(decayFactor, numDecays);
    }
}

// 예: 초기 0.1, 30에폭마다 10분의 1
// epoch 0-29: 0.1
// epoch 30-59: 0.01
// epoch 60-89: 0.001
```

### Exponential Decay

```java
public class ExponentialDecayScheduler {
    private double initialLR;
    private double decayRate;

    public double getLearningRate(int epoch) {
        return initialLR * Math.exp(-decayRate * epoch);
    }
}
```

### Cosine Annealing

```java
public class CosineAnnealingScheduler {
    private double maxLR;
    private double minLR;
    private int totalEpochs;

    public double getLearningRate(int epoch) {
        return minLR + (maxLR - minLR) *
               (1 + Math.cos(Math.PI * epoch / totalEpochs)) / 2;
    }
}

// 코사인 곡선처럼 부드럽게 감소
// 최근 연구에서 좋은 성능을 보임
```

### Warmup

```java
public class WarmupScheduler {
    private double targetLR;
    private int warmupEpochs;

    public double getLearningRate(int epoch) {
        if (epoch < warmupEpochs) {
            // 선형적으로 증가
            return targetLR * (epoch + 1) / warmupEpochs;
        }
        return targetLR;
    }
}

// 트랜스포머 등 큰 모델에서 필수
// 초기 불안정성 방지
```

---

## 6.10 [실습] 완전한 미니배치 학습 루프

### 전체 코드

```java
public class TrainingLoop {
    private double[][] weights;
    private double[] biases;
    private AdamOptimizer optimizer;
    private double learningRate = 0.001;
    private int batchSize = 32;
    private int epochs = 100;

    public void train(double[][] X, double[] y) {
        int n = X.length;
        int inputDim = X[0].length;

        // 가중치 초기화
        weights = new double[1][inputDim];
        biases = new double[1];
        initializeWeights();

        // 옵티마이저 초기화
        int totalParams = inputDim + 1;
        optimizer = new AdamOptimizer(learningRate, totalParams);

        // 학습 루프
        for (int epoch = 0; epoch < epochs; epoch++) {
            int[] indices = shuffle(n);
            double epochLoss = 0;
            int numBatches = 0;

            for (int start = 0; start < n; start += batchSize) {
                int end = Math.min(start + batchSize, n);

                // 미니배치 추출
                double[][] batchX = extractBatch(X, indices, start, end);
                double[] batchY = extractBatchY(y, indices, start, end);

                // Forward pass
                double[] predictions = forward(batchX);

                // Loss 계산
                double loss = computeMSELoss(predictions, batchY);
                epochLoss += loss;
                numBatches++;

                // Backward pass
                double[] dLoss = computeMSEGradient(predictions, batchY);
                double[][] dWeights = computeWeightGradients(batchX, dLoss);
                double[] dBiases = computeBiasGradients(dLoss);

                // 그래디언트를 하나의 배열로
                double[] allGradients = flatten(dWeights, dBiases);
                double[] allWeights = flatten(weights, biases);

                // 옵티마이저로 업데이트
                optimizer.step(allWeights, allGradients);

                // 다시 분리
                unflatten(allWeights, weights, biases);
            }

            if (epoch % 10 == 0) {
                System.out.printf("Epoch %d: Loss = %.4f%n",
                    epoch, epochLoss / numBatches);
            }
        }
    }

    private double[] forward(double[][] X) {
        double[] output = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            output[i] = dotProduct(weights[0], X[i]) + biases[0];
        }
        return output;
    }

    private void initializeWeights() {
        Random random = new Random(42);
        for (int j = 0; j < weights[0].length; j++) {
            // Xavier 초기화
            weights[0][j] = random.nextGaussian() *
                Math.sqrt(2.0 / weights[0].length);
        }
    }
}
```

---

## 6.11 [AI 연결] 실제 프레임워크에서의 경사하강법

### PyTorch 스타일 (Java 버전)

```java
// PyTorch의 학습 루프를 Java로 표현
public class PyTorchStyle {

    public void trainOneEpoch(Model model, DataLoader dataLoader,
                              Optimizer optimizer, LossFunction criterion) {

        for (Batch batch : dataLoader) {
            // 1. 그래디언트 초기화
            optimizer.zeroGrad();

            // 2. Forward pass
            Tensor output = model.forward(batch.inputs);

            // 3. Loss 계산
            Tensor loss = criterion.compute(output, batch.targets);

            // 4. Backward pass (자동 미분)
            loss.backward();

            // 5. 파라미터 업데이트
            optimizer.step();
        }
    }
}
```

### 핵심 메서드 설명

```java
// optimizer.zeroGrad()
// - 이전 배치의 그래디언트를 0으로 리셋
// - 안 하면 그래디언트가 누적됨!

// loss.backward()
// - 연쇄 법칙으로 모든 파라미터의 그래디언트 계산
// - 자동 미분 (Autograd)

// optimizer.step()
// - 계산된 그래디언트로 파라미터 업데이트
// - SGD, Adam 등의 알고리즘 적용
```

### 체크포인트와 Early Stopping

```java
public class TrainingWithCheckpoint {
    private double bestLoss = Double.MAX_VALUE;
    private int patience = 5;
    private int noImproveCount = 0;

    public boolean checkEarlyStopping(double valLoss) {
        if (valLoss < bestLoss) {
            bestLoss = valLoss;
            noImproveCount = 0;
            saveCheckpoint();  // 최고 모델 저장
            return false;
        } else {
            noImproveCount++;
            if (noImproveCount >= patience) {
                System.out.println("Early stopping!");
                return true;
            }
            return false;
        }
    }

    private void saveCheckpoint() {
        // 가중치를 파일로 저장
        // model.pt 같은 체크포인트 파일 생성
    }
}
```

---

## 연습 문제

### 1. 학습률 실험

2차 함수 $f(x) = (x-3)^2$에서 경사하강법으로 최솟값(x=3)을 찾으세요:

```java
// 다양한 학습률로 실험
// 0.01, 0.1, 0.5, 1.0, 1.5
// 각각 몇 스텝 만에 x=3 근처에 도달하는지 확인
```

### 2. 모멘텀 구현

모멘텀 SGD를 구현하고 일반 SGD와 비교하세요:

```java
public class MomentumSGD {
    // beta = 0.9로 모멘텀 적용
    // f(x,y) = x² + 10y² (타원형 등고선)
    // 시작점 (10, 10)에서 최솟값 (0, 0)까지의 경로 비교
}
```

### 3. Adam 구현

Adam 옵티마이저를 완전히 구현하세요:

```java
public class MyAdam {
    // beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8
    // 편향 보정 포함
    // 간단한 회귀 문제에 적용
}
```

### 4. 배치 크기 비교

```java
// 같은 데이터셋에서
// 배치 크기 1, 16, 64, 256으로 학습
// 수렴 속도와 최종 손실 비교
```

---

## 정리

| 개념 | 핵심 | 실무 팁 |
|-----|------|--------|
| 학습률 | 업데이트 크기 | 0.001로 시작, 조절 |
| Batch GD | 전체 데이터 사용 | 작은 데이터셋 |
| SGD | 샘플 하나씩 | 노이즈가 필요할 때 |
| Mini-batch | 절충안 | **가장 일반적** |
| 모멘텀 | 관성 추가 | 진동 감소 |
| Adam | 적응적 + 모멘텀 | **기본 선택** |

**핵심 포인트**:
1. 경사하강법 = 그래디언트 반대 방향으로 이동
2. 학습률이 너무 크면 발산, 너무 작으면 느림
3. Adam을 기본으로 사용, 필요시 SGD + 모멘텀

---

## 다음 장 예고

7장에서는 손실 함수를 배웁니다:
- MSE vs Cross-Entropy
- 손실 함수 선택 기준
- 정규화와 과적합 방지
