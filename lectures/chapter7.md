# 7장. 손실 함수 — 틀린 정도 측정하기

## 7.1 손실 함수란?

### 정의

손실 함수(Loss Function)는 **모델의 예측이 실제 값과 얼마나 다른지** 측정합니다.

$$L = \text{Loss}(\text{예측}, \text{실제})$$

```java
// 손실 함수의 기본 형태
public interface LossFunction {
    double compute(double[] predictions, double[] targets);
    double[] gradient(double[] predictions, double[] targets);
}
```

### 왜 중요한가?

```
학습 과정:
1. 예측 = model.forward(입력)
2. 손실 = Loss(예측, 정답)     ← 여기!
3. 그래디언트 = 손실.backward()
4. 가중치 -= 학습률 × 그래디언트

손실이 0에 가까워지면 → 모델이 정답을 잘 예측
```

### 좋은 손실 함수의 조건

1. **미분 가능**: 그래디언트 계산 필요
2. **최솟값 존재**: 최적화 가능해야 함
3. **의미 있는 측정**: 작을수록 좋은 예측

---

## 7.2 회귀를 위한 손실 함수

### MSE (Mean Squared Error)

가장 기본적인 회귀 손실 함수입니다.

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

```java
public class MSELoss implements LossFunction {

    @Override
    public double compute(double[] predictions, double[] targets) {
        double sum = 0;
        int n = predictions.length;

        for (int i = 0; i < n; i++) {
            double diff = predictions[i] - targets[i];
            sum += diff * diff;
        }

        return sum / n;
    }

    @Override
    public double[] gradient(double[] predictions, double[] targets) {
        int n = predictions.length;
        double[] grad = new double[n];

        for (int i = 0; i < n; i++) {
            // dL/d(pred) = 2(pred - target) / n
            grad[i] = 2.0 * (predictions[i] - targets[i]) / n;
        }

        return grad;
    }
}

// 사용 예
double[] preds = {2.5, 0.0, 2.1, 1.8};
double[] targets = {3.0, -0.5, 2.0, 2.0};

MSELoss mse = new MSELoss();
double loss = mse.compute(preds, targets);
// ((2.5-3)² + (0-(-0.5))² + (2.1-2)² + (1.8-2)²) / 4
// (0.25 + 0.25 + 0.01 + 0.04) / 4 = 0.1375
```

### MSE의 특징

**장점**:
- 계산이 간단
- 그래디언트 계산 용이
- 수학적으로 분석하기 쉬움

**단점**:
- 이상치(outlier)에 민감
- 제곱으로 인해 큰 오차에 더 큰 페널티

```java
// 이상치의 영향
double[] preds = {1, 2, 3, 100};  // 100은 이상치
double[] targets = {1, 2, 3, 4};

// MSE = (0 + 0 + 0 + 9216) / 4 = 2304
// 이상치 하나가 전체 손실을 지배!
```

### MAE (Mean Absolute Error)

절대값 오차의 평균입니다.

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

```java
public class MAELoss implements LossFunction {

    @Override
    public double compute(double[] predictions, double[] targets) {
        double sum = 0;
        int n = predictions.length;

        for (int i = 0; i < n; i++) {
            sum += Math.abs(predictions[i] - targets[i]);
        }

        return sum / n;
    }

    @Override
    public double[] gradient(double[] predictions, double[] targets) {
        int n = predictions.length;
        double[] grad = new double[n];

        for (int i = 0; i < n; i++) {
            // dL/d(pred) = sign(pred - target) / n
            double diff = predictions[i] - targets[i];
            grad[i] = Math.signum(diff) / n;
        }

        return grad;
    }
}
```

### MSE vs MAE

| 특성 | MSE | MAE |
|-----|-----|-----|
| 이상치 | 민감 | 강건 |
| 그래디언트 | 오차에 비례 | 상수 |
| 수렴 | 빠름 | 느림 |
| 미분 | 모든 점에서 가능 | x=0에서 불연속 |

```
오차에 따른 손실:
오차    MSE    MAE
-2      4      2
-1      1      1
 0      0      0
 1      1      1
 2      4      2

MSE: 포물선 (큰 오차에 더 큰 페널티)
MAE: V자 (모든 오차에 동일한 페널티)
```

### Huber Loss

MSE와 MAE의 장점을 결합합니다.

$$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

```java
public class HuberLoss implements LossFunction {
    private double delta;

    public HuberLoss(double delta) {
        this.delta = delta;  // 보통 1.0
    }

    @Override
    public double compute(double[] predictions, double[] targets) {
        double sum = 0;
        int n = predictions.length;

        for (int i = 0; i < n; i++) {
            double diff = Math.abs(predictions[i] - targets[i]);
            if (diff <= delta) {
                // 작은 오차: MSE처럼
                sum += 0.5 * diff * diff;
            } else {
                // 큰 오차: MAE처럼
                sum += delta * diff - 0.5 * delta * delta;
            }
        }

        return sum / n;
    }
}

// 작은 오차에서는 MSE의 부드러운 그래디언트
// 큰 오차(이상치)에서는 MAE의 강건함
```

---

## 7.3 분류를 위한 손실 함수

### Binary Cross-Entropy

이진 분류(0 또는 1)를 위한 손실 함수입니다.

$$\text{BCE} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

```java
public class BinaryCrossEntropyLoss implements LossFunction {
    private double epsilon = 1e-15;  // log(0) 방지

    @Override
    public double compute(double[] predictions, double[] targets) {
        double sum = 0;
        int n = predictions.length;

        for (int i = 0; i < n; i++) {
            // 수치 안정성을 위한 클리핑
            double pred = Math.max(epsilon, Math.min(1 - epsilon, predictions[i]));

            sum += targets[i] * Math.log(pred) +
                   (1 - targets[i]) * Math.log(1 - pred);
        }

        return -sum / n;
    }

    @Override
    public double[] gradient(double[] predictions, double[] targets) {
        int n = predictions.length;
        double[] grad = new double[n];

        for (int i = 0; i < n; i++) {
            double pred = Math.max(epsilon, Math.min(1 - epsilon, predictions[i]));
            // dL/d(pred) = -(target/pred - (1-target)/(1-pred)) / n
            grad[i] = (-(targets[i] / pred) + (1 - targets[i]) / (1 - pred)) / n;
        }

        return grad;
    }
}
```

### BCE의 직관적 이해

```java
// 정답이 1일 때 (y=1):
// Loss = -log(pred)
// pred가 1에 가까우면: Loss ≈ 0 (좋음!)
// pred가 0에 가까우면: Loss → ∞ (나쁨!)

// 정답이 0일 때 (y=0):
// Loss = -log(1-pred)
// pred가 0에 가까우면: Loss ≈ 0 (좋음!)
// pred가 1에 가까우면: Loss → ∞ (나쁨!)

double[] testPreds = {0.9, 0.5, 0.1};
double target = 1.0;

for (double pred : testPreds) {
    double loss = -Math.log(pred);
    System.out.printf("pred=%.1f, loss=%.4f%n", pred, loss);
}
// pred=0.9, loss=0.1054 (확신있게 맞춤)
// pred=0.5, loss=0.6931 (반반)
// pred=0.1, loss=2.3026 (확신있게 틀림 → 큰 페널티)
```

### Categorical Cross-Entropy

다중 클래스 분류를 위한 손실 함수입니다.

$$\text{CCE} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

```java
public class CategoricalCrossEntropyLoss {
    private double epsilon = 1e-15;

    public double compute(double[][] predictions, int[] targets) {
        // predictions: (배치, 클래스 수) - softmax 출력
        // targets: 정답 클래스 인덱스 (0, 1, 2, ...)

        double sum = 0;
        int n = predictions.length;

        for (int i = 0; i < n; i++) {
            int targetClass = targets[i];
            double pred = Math.max(epsilon, predictions[i][targetClass]);
            sum += Math.log(pred);
        }

        return -sum / n;
    }

    public double[][] gradient(double[][] predictions, int[] targets) {
        int n = predictions.length;
        int numClasses = predictions[0].length;
        double[][] grad = new double[n][numClasses];

        for (int i = 0; i < n; i++) {
            for (int c = 0; c < numClasses; c++) {
                // Softmax + Cross-Entropy의 그래디언트
                // = pred - target (아름다운 결과!)
                double target = (c == targets[i]) ? 1.0 : 0.0;
                grad[i][c] = (predictions[i][c] - target) / n;
            }
        }

        return grad;
    }
}

// 사용 예
double[][] preds = {
    {0.7, 0.2, 0.1},  // 샘플 1: 클래스 0 예측
    {0.1, 0.8, 0.1},  // 샘플 2: 클래스 1 예측
    {0.2, 0.2, 0.6}   // 샘플 3: 클래스 2 예측
};
int[] targets = {0, 1, 2};  // 모두 정답!

CategoricalCrossEntropyLoss cce = new CategoricalCrossEntropyLoss();
double loss = cce.compute(preds, targets);
// -1/3 * (log(0.7) + log(0.8) + log(0.6))
// -1/3 * (-0.357 - 0.223 - 0.511) = 0.364
```

### Softmax + Cross-Entropy 결합

```java
public class SoftmaxCrossEntropyLoss {
    // Softmax와 Cross-Entropy를 함께 계산 (수치적으로 안정)

    public double compute(double[][] logits, int[] targets) {
        int n = logits.length;
        double sum = 0;

        for (int i = 0; i < n; i++) {
            // Log-Sum-Exp 트릭
            double maxLogit = max(logits[i]);
            double logSumExp = 0;

            for (double logit : logits[i]) {
                logSumExp += Math.exp(logit - maxLogit);
            }
            logSumExp = maxLogit + Math.log(logSumExp);

            // Cross-Entropy
            sum += logits[i][targets[i]] - logSumExp;
        }

        return -sum / n;
    }

    // 그래디언트가 매우 간단!
    public double[][] gradient(double[][] logits, int[] targets) {
        int n = logits.length;
        int numClasses = logits[0].length;
        double[][] grad = new double[n][numClasses];

        for (int i = 0; i < n; i++) {
            double[] probs = softmax(logits[i]);
            for (int c = 0; c < numClasses; c++) {
                double target = (c == targets[i]) ? 1.0 : 0.0;
                grad[i][c] = (probs[c] - target) / n;
            }
        }

        return grad;
    }
}
```

---

## 7.4 손실 함수 선택 가이드

### 문제 유형별 선택

| 문제 유형 | 출력층 활성화 | 손실 함수 |
|----------|-------------|----------|
| 회귀 | None (선형) | MSE, MAE, Huber |
| 이진 분류 | Sigmoid | Binary Cross-Entropy |
| 다중 분류 | Softmax | Categorical Cross-Entropy |
| 다중 레이블 | Sigmoid | Binary CE (각 레이블) |

### 코드로 정리

```java
public class LossFunctionFactory {

    public static LossFunction create(String problemType) {
        switch (problemType.toLowerCase()) {
            case "regression":
                return new MSELoss();

            case "regression_robust":
                return new HuberLoss(1.0);

            case "binary_classification":
                return new BinaryCrossEntropyLoss();

            case "multiclass_classification":
                return new CategoricalCrossEntropyLoss();

            default:
                throw new IllegalArgumentException(
                    "Unknown problem type: " + problemType);
        }
    }
}

// 사용
LossFunction loss = LossFunctionFactory.create("binary_classification");
```

---

## 7.5 정규화 (Regularization)

### 과적합 문제

```
훈련 데이터:    ●  ●  ●  ●  ●
               ↑  적절한 모델: ~~~
               ↑  과적합 모델: ∿∿∿∿∿ (모든 점을 완벽히 통과)

과적합 모델은 훈련 데이터는 완벽히 맞추지만
새로운 데이터에 대해서는 성능이 나쁨
```

### L2 정규화 (Weight Decay)

가중치의 제곱합에 페널티를 부여합니다.

$$L_{total} = L_{data} + \lambda \sum_{i} w_i^2$$

```java
public class L2RegularizedLoss implements LossFunction {
    private LossFunction baseLoss;
    private double lambda;  // 정규화 강도

    public L2RegularizedLoss(LossFunction baseLoss, double lambda) {
        this.baseLoss = baseLoss;
        this.lambda = lambda;  // 보통 0.001 ~ 0.01
    }

    public double compute(double[] predictions, double[] targets,
                         double[] weights) {
        // 기본 손실
        double dataLoss = baseLoss.compute(predictions, targets);

        // L2 정규화 항
        double regLoss = 0;
        for (double w : weights) {
            regLoss += w * w;
        }

        return dataLoss + lambda * regLoss;
    }

    public double[] weightGradient(double[] weights) {
        // L2 정규화의 그래디언트 = 2λw
        double[] grad = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            grad[i] = 2 * lambda * weights[i];
        }
        return grad;
    }
}

// 효과: 큰 가중치에 페널티 → 가중치가 작아짐 → 단순한 모델
```

### L1 정규화 (Lasso)

가중치의 절대값 합에 페널티를 부여합니다.

$$L_{total} = L_{data} + \lambda \sum_{i} |w_i|$$

```java
public class L1RegularizedLoss implements LossFunction {
    private LossFunction baseLoss;
    private double lambda;

    public double compute(double[] predictions, double[] targets,
                         double[] weights) {
        double dataLoss = baseLoss.compute(predictions, targets);

        double regLoss = 0;
        for (double w : weights) {
            regLoss += Math.abs(w);
        }

        return dataLoss + lambda * regLoss;
    }

    public double[] weightGradient(double[] weights) {
        // L1 정규화의 그래디언트 = λ * sign(w)
        double[] grad = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            grad[i] = lambda * Math.signum(weights[i]);
        }
        return grad;
    }
}

// 효과: 일부 가중치가 정확히 0이 됨 → 희소한 모델 → 특성 선택
```

### L1 vs L2 비교

| 특성 | L1 | L2 |
|-----|----|----|
| 페널티 | $\|w\|$ | $w^2$ |
| 효과 | 희소성 (0이 많음) | 가중치 축소 |
| 특성 선택 | 자동으로 수행 | 모든 특성 사용 |
| 미분 | x=0에서 불연속 | 연속 |

### Dropout

훈련 중 무작위로 뉴런을 비활성화합니다.

```java
public class DropoutLayer {
    private double dropRate;  // 보통 0.5
    private Random random = new Random();
    private boolean[] mask;

    public DropoutLayer(double dropRate) {
        this.dropRate = dropRate;
    }

    public double[] forward(double[] input, boolean training) {
        double[] output = new double[input.length];
        mask = new boolean[input.length];

        if (training) {
            // 훈련 시: 랜덤하게 드롭
            double scale = 1.0 / (1.0 - dropRate);  // 스케일 보정

            for (int i = 0; i < input.length; i++) {
                if (random.nextDouble() > dropRate) {
                    mask[i] = true;
                    output[i] = input[i] * scale;
                } else {
                    mask[i] = false;
                    output[i] = 0;
                }
            }
        } else {
            // 추론 시: 그대로 통과
            System.arraycopy(input, 0, output, 0, input.length);
        }

        return output;
    }

    public double[] backward(double[] gradOutput) {
        double[] gradInput = new double[gradOutput.length];
        double scale = 1.0 / (1.0 - dropRate);

        for (int i = 0; i < gradOutput.length; i++) {
            gradInput[i] = mask[i] ? gradOutput[i] * scale : 0;
        }

        return gradInput;
    }
}
```

---

## 7.6 손실 곡선 분석

### 정상적인 학습

```
Loss
 │
 │●
 │ ●
 │  ●●
 │    ●●●
 │       ●●●●●●●●
 └─────────────────→ Epoch

특징:
- 초반에 급격히 감소
- 후반에 완만하게 수렴
- 훈련/검증 손실이 비슷하게 감소
```

### 과적합

```
Loss
 │         ╱ 검증 손실 (증가!)
 │        ╱
 │       ╱
 │●     ╱
 │ ●●  ╱
 │   ●●───── 훈련 손실 (계속 감소)
 └─────────────────→ Epoch

특징:
- 훈련 손실은 계속 감소
- 검증 손실이 증가하기 시작
- 이 지점에서 Early Stopping!
```

### 학습 불안정

```
Loss
 │    ●
 │   ● ●    ●
 │  ●   ●  ● ●
 │ ●     ●●   ●
 │●
 └─────────────────→ Epoch

원인:
- 학습률이 너무 큼
- 배치 크기가 너무 작음
- 그래디언트 폭발
```

### 학습 정체

```
Loss
 │●●
 │  ●●●●●●●●●●●●●●
 │
 │
 └─────────────────→ Epoch

원인:
- 학습률이 너무 작음
- 지역 최솟값에 갇힘
- 모델 용량 부족
```

### Java로 손실 기록

```java
public class TrainingHistory {
    private List<Double> trainLosses = new ArrayList<>();
    private List<Double> valLosses = new ArrayList<>();
    private int bestEpoch = 0;
    private double bestValLoss = Double.MAX_VALUE;

    public void record(int epoch, double trainLoss, double valLoss) {
        trainLosses.add(trainLoss);
        valLosses.add(valLoss);

        if (valLoss < bestValLoss) {
            bestValLoss = valLoss;
            bestEpoch = epoch;
        }

        // 과적합 경고
        if (trainLosses.size() > 5) {
            double recentTrainAvg = average(trainLosses, -5);
            double recentValAvg = average(valLosses, -5);

            if (recentValAvg > recentTrainAvg * 1.2) {
                System.out.println("Warning: Possible overfitting detected!");
            }
        }
    }

    public void printSummary() {
        System.out.println("=== Training Summary ===");
        System.out.printf("Best epoch: %d (val_loss: %.4f)%n",
            bestEpoch, bestValLoss);
        System.out.printf("Final train_loss: %.4f%n",
            trainLosses.get(trainLosses.size() - 1));
        System.out.printf("Final val_loss: %.4f%n",
            valLosses.get(valLosses.size() - 1));
    }
}
```

---

## 7.7 [AI 연결] 실제 학습 파이프라인

### 완전한 학습 예제

```java
public class CompleteTraining {
    // 하이퍼파라미터
    private int epochs = 100;
    private int batchSize = 32;
    private double learningRate = 0.001;
    private double l2Lambda = 0.0001;
    private double dropoutRate = 0.5;
    private int patience = 10;

    public void train(double[][] trainX, double[] trainY,
                     double[][] valX, double[] valY) {

        // 모델 구성요소
        SimpleNetwork model = new SimpleNetwork(trainX[0].length, 128, 1);
        AdamOptimizer optimizer = new AdamOptimizer(learningRate,
                                                    model.numParams());
        LossFunction loss = new MSELoss();
        TrainingHistory history = new TrainingHistory();

        // Early Stopping 변수
        double bestValLoss = Double.MAX_VALUE;
        int noImproveCount = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            // === 훈련 ===
            model.setTraining(true);
            double epochTrainLoss = 0;
            int numBatches = 0;

            int[] indices = shuffle(trainX.length);

            for (int start = 0; start < trainX.length; start += batchSize) {
                int end = Math.min(start + batchSize, trainX.length);

                // 미니배치 추출
                double[][] batchX = slice(trainX, indices, start, end);
                double[] batchY = slice(trainY, indices, start, end);

                // Forward
                double[] preds = model.forward(batchX);

                // Loss 계산 (+ L2 정규화)
                double batchLoss = loss.compute(preds, batchY);
                batchLoss += l2Lambda * model.l2Norm();
                epochTrainLoss += batchLoss;
                numBatches++;

                // Backward
                double[] dLoss = loss.gradient(preds, batchY);
                double[] grads = model.backward(dLoss);

                // L2 정규화 그래디언트 추가
                double[] l2Grads = model.l2Gradient(l2Lambda);
                addInPlace(grads, l2Grads);

                // 업데이트
                optimizer.step(model.params(), grads);
            }

            epochTrainLoss /= numBatches;

            // === 검증 ===
            model.setTraining(false);
            double[] valPreds = model.forward(valX);
            double valLoss = loss.compute(valPreds, valY);

            // 기록
            history.record(epoch, epochTrainLoss, valLoss);

            // Early Stopping 체크
            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                noImproveCount = 0;
                model.saveCheckpoint("best_model.bin");
            } else {
                noImproveCount++;
                if (noImproveCount >= patience) {
                    System.out.printf("Early stopping at epoch %d%n", epoch);
                    break;
                }
            }

            // 로깅
            if (epoch % 10 == 0) {
                System.out.printf("Epoch %d: train=%.4f, val=%.4f%n",
                    epoch, epochTrainLoss, valLoss);
            }
        }

        // 최고 모델 로드
        model.loadCheckpoint("best_model.bin");
        history.printSummary();
    }
}
```

### 손실 함수 커스터마이징

```java
// 도메인에 특화된 손실 함수 예
public class WeightedMSELoss implements LossFunction {
    private double[] sampleWeights;

    public WeightedMSELoss(double[] sampleWeights) {
        this.sampleWeights = sampleWeights;
    }

    @Override
    public double compute(double[] predictions, double[] targets) {
        double sum = 0;
        double totalWeight = 0;

        for (int i = 0; i < predictions.length; i++) {
            double diff = predictions[i] - targets[i];
            sum += sampleWeights[i] * diff * diff;
            totalWeight += sampleWeights[i];
        }

        return sum / totalWeight;
    }
}

// 불균형 데이터에서 소수 클래스에 높은 가중치
// 중요한 샘플에 더 큰 가중치
// 최근 데이터에 더 큰 가중치 (시계열)
```

---

## 연습 문제

### 1. 손실 함수 비교

```java
// 동일한 예측과 타겟에 대해 MSE, MAE, Huber 비교
double[] preds = {2.5, 0.0, 2.1, 7.8};  // 마지막이 이상치
double[] targets = {3.0, -0.5, 2.0, 2.0};

// 각 손실 함수의 값 계산
// 이상치가 전체 손실에 미치는 영향 분석
```

### 2. Cross-Entropy 구현

```java
// One-hot 인코딩된 타겟을 사용하는
// Categorical Cross-Entropy 구현
public class OneHotCrossEntropy {
    public double compute(double[][] predictions, double[][] targets) {
        // targets: (배치, 클래스) - one-hot 인코딩
        // 구현하세요
    }
}
```

### 3. L1 + L2 정규화

```java
// Elastic Net: L1 + L2 결합
// Loss = DataLoss + α * L1 + β * L2
public class ElasticNetLoss {
    // 구현하세요
}
```

### 4. 학습 곡선 분석

```java
// 훈련/검증 손실 리스트가 주어졌을 때
// - 과적합 여부 판단
// - 최적의 에폭 찾기
// - 학습이 정상적인지 판단
public class LossCurveAnalyzer {
    public String analyze(List<Double> trainLoss, List<Double> valLoss) {
        // 구현하세요
    }
}
```

---

## 정리

| 손실 함수 | 용도 | 특징 |
|----------|------|------|
| MSE | 회귀 | 간단, 이상치 민감 |
| MAE | 회귀 | 이상치 강건 |
| Huber | 회귀 | MSE + MAE 장점 |
| Binary CE | 이진 분류 | Sigmoid와 함께 |
| Categorical CE | 다중 분류 | Softmax와 함께 |
| L2 정규화 | 과적합 방지 | 가중치 축소 |
| L1 정규화 | 특성 선택 | 희소성 유도 |

**핵심 포인트**:
1. 문제에 맞는 손실 함수 선택이 중요
2. 정규화로 과적합 방지
3. 손실 곡선으로 학습 상태 모니터링

---

## 다음 장 예고

8장에서는 실전 MNIST 분류기를 만듭니다:
- 지금까지 배운 모든 개념 통합
- 손글씨 숫자 인식 신경망
- 완전한 학습 파이프라인
