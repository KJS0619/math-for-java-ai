# 8장. 실전 프로젝트 — MNIST 손글씨 분류기

## 8.1 프로젝트 개요

### 목표

지금까지 배운 모든 개념을 통합하여 **손글씨 숫자를 인식하는 신경망**을 만듭니다.

```
입력: 28×28 픽셀 손글씨 이미지 (784차원 벡터)
출력: 0~9 숫자 중 하나 (10개 클래스)

[이미지] → [신경망] → [확률] → [예측]
   ↓           ↓          ↓        ↓
 784차원    3개 레이어   10차원   argmax
```

### 사용할 개념들

| 장 | 개념 | 이 프로젝트에서의 적용 |
|---|------|---------------------|
| 1장 | 벡터, 행렬 | 이미지를 벡터로, 가중치를 행렬로 |
| 2장 | 내적, 정규화 | 레이어 계산, 입력 전처리 |
| 3장 | 행렬 곱셈 | 순전파 계산 |
| 4장 | 활성화 함수 | ReLU, Softmax |
| 5장 | 미분, 연쇄 법칙 | 역전파 |
| 6장 | 경사하강법 | Adam 옵티마이저 |
| 7장 | 손실 함수 | Cross-Entropy |

---

## 8.2 데이터 준비

### MNIST 데이터셋

```java
public class MNISTLoader {
    private static final String DATA_PATH = "data/mnist/";

    // MNIST 파일 포맷: IDX (바이너리)
    // 이미지: [매직넘버][개수][행][열][픽셀들...]
    // 레이블: [매직넘버][개수][레이블들...]

    public static double[][] loadImages(String filename) throws IOException {
        DataInputStream dis = new DataInputStream(
            new FileInputStream(DATA_PATH + filename));

        int magic = dis.readInt();       // 2051 (이미지 파일)
        int count = dis.readInt();       // 이미지 개수
        int rows = dis.readInt();        // 28
        int cols = dis.readInt();        // 28

        double[][] images = new double[count][rows * cols];

        for (int i = 0; i < count; i++) {
            for (int j = 0; j < rows * cols; j++) {
                // 0~255 픽셀 → 0~1 정규화
                images[i][j] = (dis.readUnsignedByte() & 0xFF) / 255.0;
            }
        }

        dis.close();
        return images;
    }

    public static int[] loadLabels(String filename) throws IOException {
        DataInputStream dis = new DataInputStream(
            new FileInputStream(DATA_PATH + filename));

        int magic = dis.readInt();       // 2049 (레이블 파일)
        int count = dis.readInt();       // 레이블 개수

        int[] labels = new int[count];
        for (int i = 0; i < count; i++) {
            labels[i] = dis.readUnsignedByte();
        }

        dis.close();
        return labels;
    }

    public static void main(String[] args) throws IOException {
        // 훈련 데이터: 60,000개
        double[][] trainImages = loadImages("train-images-idx3-ubyte");
        int[] trainLabels = loadLabels("train-labels-idx1-ubyte");

        // 테스트 데이터: 10,000개
        double[][] testImages = loadImages("t10k-images-idx3-ubyte");
        int[] testLabels = loadLabels("t10k-labels-idx1-ubyte");

        System.out.println("훈련 이미지: " + trainImages.length);
        System.out.println("테스트 이미지: " + testImages.length);
        System.out.println("입력 차원: " + trainImages[0].length);
    }
}
```

### 데이터 시각화 (ASCII)

```java
public class MNISTVisualizer {

    public static void printImage(double[] image, int label) {
        System.out.println("Label: " + label);
        System.out.println("─".repeat(30));

        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                double pixel = image[row * 28 + col];
                // 밝기에 따른 문자 선택
                if (pixel < 0.2) System.out.print(" ");
                else if (pixel < 0.4) System.out.print("░");
                else if (pixel < 0.6) System.out.print("▒");
                else if (pixel < 0.8) System.out.print("▓");
                else System.out.print("█");
            }
            System.out.println();
        }
    }
}

// 출력 예시:
// Label: 5
// ──────────────────────────────
//
//            ░▒▓██▓▒
//          ▒███████▓░
//         ▒███▓░░░░
//         ▓███░
//          ░███▓▒
//            ░▓███▓░
//               ░▓██▓
//                 ▓██
//               ░▓██▓
//            ░▓████▒
//          ▒█████▒
//          ░░░░░
```

### 데이터 분할

```java
public class DataSplitter {

    public static class Dataset {
        public double[][] images;
        public int[] labels;

        public Dataset(double[][] images, int[] labels) {
            this.images = images;
            this.labels = labels;
        }
    }

    public static Dataset[] trainValidationSplit(
            double[][] images, int[] labels, double validationRatio) {

        int n = images.length;
        int valSize = (int)(n * validationRatio);
        int trainSize = n - valSize;

        // 셔플
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        shuffle(indices);

        // 분할
        double[][] trainImages = new double[trainSize][];
        int[] trainLabels = new int[trainSize];
        double[][] valImages = new double[valSize][];
        int[] valLabels = new int[valSize];

        for (int i = 0; i < trainSize; i++) {
            trainImages[i] = images[indices[i]];
            trainLabels[i] = labels[indices[i]];
        }
        for (int i = 0; i < valSize; i++) {
            valImages[i] = images[indices[trainSize + i]];
            valLabels[i] = labels[indices[trainSize + i]];
        }

        return new Dataset[] {
            new Dataset(trainImages, trainLabels),
            new Dataset(valImages, valLabels)
        };
    }
}
```

---

## 8.3 신경망 구조

### 네트워크 아키텍처

```
입력층: 784 (28×28 픽셀)
    ↓
은닉층1: 256 뉴런 + ReLU
    ↓
은닉층2: 128 뉴런 + ReLU
    ↓
출력층: 10 (숫자 0~9) + Softmax

총 파라미터:
- W1: 784 × 256 = 200,704
- b1: 256
- W2: 256 × 128 = 32,768
- b2: 128
- W3: 128 × 10 = 1,280
- b3: 10
총: 235,146개
```

### 레이어 클래스

```java
public class DenseLayer {
    private double[][] weights;  // (inputSize, outputSize)
    private double[] biases;     // (outputSize)
    private double[] lastInput;  // 역전파용
    private double[] lastOutput; // 역전파용

    // 그래디언트 저장
    private double[][] dWeights;
    private double[] dBiases;

    public DenseLayer(int inputSize, int outputSize) {
        // He 초기화 (ReLU에 적합)
        double scale = Math.sqrt(2.0 / inputSize);
        Random random = new Random(42);

        weights = new double[inputSize][outputSize];
        biases = new double[outputSize];

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] = random.nextGaussian() * scale;
            }
        }
        // bias는 0으로 초기화
    }

    public double[] forward(double[] input) {
        lastInput = input;
        double[] output = new double[biases.length];

        // output = input @ weights + bias
        for (int j = 0; j < output.length; j++) {
            double sum = biases[j];
            for (int i = 0; i < input.length; i++) {
                sum += input[i] * weights[i][j];
            }
            output[j] = sum;
        }

        lastOutput = output;
        return output;
    }

    public double[] backward(double[] gradOutput) {
        int inputSize = weights.length;
        int outputSize = weights[0].length;

        // 가중치 그래디언트: dW = input.T @ gradOutput
        dWeights = new double[inputSize][outputSize];
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                dWeights[i][j] = lastInput[i] * gradOutput[j];
            }
        }

        // 편향 그래디언트: dB = gradOutput
        dBiases = gradOutput.clone();

        // 입력 그래디언트: dInput = gradOutput @ weights.T
        double[] gradInput = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                gradInput[i] += gradOutput[j] * weights[i][j];
            }
        }

        return gradInput;
    }

    // getter
    public double[][] getWeights() { return weights; }
    public double[] getBiases() { return biases; }
    public double[][] getDWeights() { return dWeights; }
    public double[] getDBiases() { return dBiases; }
}
```

### 활성화 함수 클래스

```java
public class ReLULayer {
    private boolean[] mask;

    public double[] forward(double[] input) {
        double[] output = new double[input.length];
        mask = new boolean[input.length];

        for (int i = 0; i < input.length; i++) {
            if (input[i] > 0) {
                output[i] = input[i];
                mask[i] = true;
            } else {
                output[i] = 0;
                mask[i] = false;
            }
        }
        return output;
    }

    public double[] backward(double[] gradOutput) {
        double[] gradInput = new double[gradOutput.length];
        for (int i = 0; i < gradOutput.length; i++) {
            gradInput[i] = mask[i] ? gradOutput[i] : 0;
        }
        return gradInput;
    }
}

public class SoftmaxLayer {
    private double[] lastOutput;

    public double[] forward(double[] input) {
        // 수치 안정성을 위해 최대값 빼기
        double max = Double.NEGATIVE_INFINITY;
        for (double v : input) max = Math.max(max, v);

        double[] exp = new double[input.length];
        double sum = 0;
        for (int i = 0; i < input.length; i++) {
            exp[i] = Math.exp(input[i] - max);
            sum += exp[i];
        }

        lastOutput = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            lastOutput[i] = exp[i] / sum;
        }
        return lastOutput;
    }

    // Softmax + CrossEntropy의 그래디언트는 간단: pred - target
    public double[] backward(int targetClass) {
        double[] gradInput = lastOutput.clone();
        gradInput[targetClass] -= 1.0;
        return gradInput;
    }
}
```

### 전체 네트워크

```java
public class MNISTNetwork {
    private DenseLayer fc1;  // 784 → 256
    private ReLULayer relu1;
    private DenseLayer fc2;  // 256 → 128
    private ReLULayer relu2;
    private DenseLayer fc3;  // 128 → 10
    private SoftmaxLayer softmax;

    public MNISTNetwork() {
        fc1 = new DenseLayer(784, 256);
        relu1 = new ReLULayer();
        fc2 = new DenseLayer(256, 128);
        relu2 = new ReLULayer();
        fc3 = new DenseLayer(128, 10);
        softmax = new SoftmaxLayer();
    }

    public double[] forward(double[] image) {
        double[] x = fc1.forward(image);
        x = relu1.forward(x);
        x = fc2.forward(x);
        x = relu2.forward(x);
        x = fc3.forward(x);
        x = softmax.forward(x);
        return x;
    }

    public void backward(int targetLabel) {
        // Softmax + CrossEntropy 역전파
        double[] grad = softmax.backward(targetLabel);

        // fc3 역전파
        grad = fc3.backward(grad);

        // relu2 역전파
        grad = relu2.backward(grad);

        // fc2 역전파
        grad = fc2.backward(grad);

        // relu1 역전파
        grad = relu1.backward(grad);

        // fc1 역전파
        fc1.backward(grad);
    }

    public int predict(double[] image) {
        double[] probs = forward(image);
        int maxIdx = 0;
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > probs[maxIdx]) maxIdx = i;
        }
        return maxIdx;
    }

    // 모든 레이어 반환 (옵티마이저용)
    public DenseLayer[] getDenseLayers() {
        return new DenseLayer[] { fc1, fc2, fc3 };
    }
}
```

---

## 8.4 손실 함수

### Cross-Entropy 손실

```java
public class CrossEntropyLoss {
    private double epsilon = 1e-15;

    public double compute(double[] predictions, int targetClass) {
        // L = -log(pred[target])
        double pred = Math.max(epsilon, predictions[targetClass]);
        return -Math.log(pred);
    }

    // 배치 손실
    public double computeBatch(double[][] predictions, int[] targets) {
        double totalLoss = 0;
        for (int i = 0; i < predictions.length; i++) {
            totalLoss += compute(predictions[i], targets[i]);
        }
        return totalLoss / predictions.length;
    }
}
```

---

## 8.5 옵티마이저

### Adam 구현

```java
public class AdamOptimizer {
    private double learningRate;
    private double beta1 = 0.9;
    private double beta2 = 0.999;
    private double epsilon = 1e-8;

    // 각 레이어별 모멘트
    private Map<DenseLayer, double[][]> mWeights = new HashMap<>();
    private Map<DenseLayer, double[][]> vWeights = new HashMap<>();
    private Map<DenseLayer, double[]> mBiases = new HashMap<>();
    private Map<DenseLayer, double[]> vBiases = new HashMap<>();
    private int t = 0;

    public AdamOptimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    public void step(DenseLayer[] layers) {
        t++;

        for (DenseLayer layer : layers) {
            // 초기화 (처음 호출 시)
            if (!mWeights.containsKey(layer)) {
                double[][] w = layer.getWeights();
                double[] b = layer.getBiases();
                mWeights.put(layer, new double[w.length][w[0].length]);
                vWeights.put(layer, new double[w.length][w[0].length]);
                mBiases.put(layer, new double[b.length]);
                vBiases.put(layer, new double[b.length]);
            }

            // 가중치 업데이트
            double[][] w = layer.getWeights();
            double[][] dw = layer.getDWeights();
            double[][] m = mWeights.get(layer);
            double[][] v = vWeights.get(layer);

            for (int i = 0; i < w.length; i++) {
                for (int j = 0; j < w[0].length; j++) {
                    m[i][j] = beta1 * m[i][j] + (1 - beta1) * dw[i][j];
                    v[i][j] = beta2 * v[i][j] + (1 - beta2) * dw[i][j] * dw[i][j];

                    double mHat = m[i][j] / (1 - Math.pow(beta1, t));
                    double vHat = v[i][j] / (1 - Math.pow(beta2, t));

                    w[i][j] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
                }
            }

            // 편향 업데이트
            double[] b = layer.getBiases();
            double[] db = layer.getDBiases();
            double[] mb = mBiases.get(layer);
            double[] vb = vBiases.get(layer);

            for (int i = 0; i < b.length; i++) {
                mb[i] = beta1 * mb[i] + (1 - beta1) * db[i];
                vb[i] = beta2 * vb[i] + (1 - beta2) * db[i] * db[i];

                double mHat = mb[i] / (1 - Math.pow(beta1, t));
                double vHat = vb[i] / (1 - Math.pow(beta2, t));

                b[i] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
            }
        }
    }
}
```

---

## 8.6 학습 루프

### 메인 학습 코드

```java
public class MNISTTrainer {
    // 하이퍼파라미터
    private int epochs = 10;
    private int batchSize = 32;
    private double learningRate = 0.001;

    private MNISTNetwork network;
    private AdamOptimizer optimizer;
    private CrossEntropyLoss lossFunction;

    public MNISTTrainer() {
        network = new MNISTNetwork();
        optimizer = new AdamOptimizer(learningRate);
        lossFunction = new CrossEntropyLoss();
    }

    public void train(double[][] trainImages, int[] trainLabels,
                     double[][] valImages, int[] valLabels) {

        int n = trainImages.length;

        for (int epoch = 0; epoch < epochs; epoch++) {
            double epochLoss = 0;
            int correct = 0;

            // 데이터 셔플
            int[] indices = shuffle(n);

            // 미니배치 학습
            for (int start = 0; start < n; start += batchSize) {
                int end = Math.min(start + batchSize, n);
                double batchLoss = 0;

                for (int i = start; i < end; i++) {
                    int idx = indices[i];
                    double[] image = trainImages[idx];
                    int label = trainLabels[idx];

                    // Forward
                    double[] probs = network.forward(image);

                    // Loss
                    batchLoss += lossFunction.compute(probs, label);

                    // Accuracy
                    int pred = argmax(probs);
                    if (pred == label) correct++;

                    // Backward
                    network.backward(label);
                }

                // 배치 평균 그래디언트로 업데이트
                optimizer.step(network.getDenseLayers());

                epochLoss += batchLoss;
            }

            epochLoss /= n;
            double trainAcc = (double) correct / n;

            // 검증
            double valAcc = evaluate(valImages, valLabels);

            System.out.printf("Epoch %d/%d - Loss: %.4f - Acc: %.4f - Val_Acc: %.4f%n",
                epoch + 1, epochs, epochLoss, trainAcc, valAcc);
        }
    }

    public double evaluate(double[][] images, int[] labels) {
        int correct = 0;
        for (int i = 0; i < images.length; i++) {
            int pred = network.predict(images[i]);
            if (pred == labels[i]) correct++;
        }
        return (double) correct / images.length;
    }

    private int argmax(double[] arr) {
        int maxIdx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[maxIdx]) maxIdx = i;
        }
        return maxIdx;
    }

    private int[] shuffle(int n) {
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        Random random = new Random();
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

### 실행

```java
public class Main {
    public static void main(String[] args) throws IOException {
        System.out.println("=== MNIST 손글씨 분류기 ===\n");

        // 1. 데이터 로드
        System.out.println("데이터 로딩...");
        double[][] trainImages = MNISTLoader.loadImages("train-images-idx3-ubyte");
        int[] trainLabels = MNISTLoader.loadLabels("train-labels-idx1-ubyte");
        double[][] testImages = MNISTLoader.loadImages("t10k-images-idx3-ubyte");
        int[] testLabels = MNISTLoader.loadLabels("t10k-labels-idx1-ubyte");

        // 2. 훈련/검증 분할
        System.out.println("데이터 분할...");
        DataSplitter.Dataset[] splits = DataSplitter.trainValidationSplit(
            trainImages, trainLabels, 0.1);
        double[][] trainX = splits[0].images;
        int[] trainY = splits[0].labels;
        double[][] valX = splits[1].images;
        int[] valY = splits[1].labels;

        System.out.printf("훈련: %d, 검증: %d, 테스트: %d%n%n",
            trainX.length, valX.length, testImages.length);

        // 3. 학습
        System.out.println("학습 시작...\n");
        MNISTTrainer trainer = new MNISTTrainer();
        trainer.train(trainX, trainY, valX, valY);

        // 4. 테스트
        System.out.println("\n최종 테스트...");
        double testAcc = trainer.evaluate(testImages, testLabels);
        System.out.printf("테스트 정확도: %.2f%%\n", testAcc * 100);
    }
}
```

### 예상 출력

```
=== MNIST 손글씨 분류기 ===

데이터 로딩...
데이터 분할...
훈련: 54000, 검증: 6000, 테스트: 10000

학습 시작...

Epoch 1/10 - Loss: 0.3521 - Acc: 0.8923 - Val_Acc: 0.9342
Epoch 2/10 - Loss: 0.1432 - Acc: 0.9567 - Val_Acc: 0.9523
Epoch 3/10 - Loss: 0.0987 - Acc: 0.9701 - Val_Acc: 0.9612
Epoch 4/10 - Loss: 0.0721 - Acc: 0.9783 - Val_Acc: 0.9678
Epoch 5/10 - Loss: 0.0543 - Acc: 0.9834 - Val_Acc: 0.9702
Epoch 6/10 - Loss: 0.0421 - Acc: 0.9871 - Val_Acc: 0.9715
Epoch 7/10 - Loss: 0.0332 - Acc: 0.9898 - Val_Acc: 0.9723
Epoch 8/10 - Loss: 0.0267 - Acc: 0.9916 - Val_Acc: 0.9731
Epoch 9/10 - Loss: 0.0218 - Acc: 0.9931 - Val_Acc: 0.9738
Epoch 10/10 - Loss: 0.0179 - Acc: 0.9943 - Val_Acc: 0.9745

최종 테스트...
테스트 정확도: 97.32%
```

---

## 8.7 모델 저장과 로드

### 직렬화

```java
public class ModelSerializer {

    public static void save(MNISTNetwork network, String filename)
            throws IOException {
        ObjectOutputStream oos = new ObjectOutputStream(
            new FileOutputStream(filename));

        for (DenseLayer layer : network.getDenseLayers()) {
            oos.writeObject(layer.getWeights());
            oos.writeObject(layer.getBiases());
        }

        oos.close();
        System.out.println("모델 저장됨: " + filename);
    }

    public static MNISTNetwork load(String filename)
            throws IOException, ClassNotFoundException {
        ObjectInputStream ois = new ObjectInputStream(
            new FileInputStream(filename));

        MNISTNetwork network = new MNISTNetwork();
        DenseLayer[] layers = network.getDenseLayers();

        for (DenseLayer layer : layers) {
            double[][] weights = (double[][]) ois.readObject();
            double[] biases = (double[]) ois.readObject();
            // 가중치 복사 (private 필드라면 별도 메서드 필요)
            copyWeights(layer, weights, biases);
        }

        ois.close();
        System.out.println("모델 로드됨: " + filename);
        return network;
    }
}

// 사용
ModelSerializer.save(network, "mnist_model.bin");
MNISTNetwork loadedNetwork = ModelSerializer.load("mnist_model.bin");
```

---

## 8.8 추론 (Inference)

### 단일 이미지 추론

```java
public class MNISTInference {
    private MNISTNetwork network;

    public MNISTInference(String modelPath)
            throws IOException, ClassNotFoundException {
        network = ModelSerializer.load(modelPath);
    }

    public int predict(double[] image) {
        return network.predict(image);
    }

    public double[] predictProbs(double[] image) {
        return network.forward(image);
    }

    public void predictWithVisualization(double[] image) {
        double[] probs = predictProbs(image);
        int prediction = argmax(probs);

        // 이미지 출력
        System.out.println("\n입력 이미지:");
        MNISTVisualizer.printImage(image, -1);

        // 확률 분포
        System.out.println("\n예측 확률:");
        for (int i = 0; i < 10; i++) {
            String bar = "█".repeat((int)(probs[i] * 30));
            System.out.printf("%d: %5.2f%% %s%n", i, probs[i] * 100, bar);
        }

        System.out.printf("\n예측 결과: %d (확신도: %.2f%%)%n",
            prediction, probs[prediction] * 100);
    }

    private int argmax(double[] arr) {
        int maxIdx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[maxIdx]) maxIdx = i;
        }
        return maxIdx;
    }
}

// 사용
MNISTInference inference = new MNISTInference("mnist_model.bin");
inference.predictWithVisualization(testImages[0]);
```

### 배치 추론

```java
public int[] predictBatch(double[][] images) {
    int[] predictions = new int[images.length];
    for (int i = 0; i < images.length; i++) {
        predictions[i] = predict(images[i]);
    }
    return predictions;
}
```

---

## 8.9 성능 분석

### 혼동 행렬 (Confusion Matrix)

```java
public class ConfusionMatrix {

    public static int[][] compute(int[] predictions, int[] labels, int numClasses) {
        int[][] matrix = new int[numClasses][numClasses];

        for (int i = 0; i < predictions.length; i++) {
            matrix[labels[i]][predictions[i]]++;
        }

        return matrix;
    }

    public static void print(int[][] matrix) {
        int n = matrix.length;

        System.out.println("\n혼동 행렬 (행: 실제, 열: 예측):");
        System.out.print("    ");
        for (int i = 0; i < n; i++) System.out.printf("%5d", i);
        System.out.println();
        System.out.println("    " + "─".repeat(n * 5));

        for (int i = 0; i < n; i++) {
            System.out.printf("%2d │", i);
            for (int j = 0; j < n; j++) {
                System.out.printf("%5d", matrix[i][j]);
            }
            System.out.println();
        }
    }

    public static void printMetrics(int[][] matrix) {
        int n = matrix.length;

        System.out.println("\n클래스별 성능:");
        System.out.println("─".repeat(40));

        for (int i = 0; i < n; i++) {
            int tp = matrix[i][i];
            int fp = 0, fn = 0;

            for (int j = 0; j < n; j++) {
                if (j != i) {
                    fp += matrix[j][i];  // 다른 클래스를 i로 예측
                    fn += matrix[i][j];  // i를 다른 클래스로 예측
                }
            }

            double precision = tp / (double)(tp + fp);
            double recall = tp / (double)(tp + fn);
            double f1 = 2 * precision * recall / (precision + recall);

            System.out.printf("숫자 %d: Precision=%.3f, Recall=%.3f, F1=%.3f%n",
                i, precision, recall, f1);
        }
    }
}

// 사용
int[] predictions = inference.predictBatch(testImages);
int[][] cm = ConfusionMatrix.compute(predictions, testLabels, 10);
ConfusionMatrix.print(cm);
ConfusionMatrix.printMetrics(cm);
```

### 오분류 샘플 분석

```java
public class ErrorAnalysis {

    public static void showMisclassified(double[][] images, int[] labels,
                                         int[] predictions, int maxShow) {
        System.out.println("\n=== 오분류 샘플 분석 ===\n");

        int count = 0;
        for (int i = 0; i < predictions.length && count < maxShow; i++) {
            if (predictions[i] != labels[i]) {
                System.out.printf("샘플 %d: 실제=%d, 예측=%d%n",
                    i, labels[i], predictions[i]);
                MNISTVisualizer.printImage(images[i], labels[i]);
                System.out.println();
                count++;
            }
        }
    }
}

// 어떤 숫자가 자주 혼동되는지 분석
// 예: 4와 9, 3과 8, 5와 6 등
```

---

## 8.10 개선 아이디어

### 1. 드롭아웃 추가

```java
// 과적합 방지
public class ImprovedNetwork {
    // ...
    private DropoutLayer dropout1;
    private DropoutLayer dropout2;

    public ImprovedNetwork() {
        // ...
        dropout1 = new DropoutLayer(0.3);  // 30% 드롭
        dropout2 = new DropoutLayer(0.3);
    }

    public double[] forward(double[] image, boolean training) {
        double[] x = fc1.forward(image);
        x = relu1.forward(x);
        x = dropout1.forward(x, training);  // 드롭아웃
        // ...
    }
}
```

### 2. 학습률 스케줄링

```java
// 에폭마다 학습률 감소
double lr = initialLR * Math.pow(0.95, epoch);
```

### 3. 배치 정규화

```java
// 각 레이어 후에 정규화
public class BatchNormLayer {
    // 평균 0, 분산 1로 정규화
    // 학습 가능한 scale, shift 파라미터
}
```

### 4. 데이터 증강

```java
// 이미지 변형으로 데이터 다양성 증가
public class DataAugmentation {
    public double[] augment(double[] image) {
        // 랜덤 회전 (±10도)
        // 랜덤 이동 (±2픽셀)
        // 랜덤 스케일 (0.9~1.1)
    }
}
```

---

## 정리

### 구현한 것들

| 컴포넌트 | 역할 |
|---------|------|
| MNISTLoader | 데이터 로드 |
| DenseLayer | 완전 연결 레이어 |
| ReLULayer, SoftmaxLayer | 활성화 함수 |
| CrossEntropyLoss | 손실 함수 |
| AdamOptimizer | 옵티마이저 |
| MNISTNetwork | 전체 네트워크 |
| MNISTTrainer | 학습 루프 |

### 배운 것들의 통합

```
1장 벡터/행렬 → 이미지(784차원 벡터), 가중치(행렬)
2장 내적 → 뉴런의 가중합
3장 행렬곱 → 레이어 계산 (W × x + b)
4장 활성화 → ReLU, Softmax
5장 미분 → 역전파 그래디언트
6장 경사하강 → Adam 최적화
7장 손실함수 → Cross-Entropy
```

### 결과

- **97%+ 정확도** (10 에폭 학습)
- 순수 Java로 구현
- 외부 라이브러리 없이 전체 파이프라인 구축

---

## 다음 단계

이 책을 마친 후 다음을 학습해 보세요:

1. **합성곱 신경망 (CNN)**: 이미지에 더 적합
2. **딥러닝 프레임워크**: PyTorch, TensorFlow
3. **Vol.2 통계학**: 확률, 분포, 가설검정
4. **고급 주제**: Transformer, 강화학습

축하합니다! 🎉
**Java 개발자의 AI 수학 기초**를 완료했습니다!
