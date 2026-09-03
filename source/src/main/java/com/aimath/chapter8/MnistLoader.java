package com.aimath.chapter8;

import java.io.*;

/**
 * MNIST 데이터 로더
 * IDX 파일 포맷으로 저장된 MNIST 데이터를 로드
 *
 * 파일 다운로드: http://yann.lecun.com/exdb/mnist/
 * - train-images-idx3-ubyte.gz
 * - train-labels-idx1-ubyte.gz
 * - t10k-images-idx3-ubyte.gz
 * - t10k-labels-idx1-ubyte.gz
 */
public class MnistLoader {

    private static final String DEFAULT_PATH = "data/mnist/";

    /**
     * MNIST 이미지 파일 로드
     *
     * @param filename 파일명 (예: "train-images-idx3-ubyte")
     * @return 이미지 배열 [이미지 수][784]
     */
    public static double[][] loadImages(String filename) throws IOException {
        return loadImages(DEFAULT_PATH, filename);
    }

    public static double[][] loadImages(String path, String filename) throws IOException {
        try (DataInputStream dis = new DataInputStream(
                new BufferedInputStream(new FileInputStream(path + filename)))) {

            // 매직 넘버 확인 (2051 = 이미지 파일)
            int magic = dis.readInt();
            if (magic != 2051) {
                throw new IOException("Invalid MNIST image file: magic=" + magic);
            }

            int count = dis.readInt();
            int rows = dis.readInt();
            int cols = dis.readInt();

            System.out.printf("Loading %d images (%d×%d)...%n", count, rows, cols);

            double[][] images = new double[count][rows * cols];

            for (int i = 0; i < count; i++) {
                for (int j = 0; j < rows * cols; j++) {
                    // 0~255 → 0.0~1.0 정규화
                    images[i][j] = (dis.readUnsignedByte() & 0xFF) / 255.0;
                }

                // 진행률 표시
                if ((i + 1) % 10000 == 0) {
                    System.out.printf("  Loaded %d/%d images%n", i + 1, count);
                }
            }

            return images;
        }
    }

    /**
     * MNIST 레이블 파일 로드
     *
     * @param filename 파일명 (예: "train-labels-idx1-ubyte")
     * @return 레이블 배열 [레이블 수]
     */
    public static int[] loadLabels(String filename) throws IOException {
        return loadLabels(DEFAULT_PATH, filename);
    }

    public static int[] loadLabels(String path, String filename) throws IOException {
        try (DataInputStream dis = new DataInputStream(
                new BufferedInputStream(new FileInputStream(path + filename)))) {

            // 매직 넘버 확인 (2049 = 레이블 파일)
            int magic = dis.readInt();
            if (magic != 2049) {
                throw new IOException("Invalid MNIST label file: magic=" + magic);
            }

            int count = dis.readInt();
            System.out.printf("Loading %d labels...%n", count);

            int[] labels = new int[count];
            for (int i = 0; i < count; i++) {
                labels[i] = dis.readUnsignedByte();
            }

            return labels;
        }
    }

    /**
     * 이미지를 ASCII 아트로 출력
     */
    public static void printImage(double[] image, int label) {
        System.out.println("Label: " + label);
        System.out.println("-".repeat(30));

        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                double pixel = image[row * 28 + col];

                // 밝기에 따른 문자 선택
                char c;
                if (pixel < 0.1) c = ' ';
                else if (pixel < 0.3) c = '.';
                else if (pixel < 0.5) c = ':';
                else if (pixel < 0.7) c = '+';
                else if (pixel < 0.9) c = '#';
                else c = '@';

                System.out.print(c);
            }
            System.out.println();
        }
    }

    /**
     * 데이터셋 통계 출력
     */
    public static void printDatasetStats(double[][] images, int[] labels) {
        System.out.println("=== 데이터셋 통계 ===");
        System.out.println("이미지 수: " + images.length);
        System.out.println("입력 차원: " + images[0].length);

        // 클래스별 개수
        int[] counts = new int[10];
        for (int label : labels) {
            counts[label]++;
        }

        System.out.println("\n클래스별 샘플 수:");
        for (int i = 0; i < 10; i++) {
            String bar = repeat("=", counts[i] / 100);
            System.out.printf("  %d: %5d %s%n", i, counts[i], bar);
        }
    }

    /**
     * 미니배치 생성기
     */
    public static class BatchIterator {
        private final double[][] images;
        private final int[] labels;
        private final int batchSize;
        private int currentIndex;

        public BatchIterator(double[][] images, int[] labels, int batchSize) {
            this.images = images;
            this.labels = labels;
            this.batchSize = batchSize;
            this.currentIndex = 0;
        }

        public boolean hasNext() {
            return currentIndex < images.length;
        }

        public Batch next() {
            int end = Math.min(currentIndex + batchSize, images.length);
            int size = end - currentIndex;

            double[][] batchImages = new double[size][];
            int[] batchLabels = new int[size];

            for (int i = 0; i < size; i++) {
                batchImages[i] = images[currentIndex + i];
                batchLabels[i] = labels[currentIndex + i];
            }

            currentIndex = end;
            return new Batch(batchImages, batchLabels);
        }

        public void reset() {
            currentIndex = 0;
        }

        public static class Batch {
            public final double[][] images;
            public final int[] labels;

            public Batch(double[][] images, int[] labels) {
                this.images = images;
                this.labels = labels;
            }
        }
    }

    private static String repeat(String s, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) sb.append(s);
        return sb.toString();
    }

    /**
     * 테스트용 메인
     */
    public static void main(String[] args) {
        System.out.println("=== MNIST 데이터 로더 ===\n");

        try {
            // 훈련 데이터 로드
            double[][] trainImages = loadImages("train-images-idx3-ubyte");
            int[] trainLabels = loadLabels("train-labels-idx1-ubyte");

            // 테스트 데이터 로드
            double[][] testImages = loadImages("t10k-images-idx3-ubyte");
            int[] testLabels = loadLabels("t10k-labels-idx1-ubyte");

            System.out.println();
            printDatasetStats(trainImages, trainLabels);

            // 샘플 이미지 출력
            System.out.println("\n=== 샘플 이미지 ===");
            for (int i = 0; i < 3; i++) {
                printImage(trainImages[i], trainLabels[i]);
                System.out.println();
            }

        } catch (FileNotFoundException e) {
            System.out.println("MNIST 데이터 파일을 찾을 수 없습니다.");
            System.out.println("다음 위치에 파일을 다운로드하세요: " + DEFAULT_PATH);
            System.out.println("  - train-images-idx3-ubyte");
            System.out.println("  - train-labels-idx1-ubyte");
            System.out.println("  - t10k-images-idx3-ubyte");
            System.out.println("  - t10k-labels-idx1-ubyte");
            System.out.println("\n다운로드: http://yann.lecun.com/exdb/mnist/");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
