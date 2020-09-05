package com.shebogholo.library;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.Build;
import android.util.Log;
import android.widget.Toast;

import androidx.annotation.RequiresApi;
import androidx.annotation.WorkerThread;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;


public class DetectContent {
    //    Detect text aggressiveness using machine learning model
    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    public static String detect(Context context, String text){
        TextClassificationClient textClassificationClient = new TextClassificationClient(context);
        textClassificationClient.load();
        List<DetectContent.TextClassificationClient.Result> results = textClassificationClient.classify(text);
        DetectContent.TextClassificationClient.Result result = results.get(1);
        System.out.println(result.confidence);
        textClassificationClient.unload();
        if (result.confidence > 0.40){
            return "unsuccessful";
        }else{
            return text;
        }
    }

    static class TextClassificationClient {
        private static final String TAG = "TextClassification";
        private static final String MODEL_PATH = "text_classification.tflite";
        private static final String DIC_PATH = "vocab.txt";
        private static final String LABEL_PATH = "labels.txt";

        private static final int SENTENCE_LEN = 256;  // The maximum length of an input sentence.
        // Simple delimiter to split words.
        private static final String SIMPLE_SPACE_OR_PUNCTUATION = " |\\,|\\.|\\!|\\?|\n";
        private static final String START = "<START>";
        private static final String PAD = "<PAD>";
        private static final String UNKNOWN = "<UNKNOWN>";

        /** Number of results to show in the UI. */
        private static final int MAX_RESULTS = 1;

        private final Context context;
        private final Map<String, Integer> dic = new HashMap<>();
        private final List<String> labels = new ArrayList<>();
        private Interpreter tflite;

        public TextClassificationClient(Context context) {
            this.context = context;
        }

        /** An immutable result returned by a TextClassifier describing what was classified. */
        public static class Result {
            /**
             * A unique identifier for what has been classified. Specific to the class, not the instance of
             * the object.
             */
            private final String id;

            /** Display name for the result. */
            private final String title;

            /** A sortable score for how good the result is relative to others. Higher should be better. */
            private final Float confidence;

            public Result(final String id, final String title, final Float confidence) {
                this.id = id;
                this.title = title;
                this.confidence = confidence;
            }

            public String getId() {
                return id;
            }

            public String getTitle() {
                return title;
            }

            public Float getConfidence() {
                return confidence;
            }

            @Override
            public String toString() {
                String resultString = "";
                if (id != null) {
                    resultString += "[" + id + "] ";
                }

                if (title != null) {
                    resultString += title + " ";
                }

                if (confidence != null) {
                    resultString += String.format("(%.1f%%) ", confidence * 100.0f);
                }

                return resultString.trim();
            }
        }

        @RequiresApi(api = Build.VERSION_CODES.KITKAT)
        @WorkerThread
        public void load() {
            loadModel();
            loadDictionary();
            loadLabels();
        }

        /** Load TF Lite model. */
        @RequiresApi(api = Build.VERSION_CODES.KITKAT)
        @WorkerThread
        private synchronized void loadModel() {
            Log.v(TAG, "Loading model....");
            try {
                ByteBuffer buffer = loadModelFile(this.context.getAssets());
                tflite = new Interpreter(buffer);
                Log.v(TAG, "TFLite model loaded.");
            } catch (IOException ex) {
                Log.e(TAG, ex.getMessage());
            }
        }
        /** Load words dictionary. */
        @RequiresApi(api = Build.VERSION_CODES.KITKAT)
        @WorkerThread
        private synchronized void loadDictionary() {
            try {
                loadDictionaryFile(this.context.getAssets());
                Log.v(TAG, "Dictionary loaded.");
            } catch (IOException ex) {
                Log.e(TAG, ex.getMessage());
            }
        }
        /** Load labels. */
        @RequiresApi(api = Build.VERSION_CODES.KITKAT)
        @WorkerThread
        private synchronized void loadLabels() {
            try {
                loadLabelFile(this.context.getAssets());
                Log.v(TAG, "Labels loaded.");
            } catch (IOException ex) {
                Log.e(TAG, ex.getMessage());
            }
        }

        /** Load TF Lite model from assets. */
        @RequiresApi(api = Build.VERSION_CODES.KITKAT)
        private static MappedByteBuffer loadModelFile(AssetManager assetManager) throws IOException {
            try (AssetFileDescriptor fileDescriptor = assetManager.openFd(MODEL_PATH);
                 FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
                FileChannel fileChannel = inputStream.getChannel();
                long startOffset = fileDescriptor.getStartOffset();
                long declaredLength = fileDescriptor.getDeclaredLength();
                return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
            }
        }

        @RequiresApi(api = Build.VERSION_CODES.KITKAT)
        private void loadLabelFile(AssetManager assetManager) throws IOException {
            try (InputStream ins = assetManager.open(LABEL_PATH);
                 BufferedReader reader = new BufferedReader(new InputStreamReader(ins))) {
                // Each line in the label file is a label.
                while (reader.ready()) {
                    labels.add(reader.readLine());
                }
            }
        }

        /** Load labels from assets. */
        @RequiresApi(api = Build.VERSION_CODES.KITKAT)
        private void loadDictionaryFile(AssetManager assetManager) throws IOException {
            try (InputStream ins = assetManager.open(DIC_PATH);
                 BufferedReader reader = new BufferedReader(new InputStreamReader(ins))) {
                // Each line in the dictionary has two columns.
                // First column is a word, and the second is the index of this word.
                while (reader.ready()) {
                    List<String> line = Arrays.asList(reader.readLine().split(" "));
                    if (line.size() < 2) {
                        continue;
                    }
                    dic.put(line.get(0), Integer.parseInt(line.get(1)));
                }
            }
        }

        /** Pre-prosessing: tokenize and map the input words into a float array. */
        float[][] tokenizeInputText(String text) {
            float[] tmp = new float[SENTENCE_LEN];
            List<String> array = Arrays.asList(text.split(SIMPLE_SPACE_OR_PUNCTUATION));
            System.out.print(array);

            int index = 0;
            // Prepend <START> if it is in vocabulary file.
            if (dic.containsKey(START)) {
                tmp[index++] = dic.get(START);
            }

            for (String word : array) {
                if (index >= SENTENCE_LEN) {
                    break;
                }
                tmp[index++] = dic.containsKey(word) ? dic.get(word) : (int) dic.get(UNKNOWN);
            }
            // Padding and wrapping.
            Arrays.fill(tmp, index, SENTENCE_LEN - 1, (int) dic.get(PAD));
            float[][] ans = {tmp};
            return ans;
        }
        Map<String, Integer> getDic() {
            return this.dic;
        }
        Interpreter getTflite() {
            return this.tflite;
        }

        List<String> getLabels() {
            return this.labels;
        }

        @WorkerThread
        public synchronized void unload() {
            tflite.close();
            dic.clear();
            labels.clear();
        }
        @WorkerThread
        public synchronized List<Result> classify(String text) {
            // Pre-prosessing.
            float[][] input = tokenizeInputText(text);

            // Run inference.
            Log.v(TAG, "Classifying text with TF Lite...");
            float[][] output = new float[1][labels.size()];
            tflite.run(input, output);

            // Find the best classifications.
            PriorityQueue<DetectContent.TextClassificationClient.Result> pq = new PriorityQueue<>(MAX_RESULTS, new Comparator<DetectContent.TextClassificationClient.Result>() {
                @Override
                public int compare(DetectContent.TextClassificationClient.Result lhs, DetectContent.TextClassificationClient.Result rhs) {
                    return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                }
            });
            for (int i = 0; i < labels.size(); i++) {
                pq.add(new DetectContent.TextClassificationClient.Result("" + i, labels.get(i), output[0][i]));
            }
            final ArrayList<DetectContent.TextClassificationClient.Result> results = new ArrayList<>();
            while (!pq.isEmpty()) {
                results.add(pq.poll());
            }

            // Return the probability of each class.
            return results;
        }
    }
}
