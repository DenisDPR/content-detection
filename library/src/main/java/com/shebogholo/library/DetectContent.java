package com.shebogholo.library;

import android.content.Context;
import android.os.Build;
import android.widget.Toast;

import androidx.annotation.RequiresApi;

import java.util.List;

public class DetectContent {
    private Context context;
    private static TextClassificationClient textClassificationClient;
    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    public DetectContent() {
        textClassificationClient = new TextClassificationClient(context);
        textClassificationClient.load();
    }

    //    Detect text aggressiveness using machine learning model
    public static void detect(Context context, String text){
        List<TextClassificationClient.Result> results = textClassificationClient.classify(text);
        TextClassificationClient.Result result = results.get(0);
        Toast.makeText(context, ""+result.getTitle(), Toast.LENGTH_LONG).show();
    }
}
