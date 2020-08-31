package com.shebogholo.library;

import android.content.Context;
import android.widget.Toast;

public class DetectContent {
    //    Detect text aggressiveness using machine learning model
    public static void detect(Context context, String text){
        Toast.makeText(context, text, Toast.LENGTH_SHORT).show();
    }
}
