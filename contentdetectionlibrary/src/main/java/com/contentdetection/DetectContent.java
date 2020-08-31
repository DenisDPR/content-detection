package com.contentdetection;

import android.content.Context;
import android.widget.Toast;

public class DetectContent {
    public static void s(Context context, String text){
        Toast.makeText(context, text, Toast.LENGTH_SHORT).show();
    }
}
