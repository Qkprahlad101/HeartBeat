package com.example.opencv;

import androidx.appcompat.app.AppCompatActivity;

import android.content.pm.ActivityInfo;
import android.content.res.Configuration;
import android.graphics.Camera;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCamera2View;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "MainActivity";
    JavaCameraView javaCameraView;
    File cascFile;
    CascadeClassifier faceDetector;
    private Mat mrgba, mgrey;

    TextView mfpsView, rgbView, hbView;
    int mFPS;
    long startTime = 0;
    long currentTime = 1000;

    int n,m;
    double[] sin,cos,window;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        rgbView = findViewById(R.id.rgbview);
        hbView = findViewById(R.id.hbview);
        javaCameraView = findViewById(R.id.javacamview);
        /* //To check if OpenCV works or not
        if (!OpenCVLoader.initDebug())
            Log.e("OpenCv", "Unable to load OpenCV");
        else
            Log.d("OpenCv", "OpenCV loaded");

         */

        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, baseCallback);
        } else {
            baseCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        javaCameraView.setCvCameraViewListener(this);

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mfpsView = (TextView) findViewById(R.id.fpstextview);
            }
        });
    }


    @Override
    public void onCameraViewStarted(int width, int height) {

        mrgba = new Mat();
        mgrey = new Mat();

    }

    @Override
    public void onCameraViewStopped() {
        mrgba.release();
        mgrey.release();

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mrgba = inputFrame.rgba();
        mgrey = inputFrame.gray();

        Mat rgbaT = mrgba.t();
        Core.flip(mrgba.t(), rgbaT, 1);
        Imgproc.resize(rgbaT, rgbaT, mrgba.size());

        //detect face
        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(mrgba, faceDetections);

        float changeGreen = 0;
        //Face is being detected here
        for (Rect rect : faceDetections.toArray()) {
            //makes rectangle
            Imgproc.rectangle(mrgba, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height)
                    , new Scalar(255, 0, 0)); //imgproc is used for image processing
            //shows rgb colour
            double[] rgb = mrgba.get(0, 0);
            changeGreen = (float) Math.abs(changeGreen - rgb[1]);
            rgbView.setText("Red: " + rgb[0] + ",Green: " + rgb[1] + ",Blue: " + rgb[2] + ",ChangeGreen: " + changeGreen); // showing RGB Colours
            Fast((int) changeGreen);
        }

        //To check for FPS
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if (currentTime - startTime >= 1000) {
                    mfpsView.setText("FPS: " + String.valueOf(mFPS));
                    mFPS = 0;
                    startTime = System.currentTimeMillis();
                }
                currentTime = System.currentTimeMillis();
                mFPS += 1;

            }
        });

        return mrgba;
    }

    public void Fast(int n) {
        double[] cos;
        double[] sin;
            m = (int) (Math.log(n) / Math.log(2));

        // Make sure n is a power of 2
        /*
        if (n != (1 << m))
            throw new RuntimeException("FFT length must be power of 2");
            
         */

        // precompute tables
        cos = new double[n / 2];
        sin = new double[n / 2];

        for (int i = 0; i < n / 2; i++) {
            cos[i] = Math.cos(-2 * Math.PI * i / n);
            sin[i] = Math.sin(-2 * Math.PI * i / n);
            hbView.setText("HeartBeat: "+cos[i]);
        }


    }


    private BaseLoaderCallback baseCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt2);
                    File cascadeDir = getDir("cascade", MODE_PRIVATE);
                    cascFile = new File(cascadeDir, "haarcascade_frontalface_alt2.xml");
                    FileOutputStream fos = null;
                    try {
                        fos = new FileOutputStream(cascFile);
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    }

                    byte[] buffer = new byte[4096];
                    int bytesRead = 0;

                    while (true) {
                        try {
                            if (!((bytesRead = is.read(buffer)) != -1)) break;
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                        try {
                            fos.write(buffer, 0, bytesRead);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                    try {
                        is.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    try {
                        fos.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    faceDetector = new CascadeClassifier(cascFile.getAbsolutePath());
                    if (faceDetector.empty()) {
                        faceDetector = null;
                    } else {
                        cascadeDir.delete();
                    }
                    javaCameraView.enableView();
                }
                break;

                default:
                    super.onManagerConnected(status);
            }
        }
    };
}