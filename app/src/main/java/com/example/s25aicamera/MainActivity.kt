package com.example.s25aicamera

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.os.Bundle
import android.provider.MediaStore
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.work.Data
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkInfo
import androidx.work.WorkManager
import java.io.File
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.UUID
import java.util.concurrent.atomic.AtomicInteger

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var status: TextView

    private var imageCapture: ImageCapture? = null
    private var camera: Camera? = null
    private var zoomRatio = 1f

    private val cameraExecutor by lazy {
        ContextCompat.getMainExecutor(this)
    }

    private val pendingAi = AtomicInteger(0)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.preview)
        status = findViewById(R.id.status)

        status.text = "S25 AI CAMERA • READY"

        findViewById<Button>(R.id.shutter)
            .setOnClickListener {
                takePhotoFast()
            }

        if (
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                10
            )
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(
            requestCode,
            permissions,
            grantResults
        )

        if (
            requestCode == 10 &&
            grantResults.firstOrNull() ==
            PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        }
    }

    private fun startCamera() {
        val future =
            ProcessCameraProvider.getInstance(this)

        future.addListener({

            val provider = future.get()

            val preview =
                Preview.Builder()
                    .build()
                    .also {
                        it.setSurfaceProvider(
                            previewView.surfaceProvider
                        )
                    }

            /*
             * MINIMIZE_LATENCY is intentional.
             *
             * v0.3 used MAXIMIZE_QUALITY, which makes the
             * shutter feel slower.
             *
             * AI processing happens after capture now.
             */
            imageCapture =
                ImageCapture.Builder()
                    .setCaptureMode(
                        ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY
                    )
                    .setJpegQuality(95)
                    .build()

            provider.unbindAll()

            camera =
                provider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageCapture
                )

            setupZoomButtons()
            setupPinchZoom()

            status.text = "1× • READY"

        }, cameraExecutor)
    }

    private fun setupZoomButtons() {
        val bar =
            findViewById<LinearLayout>(R.id.zoomBar)

        bar.removeAllViews()

        listOf(
            1f,
            3f,
            5f,
            10f,
            20f,
            30f
        ).forEach { zoom ->

            Button(this).apply {

                text = "${zoom.toInt()}×"
                textSize = 11f

                setOnClickListener {
                    setZoom(zoom)
                }

                bar.addView(
                    this,
                    LinearLayout.LayoutParams(
                        0,
                        56,
                        1f
                    )
                )
            }
        }
    }

    private fun setZoom(requested: Float) {
        val currentCamera =
            camera ?: return

        val state =
            currentCamera.cameraInfo.zoomState.value
                ?: return

        zoomRatio =
            requested.coerceIn(
                state.minZoomRatio,
                state.maxZoomRatio
            )

        currentCamera.cameraControl
            .setZoomRatio(zoomRatio)

        updateStatus()
    }

    private fun setupPinchZoom() {
        val detector =
            ScaleGestureDetector(
                this,
                object :
                    ScaleGestureDetector
                    .SimpleOnScaleGestureListener() {

                    override fun onScale(
                        detector: ScaleGestureDetector
                    ): Boolean {

                        val currentCamera =
                            camera ?: return false

                        val state =
                            currentCamera
                                .cameraInfo
                                .zoomState
                                .value
                                ?: return false

                        zoomRatio =
                            (
                                state.zoomRatio *
                                detector.scaleFactor
                            ).coerceIn(
                                state.minZoomRatio,
                                state.maxZoomRatio
                            )

                        currentCamera
                            .cameraControl
                            .setZoomRatio(zoomRatio)

                        updateStatus()

                        return true
                    }
                }
            )

        previewView.setOnTouchListener {
                _,
                event: MotionEvent ->

            detector.onTouchEvent(event)
            true
        }
    }

    /*
     * FAST SHUTTER
     *
     * This method does NOT run AI.
     *
     * 1. Camera captures immediately.
     * 2. Original is saved.
     * 3. AI job is added to WorkManager.
     * 4. Camera is immediately usable again.
     */
    private fun takePhotoFast() {
        val capture =
            imageCapture ?: return

        val capturedZoom = zoomRatio

        val id =
            UUID.randomUUID().toString()

        val queueDir =
            File(filesDir, "ai_queue")

        if (!queueDir.exists()) {
            queueDir.mkdirs()
        }

        val tempFile =
            File(
                queueDir,
                "AI_$id.jpg"
            )

        val options =
            ImageCapture.OutputFileOptions
                .Builder(tempFile)
                .build()

        status.text = "📸"

        capture.takePicture(
            options,
            cameraExecutor,
            object :
                ImageCapture
                .OnImageSavedCallback {

                override fun onImageSaved(
                    output:
                    ImageCapture.OutputFileResults
                ) {
                    /*
                     * Camera is done here.
                     * From this point the UI is free.
                     */

                    try {
                        saveOriginal(
                            tempFile,
                            capturedZoom
                        )
                    } catch (_: Throwable) {
                        /*
                         * Even if Original saving fails,
                         * AI queue can still continue.
                         */
                    }

                    enqueueAi(
                        tempFile,
                        capturedZoom
                    )

                    updateStatus()
                }

                override fun onError(
                    exception:
                    ImageCaptureException
                ) {
                    status.text =
                        "CAMERA ERROR: " +
                        exception.message
                }
            }
        )
    }

    private fun enqueueAi(
        file: File,
        zoom: Float
    ) {
        val data =
            Data.Builder()
                .putString(
                    "photo_path",
                    file.absolutePath
                )
                .putFloat(
                    "zoom",
                    zoom
                )
                .build()

        val request =
            OneTimeWorkRequestBuilder
                <AiPhotoWorker>()
                .setInputData(data)
                .addTag("S25_AI_PHOTO")
                .build()

        pendingAi.incrementAndGet()

        updateStatus()

        val manager =
            WorkManager.getInstance(this)

        manager.enqueue(request)

        /*
         * Observe only this particular job.
         *
         * This is only for the small counter in the
         * camera UI. The Worker itself does NOT depend
         * on MainActivity remaining open.
         */
        manager
            .getWorkInfoByIdLiveData(request.id)
            .observe(this) { info ->

                if (
                    info?.state ==
                    WorkInfo.State.SUCCEEDED ||
                    info?.state ==
                    WorkInfo.State.FAILED ||
                    info?.state ==
                    WorkInfo.State.CANCELLED
                ) {
                    pendingAi
                        .updateAndGet {
                            value ->
                            (value - 1)
                                .coerceAtLeast(0)
                        }

                    updateStatus()
                }
            }
    }

    private fun saveOriginal(
        source: File,
        zoom: Float
    ) {
        val date =
            SimpleDateFormat(
                "yyyyMMdd_HHmmss_SSS",
                Locale.US
            ).format(
                System.currentTimeMillis()
            )

        val name =
            "S25AI_${date}_" +
            "${zoom.toInt()}x_ORIGINAL.jpg"

        val values =
            ContentValues().apply {

                put(
                    MediaStore.Images.Media
                        .DISPLAY_NAME,
                    name
                )

                put(
                    MediaStore.Images.Media
                        .MIME_TYPE,
                    "image/jpeg"
                )

                put(
                    MediaStore.Images.Media
                        .RELATIVE_PATH,
                    "Pictures/" +
                    "S25 AI Camera/" +
                    "Original"
                )

                put(
                    MediaStore.Images.Media
                        .IS_PENDING,
                    1
                )
            }

        val uri =
            contentResolver.insert(
                MediaStore.Images.Media
                    .EXTERNAL_CONTENT_URI,
                values
            ) ?: return

        contentResolver
            .openOutputStream(uri)
            ?.use { output ->

                source.inputStream()
                    .use { input ->

                        input.copyTo(output)
                    }
            }

        values.clear()

        values.put(
            MediaStore.Images.Media.IS_PENDING,
            0
        )

        contentResolver.update(
            uri,
            values,
            null,
            null
        )
    }

    private fun updateStatus() {
        runOnUiThread {

            val count =
                pendingAi.get()

            val zoomText =
                String.format(
                    Locale.US,
                    "%.1f",
                    zoomRatio
                )

            status.text =
                if (count > 0) {
                    "$zoomText×   ✨ AI $count"
                } else {
                    "$zoomText× • READY"
                }
        }
    }
}
