package com.example.s25aicamera

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
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
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.max

class MainActivity : AppCompatActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var status: TextView
    private var imageCapture: ImageCapture? = null
    private var camera: Camera? = null
    private var zoomRatio = 1f
    private val cameraExecutor by lazy { ContextCompat.getMainExecutor(this) }
    private val aiExecutor = Executors.newSingleThreadExecutor()
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.preview)
        status = findViewById(R.id.status)
        findViewById<Button>(R.id.shutter).setOnClickListener { takePhoto() }
        aiExecutor.execute { initAi() }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
        else ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 10)
    }

    override fun onRequestPermissionsResult(r: Int, p: Array<out String>, g: IntArray) {
        super.onRequestPermissionsResult(r, p, g)
        if (r == 10 && g.firstOrNull() == PackageManager.PERMISSION_GRANTED) startCamera()
    }

    private fun initAi() {
        try {
            statusOnUi("⏳ AI model loading…")
            ortEnv = OrtEnvironment.getEnvironment()
            val modelBytes = assets.open("realesrgan_x2plus.onnx").use { it.readBytes() }
            val opts = OrtSession.SessionOptions().apply {
                try { addNnapi() } catch (_: Throwable) { }
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            ortSession = ortEnv!!.createSession(modelBytes, opts)
            statusOnUi("✨ AI READY")
        } catch (t: Throwable) {
            statusOnUi("AI ERROR: ${t.message}")
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
            imageCapture = ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY).build()
            provider.unbindAll()
            camera = provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
            setupZoomButtons(); setupPinchZoom()
        }, cameraExecutor)
    }

    private fun setupZoomButtons() {
        val bar = findViewById<LinearLayout>(R.id.zoomBar); bar.removeAllViews()
        listOf(1f, 3f, 5f, 10f, 20f, 30f).forEach { z ->
            Button(this).apply {
                text = "${z.toInt()}×"; textSize = 11f
                setOnClickListener { setZoom(z) }
                bar.addView(this, LinearLayout.LayoutParams(0, 56, 1f))
            }
        }
    }

    private fun setZoom(requested: Float) {
        val c = camera ?: return
        val zs = c.cameraInfo.zoomState.value ?: return
        zoomRatio = requested.coerceIn(zs.minZoomRatio, zs.maxZoomRatio)
        c.cameraControl.setZoomRatio(zoomRatio)
        status.text = "${String.format(Locale.US, "%.1f", zoomRatio)}× • AI READY"
    }

    private fun setupPinchZoom() {
        val detector = ScaleGestureDetector(this, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(d: ScaleGestureDetector): Boolean {
                val c = camera ?: return false
                val s = c.cameraInfo.zoomState.value ?: return false
                zoomRatio = (s.zoomRatio * d.scaleFactor).coerceIn(s.minZoomRatio, s.maxZoomRatio)
                c.cameraControl.setZoomRatio(zoomRatio)
                status.text = "${String.format(Locale.US, "%.1f", zoomRatio)}× • AI READY"
                return true
            }
        })
        previewView.setOnTouchListener { _, e: MotionEvent -> detector.onTouchEvent(e); true }
    }

    private fun takePhoto() {
        val capture = imageCapture ?: return
        if (ortSession == null) { status.text = "⏳ AI is still loading"; return }
        status.text = "📸 CAPTURING…"
        val temp = File(cacheDir, "capture_${System.currentTimeMillis()}.jpg")
        val options = ImageCapture.OutputFileOptions.Builder(temp).build()
        capture.takePicture(options, cameraExecutor, object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(o: ImageCapture.OutputFileResults) {
                status.text = "✨ AI PROCESSING ${zoomRatio.toInt()}×…"
                aiExecutor.execute {
                    try {
                        val src = BitmapFactory.decodeFile(temp.absolutePath) ?: error("Cannot decode photo")
                        val oriented = rotateIfNeeded(src, capture.targetRotation)
                        val enhanced = enhanceWithAi(oriented)
                        saveBitmap(enhanced)
                        temp.delete()
                        statusOnUi("✓ AI PHOTO SAVED • ${enhanced.width}×${enhanced.height}")
                    } catch (t: Throwable) {
                        statusOnUi("AI ERROR: ${t.message}")
                    }
                }
            }
            override fun onError(e: ImageCaptureException) { status.text = "CAMERA ERROR: ${e.message}" }
        })
    }

    private fun enhanceWithAi(source: Bitmap): Bitmap {
        val session = ortSession ?: error("AI model not loaded")
        // v0.2 deliberately caps the neural-network input. Real-ESRGAN x2 is heavy;
        // this keeps RAM usage sane on-device while still giving a true AI reconstruction.
        val maxInput = when {
            zoomRatio >= 20f -> 1024
            zoomRatio >= 10f -> 1152
            else -> 1280
        }
        val scaleDown = max(source.width, source.height).toFloat() / maxInput
        val inputBmp = if (scaleDown > 1f) Bitmap.createScaledBitmap(
            source, (source.width / scaleDown).toInt().coerceAtLeast(2),
            (source.height / scaleDown).toInt().coerceAtLeast(2), true
        ) else source
        val w = inputBmp.width - (inputBmp.width % 2)
        val h = inputBmp.height - (inputBmp.height % 2)
        val even = if (w != inputBmp.width || h != inputBmp.height) Bitmap.createBitmap(inputBmp, 0, 0, w, h) else inputBmp

        val pixels = IntArray(w * h); even.getPixels(pixels, 0, w, 0, 0, w, h)
        val data = FloatArray(3 * w * h)
        val plane = w * h
        for (i in pixels.indices) {
            val p = pixels[i]
            data[i] = ((p shr 16) and 255) / 255f
            data[plane + i] = ((p shr 8) and 255) / 255f
            data[2 * plane + i] = (p and 255) / 255f
        }
        val inputName = session.inputNames.first()
        OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(data), longArrayOf(1, 3, h.toLong(), w.toLong())).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { result ->
                val out = result[0] as OnnxTensor
                val shape = out.info.shape
                val oh = shape[2].toInt(); val ow = shape[3].toInt()
                val floats = out.floatBuffer
                val outData = FloatArray(3 * ow * oh); floats.get(outData)
                val outPixels = IntArray(ow * oh); val op = ow * oh
                for (i in outPixels.indices) {
                    val r = (outData[i].coerceIn(0f, 1f) * 255f).toInt()
                    val g = (outData[op + i].coerceIn(0f, 1f) * 255f).toInt()
                    val b = (outData[2 * op + i].coerceIn(0f, 1f) * 255f).toInt()
                    outPixels[i] = (255 shl 24) or (r shl 16) or (g shl 8) or b
                }
                return Bitmap.createBitmap(outPixels, ow, oh, Bitmap.Config.ARGB_8888)
            }
        }
    }

    private fun rotateIfNeeded(bitmap: Bitmap, rotation: Int): Bitmap = bitmap // EXIF is normally baked correctly by CameraX output

    private fun saveBitmap(bitmap: Bitmap) {
        val name = "S25AI_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(System.currentTimeMillis())}_${zoomRatio.toInt()}x.jpg"
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/S25 AI Camera")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: error("MediaStore insert failed")
        contentResolver.openOutputStream(uri)?.use { bitmap.compress(Bitmap.CompressFormat.JPEG, 96, it) }
        values.clear(); values.put(MediaStore.Images.Media.IS_PENDING, 0); contentResolver.update(uri, values, null, null)
    }

    private fun statusOnUi(s: String) = runOnUiThread { status.text = s }

    override fun onDestroy() {
        super.onDestroy(); aiExecutor.shutdown(); ortSession?.close(); ortEnv?.close()
    }
}
