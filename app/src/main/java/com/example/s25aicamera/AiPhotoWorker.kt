package com.example.s25aicamera

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.provider.MediaStore
import androidx.work.Worker
import androidx.work.WorkerParameters
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer
import java.text.SimpleDateFormat
import java.util.Locale
import kotlin.math.max
import kotlin.math.pow

class AiPhotoWorker(
    appContext: Context,
    params: WorkerParameters
) : Worker(appContext, params) {

    override fun doWork(): Result {
        val path = inputData.getString("photo_path") ?: return Result.failure()
        val zoom = inputData.getFloat("zoom", 1f)
        val file = File(path)

        if (!file.exists()) return Result.failure()

        return try {
            val source = BitmapFactory.decodeFile(path) ?: return Result.failure()

            // Fast visual enhancement first:
            // brighter, richer colors, better contrast.
            val styled = enhanceColor(source, zoom)

            // Neural enhancement is mainly useful when digital zoom is high.
            val enhanced =
                if (zoom >= 5f) {
                    try {
                        runRealEsrgan(styled, zoom)
                    } catch (_: Throwable) {
                        styled
                    }
                } else {
                    styled
                }

            saveEnhanced(enhanced, zoom)

            source.recycle()
            if (styled !== source && styled !== enhanced) styled.recycle()

            file.delete()

            Result.success()
        } catch (t: Throwable) {
            // Keep the queued source on disk so it isn't silently lost.
            Result.retry()
        }
    }

    private fun enhanceColor(source: Bitmap, zoom: Float): Bitmap {
        val w = source.width
        val h = source.height

        val src = IntArray(w * h)
        val dst = IntArray(w * h)

        source.getPixels(src, 0, w, 0, 0, w, h)

        // Noticeable but still fairly natural Samsung-like punch.
        val exposure = if (zoom >= 10f) 1.07f else 1.10f
        val contrast = if (zoom >= 10f) 1.10f else 1.12f
        val saturation = if (zoom >= 10f) 1.16f else 1.20f

        val hsv = FloatArray(3)

        for (i in src.indices) {
            val p = src[i]

            var r = Color.red(p) / 255f
            var g = Color.green(p) / 255f
            var b = Color.blue(p) / 255f

            // Exposure
            r *= exposure
            g *= exposure
            b *= exposure

            // Contrast around middle gray
            r = ((r - 0.5f) * contrast + 0.5f).coerceIn(0f, 1f)
            g = ((g - 0.5f) * contrast + 0.5f).coerceIn(0f, 1f)
            b = ((b - 0.5f) * contrast + 0.5f).coerceIn(0f, 1f)

            val rr = (r * 255).toInt()
            val gg = (g * 255).toInt()
            val bb = (b * 255).toInt()

            Color.RGBToHSV(rr, gg, bb, hsv)

            // Vibrance-style saturation:
            // already saturated colors get less additional boost.
            val sat = hsv[1]
            val boost = 1f + (saturation - 1f) * (1f - sat * 0.45f)
            hsv[1] = (sat * boost).coerceIn(0f, 1f)

            // Slight lift to dark/mid tones.
            hsv[2] = hsv[2].pow(0.94f).coerceIn(0f, 1f)

            dst[i] = Color.HSVToColor(hsv)
        }

        var result = Bitmap.createBitmap(dst, w, h, Bitmap.Config.ARGB_8888)

        // Fast sharpening pass.
        result = sharpen(result, if (zoom >= 10f) 0.42f else 0.28f)

        return result
    }

    private fun sharpen(source: Bitmap, amount: Float): Bitmap {
        val w = source.width
        val h = source.height

        // Avoid an expensive full-resolution convolution.
        // This simple local kernel is fast enough for the first v0.4 test.
        val src = IntArray(w * h)
        val dst = IntArray(w * h)

        source.getPixels(src, 0, w, 0, 0, w, h)
        src.copyInto(dst)

        for (y in 1 until h - 1) {
            val row = y * w

            for (x in 1 until w - 1) {
                val i = row + x

                val center = src[i]
                val left = src[i - 1]
                val right = src[i + 1]
                val top = src[i - w]
                val bottom = src[i + w]

                fun channel(pixel: Int, shift: Int): Int =
                    (pixel shr shift) and 255

                fun sharpenChannel(shift: Int): Int {
                    val c = channel(center, shift).toFloat()
                    val blur = (
                        channel(left, shift) +
                        channel(right, shift) +
                        channel(top, shift) +
                        channel(bottom, shift)
                    ) / 4f

                    return (c + (c - blur) * amount)
                        .toInt()
                        .coerceIn(0, 255)
                }

                val r = sharpenChannel(16)
                val g = sharpenChannel(8)
                val b = sharpenChannel(0)

                dst[i] =
                    (255 shl 24) or
                    (r shl 16) or
                    (g shl 8) or
                    b
            }
        }

        return Bitmap.createBitmap(dst, w, h, Bitmap.Config.ARGB_8888)
    }

    private fun runRealEsrgan(source: Bitmap, zoom: Float): Bitmap {
        val env = OrtEnvironment.getEnvironment()

        val modelBytes = applicationContext.assets
            .open("realesrgan_x2plus.onnx")
            .use { it.readBytes() }

        val options = OrtSession.SessionOptions().apply {
            try {
                addNnapi()
            } catch (_: Throwable) {
            }

            setOptimizationLevel(
                OrtSession.SessionOptions.OptLevel.ALL_OPT
            )
        }

        env.createSession(modelBytes, options).use { session ->

            // v0.4 prioritizes speed.
            // The AI reconstructs a smaller representation,
            // then we return to a useful photo size.
            val maxInput = when {
                zoom >= 20f -> 640
                zoom >= 10f -> 704
                else -> 768
            }

            val longest = max(source.width, source.height)

            val inputBmp =
                if (longest > maxInput) {
                    val scale = maxInput.toFloat() / longest

                    Bitmap.createScaledBitmap(
                        source,
                        (source.width * scale).toInt().coerceAtLeast(2),
                        (source.height * scale).toInt().coerceAtLeast(2),
                        true
                    )
                } else {
                    source
                }

            val w = inputBmp.width - inputBmp.width % 2
            val h = inputBmp.height - inputBmp.height % 2

            val even =
                if (w != inputBmp.width || h != inputBmp.height) {
                    Bitmap.createBitmap(inputBmp, 0, 0, w, h)
                } else {
                    inputBmp
                }

            val pixels = IntArray(w * h)
            even.getPixels(pixels, 0, w, 0, 0, w, h)

            val plane = w * h
            val data = FloatArray(plane * 3)

            for (i in pixels.indices) {
                val p = pixels[i]

                data[i] =
                    Color.red(p) / 255f

                data[plane + i] =
                    Color.green(p) / 255f

                data[plane * 2 + i] =
                    Color.blue(p) / 255f
            }

            val tensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(data),
                longArrayOf(
                    1,
                    3,
                    h.toLong(),
                    w.toLong()
                )
            )

            tensor.use {
                session.run(
                    mapOf(session.inputNames.first() to tensor)
                ).use { result ->

                    val out = result[0] as OnnxTensor
                    val shape = out.info.shape

                    val oh = shape[2].toInt()
                    val ow = shape[3].toInt()

                    val values = FloatArray(3 * ow * oh)
                    out.floatBuffer.get(values)

                    val op = ow * oh
                    val output = IntArray(op)

                    for (i in output.indices) {
                        val r =
                            (values[i].coerceIn(0f, 1f) * 255)
                                .toInt()

                        val g =
                            (values[op + i].coerceIn(0f, 1f) * 255)
                                .toInt()

                        val b =
                            (values[op * 2 + i].coerceIn(0f, 1f) * 255)
                                .toInt()

                        output[i] =
                            (255 shl 24) or
                            (r shl 16) or
                            (g shl 8) or
                            b
                    }

                    val ai = Bitmap.createBitmap(
                        output,
                        ow,
                        oh,
                        Bitmap.Config.ARGB_8888
                    )

                    /*
                     * Keep enough resolution for viewing/zooming,
                     * but don't create giant fake 2x files.
                     */
                    val targetLongest =
                        max(source.width, source.height)
                            .coerceAtMost(4096)

                    val aiLongest = max(ai.width, ai.height)

                    return if (aiLongest != targetLongest) {
                        val scale =
                            targetLongest.toFloat() / aiLongest

                        Bitmap.createScaledBitmap(
                            ai,
                            (ai.width * scale).toInt(),
                            (ai.height * scale).toInt(),
                            true
                        )
                    } else {
                        ai
                    }
                }
            }
        }
    }

    private fun saveEnhanced(bitmap: Bitmap, zoom: Float) {
        val date = SimpleDateFormat(
            "yyyyMMdd_HHmmss_SSS",
            Locale.US
        ).format(System.currentTimeMillis())

        val name =
            "S25AI_${date}_${zoom.toInt()}x_AI.jpg"

        val values = ContentValues().apply {
            put(
                MediaStore.Images.Media.DISPLAY_NAME,
                name
            )

            put(
                MediaStore.Images.Media.MIME_TYPE,
                "image/jpeg"
            )

            put(
                MediaStore.Images.Media.RELATIVE_PATH,
                "Pictures/S25 AI Camera/Enhanced"
            )

            put(
                MediaStore.Images.Media.IS_PENDING,
                1
            )
        }

        val uri = applicationContext.contentResolver.insert(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            values
        ) ?: error("Cannot create Enhanced photo")

        applicationContext.contentResolver
            .openOutputStream(uri)
            ?.use {
                bitmap.compress(
                    Bitmap.CompressFormat.JPEG,
                    97,
                    it
                )
            }

        values.clear()
        values.put(
            MediaStore.Images.Media.IS_PENDING,
            0
        )

        applicationContext.contentResolver.update(
            uri,
            values,
            null,
            null
        )
    }
}
