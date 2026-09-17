package com.example.s25aicamera

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
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
import kotlin.math.min
import kotlin.math.pow

class AiPhotoWorker(
    appContext: Context,
    params: WorkerParameters
) : Worker(appContext, params) {

    override fun doWork(): Result {

        val path =
            inputData.getString("photo_path")
                ?: return Result.failure()

        val zoom =
            inputData.getFloat("zoom", 1f)

        val file = File(path)

        if (!file.exists()) {
            return Result.failure()
        }

        return try {

            val source =
                BitmapFactory.decodeFile(path)
                    ?: return Result.failure()

            /*
             * 1-5x:
             * fast color/detail enhancement.
             *
             * 10-30x:
             * true tiled x4 neural reconstruction.
             */
            val result =
                if (zoom >= 10f) {

                    val ai =
                        runTiledX4(
                            source,
                            zoom
                        )

                    enhanceColorAndSharpness(
                        ai,
                        aiZoom = true
                    )

                } else {

                    enhanceColorAndSharpness(
                        source,
                        aiZoom = false
                    )
                }

            saveEnhanced(
                result,
                zoom
            )

            file.delete()

            Result.success()

        } catch (t: Throwable) {

            /*
             * Don't destroy the queued photo
             * if Android interrupted processing.
             */
            Result.retry()
        }
    }

    /*
     * ==============================
     * AI ZOOM x4
     * ==============================
     *
     * v0.4:
     * whole image -> ~640px -> AI.
     *
     * v0.5:
     * preserve source -> tiles ->
     * AI x4 each tile -> stitch -> 4K.
     */
    private fun runTiledX4(
        source: Bitmap,
        zoom: Float
    ): Bitmap {

        val env =
            OrtEnvironment.getEnvironment()

        val modelBytes =
            applicationContext
                .assets
                .open(
                    "realesrgan_x4plus.onnx"
                )
                .use {
                    it.readBytes()
                }

        val options =
            OrtSession
                .SessionOptions()
                .apply {

                    /*
                     * Try Samsung/Android
                     * hardware acceleration.
                     */
                    try {
                        addNnapi()
                    } catch (_: Throwable) {
                    }

                    setOptimizationLevel(
                        OrtSession
                            .SessionOptions
                            .OptLevel
                            .ALL_OPT
                    )
                }

        env.createSession(
            modelBytes,
            options
        ).use { session ->

            /*
             * IMPORTANT:
             *
             * We no longer shrink everything
             * to 640px.
             *
             * We preserve much more information.
             */
            val maxSourceLong =
                when {
                    zoom >= 20f -> 2304
                    zoom >= 10f -> 2560
                    else -> 2560
                }

            val longest =
                max(
                    source.width,
                    source.height
                )

            val working =
                if (longest > maxSourceLong) {

                    val scale =
                        maxSourceLong.toFloat() /
                        longest.toFloat()

                    Bitmap.createScaledBitmap(
                        source,
                        (
                            source.width *
                            scale
                        ).toInt()
                            .coerceAtLeast(2),

                        (
                            source.height *
                            scale
                        ).toInt()
                            .coerceAtLeast(2),

                        true
                    )

                } else {
                    source
                }

            /*
             * Tile size.
             *
             * Small enough to avoid massive
             * RAM spikes from x4 SR.
             */
            val tileSize = 256

            /*
             * Overlap prevents visible
             * seams between AI tiles.
             */
            val overlap = 16

            val innerSize =
                tileSize -
                overlap * 2

            val aiScale = 4

            /*
             * Temporary full x4 canvas.
             *
             * We eventually scale this to 4K.
             */
            val fullX4 =
                Bitmap.createBitmap(
                    working.width * aiScale,
                    working.height * aiScale,
                    Bitmap.Config.ARGB_8888
                )

            val canvas =
                Canvas(fullX4)

            val paint =
                Paint(
                    Paint.FILTER_BITMAP_FLAG
                )

            var y = 0

            while (y < working.height) {

                var x = 0

                while (x < working.width) {

                    val left =
                        max(
                            0,
                            x - overlap
                        )

                    val top =
                        max(
                            0,
                            y - overlap
                        )

                    val right =
                        min(
                            working.width,
                            x +
                            innerSize +
                            overlap
                        )

                    val bottom =
                        min(
                            working.height,
                            y +
                            innerSize +
                            overlap
                        )

                    var crop =
                        Bitmap.createBitmap(
                            working,
                            left,
                            top,
                            right - left,
                            bottom - top
                        )

                    /*
                     * Make dimensions divisible
                     * by 4 for the neural model.
                     */
                    val paddedW =
                        (
                            (
                                crop.width + 3
                            ) / 4
                        ) * 4

                    val paddedH =
                        (
                            (
                                crop.height + 3
                            ) / 4
                        ) * 4

                    if (
                        paddedW != crop.width ||
                        paddedH != crop.height
                    ) {

                        val padded =
                            Bitmap.createBitmap(
                                paddedW,
                                paddedH,
                                Bitmap.Config.ARGB_8888
                            )

                        Canvas(padded)
                            .drawBitmap(
                                crop,
                                0f,
                                0f,
                                paint
                            )

                        crop = padded
                    }

                    /*
                     * REAL AI x4 inference.
                     */
                    val aiTile =
                        inferX4(
                            session,
                            env,
                            crop
                        )

                    /*
                     * Remove overlap from the
                     * interior of the AI tile.
                     */
                    val innerLeft =
                        if (left == 0) {
                            0
                        } else {
                            overlap
                        }

                    val innerTop =
                        if (top == 0) {
                            0
                        } else {
                            overlap
                        }

                    val innerRight =
                        if (
                            right ==
                            working.width
                        ) {
                            right - left
                        } else {
                            crop.width -
                            overlap
                        }

                    val innerBottom =
                        if (
                            bottom ==
                            working.height
                        ) {
                            bottom - top
                        } else {
                            crop.height -
                            overlap
                        }

                    val srcRect =
                        Rect(
                            innerLeft *
                            aiScale,

                            innerTop *
                            aiScale,

                            innerRight *
                            aiScale,

                            innerBottom *
                            aiScale
                        )

                    val dstLeft =
                        (
                            left +
                            innerLeft
                        ) * aiScale

                    val dstTop =
                        (
                            top +
                            innerTop
                        ) * aiScale

                    val dstRect =
                        Rect(
                            dstLeft,
                            dstTop,

                            dstLeft +
                            srcRect.width(),

                            dstTop +
                            srcRect.height()
                        )

                    canvas.drawBitmap(
                        aiTile,
                        srcRect,
                        dstRect,
                        paint
                    )

                    x += innerSize
                }

                y += innerSize
            }

            /*
             * ==============================
             * FINAL 4K OUTPUT
             * ==============================
             *
             * Portrait:
             * up to 2160 x 3840
             *
             * Landscape:
             * up to 3840 x 2160
             *
             * Aspect ratio is preserved.
             */
            val portrait =
                fullX4.height >=
                fullX4.width

            val maxWidth =
                if (portrait) {
                    2160
                } else {
                    3840
                }

            val maxHeight =
                if (portrait) {
                    3840
                } else {
                    2160
                }

            val fit =
                min(
                    maxWidth.toFloat() /
                    fullX4.width,

                    maxHeight.toFloat() /
                    fullX4.height
                )

            /*
             * Normally x4 output is larger
             * than 4K, so downsampling improves
             * perceived sharpness and reduces
             * AI artifacts.
             */
            return if (fit < 1f) {

                Bitmap.createScaledBitmap(
                    fullX4,

                    (
                        fullX4.width *
                        fit
                    ).toInt(),

                    (
                        fullX4.height *
                        fit
                    ).toInt(),

                    true
                )

            } else {

                fullX4
            }
        }
    }

    /*
     * Run one tile through
     * RealESRGAN x4plus.
     */
    private fun inferX4(
        session: OrtSession,
        env: OrtEnvironment,
        bitmap: Bitmap
    ): Bitmap {

        val width =
            bitmap.width

        val height =
            bitmap.height

        val pixels =
            IntArray(
                width * height
            )

        bitmap.getPixels(
            pixels,
            0,
            width,
            0,
            0,
            width,
            height
        )

        val plane =
            width * height

        val data =
            FloatArray(
                plane * 3
            )

        for (
            i in pixels.indices
        ) {

            val pixel =
                pixels[i]

            data[i] =
                Color.red(pixel) /
                255f

            data[
                plane + i
            ] =
                Color.green(pixel) /
                255f

            data[
                plane * 2 + i
            ] =
                Color.blue(pixel) /
                255f
        }

        OnnxTensor
            .createTensor(
                env,
                FloatBuffer.wrap(
                    data
                ),
                longArrayOf(
                    1,
                    3,
                    height.toLong(),
                    width.toLong()
                )
            )
            .use { tensor ->

                session
                    .run(
                        mapOf(
                            session
                                .inputNames
                                .first()
                                to tensor
                        )
                    )
                    .use { result ->

                        val outputTensor =
                            result[0]
                                as OnnxTensor

                        val shape =
                            outputTensor
                                .info
                                .shape

                        val outHeight =
                            shape[2]
                                .toInt()

                        val outWidth =
                            shape[3]
                                .toInt()

                        val values =
                            FloatArray(
                                3 *
                                outWidth *
                                outHeight
                            )

                        outputTensor
                            .floatBuffer
                            .get(values)

                        val planeOut =
                            outWidth *
                            outHeight

                        val output =
                            IntArray(
                                planeOut
                            )

                        for (
                            i in output.indices
                        ) {

                            val r =
                                (
                                    values[i]
                                        .coerceIn(
                                            0f,
                                            1f
                                        ) *
                                    255f
                                ).toInt()

                            val g =
                                (
                                    values[
                                        planeOut +
                                        i
                                    ]
                                        .coerceIn(
                                            0f,
                                            1f
                                        ) *
                                    255f
                                ).toInt()

                            val b =
                                (
                                    values[
                                        planeOut *
                                        2 +
                                        i
                                    ]
                                        .coerceIn(
                                            0f,
                                            1f
                                        ) *
                                    255f
                                ).toInt()

                            output[i] =
                                (
                                    255 shl 24
                                ) or
                                (
                                    r shl 16
                                ) or
                                (
                                    g shl 8
                                ) or
                                b
                        }

                        return Bitmap
                            .createBitmap(
                                output,
                                outWidth,
                                outHeight,
                                Bitmap.Config
                                    .ARGB_8888
                            )
                    }
            }
    }

    /*
     * ==============================
     * PHOTO LOOK
     * ==============================
     *
     * Brighter
     * More contrast
     * More vibrant
     * Sharper
     */
    private fun enhanceColorAndSharpness(
        source: Bitmap,
        aiZoom: Boolean
    ): Bitmap {

        val width =
            source.width

        val height =
            source.height

        val src =
            IntArray(
                width * height
            )

        val dst =
            IntArray(
                width * height
            )

        source.getPixels(
            src,
            0,
            width,
            0,
            0,
            width,
            height
        )

        /*
         * AI Zoom gets stronger finishing.
         */
        val exposure =
            if (aiZoom) {
                1.10f
            } else {
                1.08f
            }

        val contrast =
            if (aiZoom) {
                1.16f
            } else {
                1.11f
            }

        val saturation =
            if (aiZoom) {
                1.25f
            } else {
                1.18f
            }

        val hsv =
            FloatArray(3)

        for (
            i in src.indices
        ) {

            val pixel =
                src[i]

            var r =
                Color.red(pixel) /
                255f

            var g =
                Color.green(pixel) /
                255f

            var b =
                Color.blue(pixel) /
                255f

            /*
             * Exposure
             */
            r *= exposure
            g *= exposure
            b *= exposure

            /*
             * Contrast
             */
            r =
                (
                    (
                        r - 0.5f
                    ) *
                    contrast +
                    0.5f
                ).coerceIn(
                    0f,
                    1f
                )

            g =
                (
                    (
                        g - 0.5f
                    ) *
                    contrast +
                    0.5f
                ).coerceIn(
                    0f,
                    1f
                )

            b =
                (
                    (
                        b - 0.5f
                    ) *
                    contrast +
                    0.5f
                ).coerceIn(
                    0f,
                    1f
                )

            Color.RGBToHSV(
                (
                    r * 255
                ).toInt(),

                (
                    g * 255
                ).toInt(),

                (
                    b * 255
                ).toInt(),

                hsv
            )

            /*
             * Vibrance rather than blindly
             * oversaturating everything.
             */
            val currentSat =
                hsv[1]

            val boost =
                1f +
                (
                    saturation -
                    1f
                ) *
                (
                    1f -
                    currentSat *
                    0.35f
                )

            hsv[1] =
                (
                    currentSat *
                    boost
                ).coerceIn(
                    0f,
                    1f
                )

            /*
             * Lift shadows / mids.
             */
            hsv[2] =
                hsv[2]
                    .pow(
                        0.93f
                    )
                    .coerceIn(
                        0f,
                        1f
                    )

            dst[i] =
                Color.HSVToColor(
                    hsv
                )
        }

        val colored =
            Bitmap.createBitmap(
                dst,
                width,
                height,
                Bitmap.Config.ARGB_8888
            )

        return sharpen(
            colored,
            if (aiZoom) {
                0.55f
            } else {
                0.25f
            }
        )
    }

    /*
     * Edge detail enhancement.
     */
    private fun sharpen(
        source: Bitmap,
        amount: Float
    ): Bitmap {

        val width =
            source.width

        val height =
            source.height

        val src =
            IntArray(
                width * height
            )

        val dst =
            IntArray(
                width * height
            )

        source.getPixels(
            src,
            0,
            width,
            0,
            0,
            width,
            height
        )

        src.copyInto(dst)

        fun channel(
            pixel: Int,
            shift: Int
        ): Int =
            (
                pixel shr shift
            ) and 255

        for (
            y in 1
            until height - 1
        ) {

            for (
                x in 1
                until width - 1
            ) {

                val i =
                    y *
                    width +
                    x

                fun sharpenChannel(
                    shift: Int
                ): Int {

                    val center =
                        channel(
                            src[i],
                            shift
                        ).toFloat()

                    val blur =
                        (
                            channel(
                                src[i - 1],
                                shift
                            ) +

                            channel(
                                src[i + 1],
                                shift
                            ) +

                            channel(
                                src[
                                    i -
                                    width
                                ],
                                shift
                            ) +

                            channel(
                                src[
                                    i +
                                    width
                                ],
                                shift
                            )
                        ) / 4f

                    return (
                        center +
                        (
                            center -
                            blur
                        ) *
                        amount
                    )
                        .toInt()
                        .coerceIn(
                            0,
                            255
                        )
                }

                val r =
                    sharpenChannel(
                        16
                    )

                val g =
                    sharpenChannel(
                        8
                    )

                val b =
                    sharpenChannel(
                        0
                    )

                dst[i] =
                    (
                        255 shl 24
                    ) or
                    (
                        r shl 16
                    ) or
                    (
                        g shl 8
                    ) or
                    b
            }
        }

        return Bitmap
            .createBitmap(
                dst,
                width,
                height,
                Bitmap.Config
                    .ARGB_8888
            )
    }

    /*
     * ==============================
     * SAVE FINAL AI 4K PHOTO
     * ==============================
     */
    private fun saveEnhanced(
        bitmap: Bitmap,
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
            "S25AI_${date}_${zoom.toInt()}x_AI4K.jpg"

        val values =
            ContentValues().apply {

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

        val uri =
            applicationContext
                .contentResolver
                .insert(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                    values
                )
                ?: error(
                    "Cannot create AI 4K photo"
                )

        applicationContext
            .contentResolver
            .openOutputStream(uri)
            ?.use { output ->

                bitmap.compress(
                    Bitmap.CompressFormat.JPEG,
                    98,
                    output
                )
            }

        values.clear()

        values.put(
            MediaStore.Images.Media.IS_PENDING,
            0
        )

        applicationContext
            .contentResolver
            .update(
                uri,
                values,
                null,
                null
            )
    }
}
