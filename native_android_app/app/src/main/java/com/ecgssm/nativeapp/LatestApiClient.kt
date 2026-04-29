package com.ecgssm.nativeapp

import android.util.Log
import okhttp3.*
import org.json.JSONObject
import java.io.IOException
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine

class LatestApiClient {
    private val client = OkHttpClient()

    suspend fun fetchLatest(baseUrl: String): LatestReading = suspendCoroutine { continuation ->
        val url = if (baseUrl.endsWith("/")) "${baseUrl}latest" else "$baseUrl/latest"
        
        val request = Request.Builder()
            .url(url)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                continuation.resumeWith(Result.failure(e))
            }

            override fun onResponse(call: Call, response: Response) {
                response.use { 
                    if (!response.isSuccessful) {
                        continuation.resumeWith(Result.failure(IOException("Server returned ${response.code}")))
                        return
                    }

                    val body = response.body?.string()
                    if (body == null) {
                        continuation.resumeWith(Result.failure(IOException("Empty response from server")))
                        return
                    }

                    try {
                        val json = JSONObject(body)
                        val reading = parseJson(json, body)
                        if (reading != null) {
                            continuation.resume(reading)
                        } else {
                            continuation.resumeWith(Result.failure(IOException("Could not find diagnosis data in JSON")))
                        }
                    } catch (e: Exception) {
                        continuation.resumeWith(Result.failure(IOException("Failed to parse JSON: ${e.message}")))
                    }
                }
            }
        })
    }

    private fun parseJson(json: JSONObject, rawJson: String): LatestReading? {
        // Try to find label
        val label = findString(json, listOf("label", "diagnosis", "prediction", "class"))
        
        // Try to find confidence
        var confidence = findDouble(json, listOf("confidence", "probability", "prob", "score", "accuracy"))
        
        // If confidence is null, try to look into a "probabilities" or "results" object if we have a label
        if (confidence == null && label != null) {
            confidence = confidenceFromProbabilities(json, label)
        }

        if (label == null) return null

        return LatestReading(
            label = label,
            confidence = confidence ?: 0.0,
            rawJson = rawJson
        )
    }

    private fun findString(json: JSONObject, keys: List<String>): String? {
        for (key in keys) {
            if (json.has(key)) return json.optString(key)
        }
        // Try nested result
        if (json.has("result")) {
            val result = json.optJSONObject("result")
            if (result != null) return findString(result, keys)
        }
        return null
    }

    private fun findDouble(json: JSONObject, keys: List<String>): Double? {
        for (key in keys) {
            if (json.has(key)) {
                val value = json.opt(key)
                when (value) {
                    is Number -> return value.toDouble()
                    is String -> value.toDoubleOrNull()?.let { return it }
                }
            }
        }
        // Try nested result
        if (json.has("result")) {
            val result = json.optJSONObject("result")
            if (result != null) return findDouble(result, keys)
        }
        return null
    }

    private fun confidenceFromProbabilities(json: JSONObject, label: String): Double? {
        val probabilities = json.optJSONObject("probabilities") ?: return null
        return when {
            probabilities.has(label) -> probabilities.optDouble(label)
            probabilities.has(label.substringBefore(" ")) -> probabilities.optDouble(label.substringBefore(" "))
            else -> null
        }?.takeIf { !it.isNaN() }
    }

    private fun normalizeConfidence(value: Double): Double {
        return when {
            value.isNaN() -> 0.0
            value > 1.0 && value <= 100.0 -> value / 100.0
            value < 0.0 -> 0.0
            value > 1.0 -> 1.0
            else -> value
        }
    }
}
