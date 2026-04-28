package com.ecgssm.nativeapp

import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject

class LatestApiClient(
    private val client: OkHttpClient = OkHttpClient()
) {
    suspend fun fetchLatest(baseUrl: String): LatestReading {
        val normalizedBaseUrl = normalizeBaseUrl(baseUrl)
        val request = Request.Builder()
            .url("$normalizedBaseUrl/latest")
            .get()
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IllegalStateException("Server returned HTTP ${response.code}")
            }

            val body = response.body?.string().orEmpty()
            if (body.isBlank()) {
                throw IllegalStateException("The /latest endpoint returned an empty response.")
            }

            return parseLatestReading(body)
        }
    }

    private fun normalizeBaseUrl(baseUrl: String): String {
        val trimmed = baseUrl.trim().trimEnd('/')
        require(trimmed.isNotEmpty()) { "Enter the Raspberry Pi host, for example 192.168.1.50:8000." }
        return if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
            trimmed
        } else {
            "http://$trimmed"
        }
    }

    private fun parseLatestReading(rawJson: String): LatestReading {
        val root = JSONObject(rawJson)
        val payload = when {
            root.has("result") && root.opt("result") is JSONObject -> root.getJSONObject("result")
            root.has("latest") && root.opt("latest") is JSONObject -> root.getJSONObject("latest")
            else -> root
        }

        val label = firstNonBlankString(
            payload,
            "label",
            "diagnosis",
            "prediction",
            "predicted_label",
            "class_name"
        ) ?: throw IllegalStateException("Could not find a diagnosis label in the /latest response.")

        val confidence = firstNumber(
            payload,
            "confidence",
            "probability",
            "score",
            "max_probability"
        ) ?: confidenceFromProbabilities(payload, label)
        ?: throw IllegalStateException("Could not find a confidence value in the /latest response.")

        return LatestReading(
            label = label,
            confidence = normalizeConfidence(confidence),
            rawJson = rawJson
        )
    }

    private fun firstNonBlankString(json: JSONObject, vararg keys: String): String? {
        for (key in keys) {
            val value = json.optString(key, "").trim()
            if (value.isNotEmpty()) {
                return value
            }
        }
        return null
    }

    private fun firstNumber(json: JSONObject, vararg keys: String): Double? {
        for (key in keys) {
            if (!json.has(key)) {
                continue
            }
            val value = json.opt(key)
            when (value) {
                is Number -> return value.toDouble()
                is String -> value.toDoubleOrNull()?.let { return it }
            }
        }
        return null
    }

    private fun confidenceFromProbabilities(json: JSONObject, label: String): Double? {
        val probabilities = json.optJSONObject("probabilities") ?: return null
        return when {
            probabilities.has(label) -> probabilities.optDouble(label)
            probabilities.has(label.substringBefore(" ")) -> probabilities.optDouble(label.substringBefore(" "))
            else -> null
        }.takeIf { !it.isNaN() }
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

