package com.ecgssm.nativeapp

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.ecgssm.nativeapp.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val apiClient = LatestApiClient()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.baseUrlInput.setText(DEFAULT_BASE_URL)
        binding.fetchButton.setOnClickListener {
            fetchLatestReading()
        }
    }

    private fun fetchLatestReading() {
        val baseUrl = binding.baseUrlInput.text?.toString().orEmpty()
        setLoadingState(true)
        binding.statusValue.text = getString(R.string.status_loading)

        lifecycleScope.launch {
            runCatching {
                withContext(Dispatchers.IO) {
                    apiClient.fetchLatest(baseUrl)
                }
            }.onSuccess { reading ->
                renderReading(reading)
            }.onFailure { error ->
                renderError(error.message ?: getString(R.string.error_unknown))
            }

            setLoadingState(false)
        }
    }

    private fun renderReading(reading: LatestReading) {
        binding.labelValue.text = reading.label
        binding.confidenceValue.text = getString(R.string.confidence_format, reading.confidence * 100.0)
        binding.statusValue.text = getString(R.string.status_success)
        binding.rawResponseValue.text = reading.rawJson
    }

    private fun renderError(message: String) {
        binding.labelValue.text = getString(R.string.placeholder_value)
        binding.confidenceValue.text = getString(R.string.placeholder_value)
        binding.statusValue.text = message
        binding.rawResponseValue.text = getString(R.string.placeholder_value)
    }

    private fun setLoadingState(isLoading: Boolean) {
        binding.fetchButton.isEnabled = !isLoading
        binding.progressIndicator.visibility = if (isLoading) android.view.View.VISIBLE else android.view.View.GONE
    }

    companion object {
        private const val DEFAULT_BASE_URL = "http://192.168.1.50:8000"
    }
}

