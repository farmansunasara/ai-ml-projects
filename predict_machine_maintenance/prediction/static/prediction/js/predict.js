document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predict-form');
    const fftToggle = document.getElementById('fft-toggle');
    const fftSection = document.getElementById('fft-section');
    const resultSection = document.getElementById('result');

    // Initialize FFT input fields
    ['Temperature', 'Vibration', 'Pressure'].forEach(type => {
        const container = document.querySelector(`.fft-inputs[data-type="${type}"]`);
        for (let i = 0; i < 10; i++) {
            const input = document.createElement('input');
            input.type = 'number';
            input.className = 'form-control form-control-sm fft-input';
            input.dataset.fftType = type;
            input.dataset.fftIndex = i;
            input.placeholder = `Component ${i + 1}`;
            input.step = '0.01';
            container.appendChild(input);
        }
    });

    // Toggle FFT section visibility
    fftToggle.addEventListener('change', () => {
        fftSection.style.display = fftToggle.checked ? 'block' : 'none';
        const fftInputs = document.querySelectorAll('.fft-input');
        fftInputs.forEach(input => {
            input.required = fftToggle.checked;
        });
    });

    // Form validation and submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        if (!form.checkValidity()) {
            e.stopPropagation();
            form.classList.add('was-validated');
            return;
        }

        const deviceId = document.getElementById('device').value.trim();
        const metricInputs = document.querySelectorAll('.metric-input');
        const sensorData = {};

        // Collect basic metrics
        metricInputs.forEach(input => {
            sensorData[input.dataset.metric] = parseFloat(input.value);
        });

        // Collect FFT data if enabled
        if (fftToggle.checked) {
            ['Temperature', 'Vibration', 'Pressure'].forEach(type => {
                const fftInputs = document.querySelectorAll(`.fft-input[data-fft-type="${type}"]`);
                fftInputs.forEach((input, idx) => {
                    sensorData[`FFT_${type.slice(0,4)}_${idx}`] = parseFloat(input.value) || 0;
                });
            });
        }

        try {
            const response = await fetch('/api/predict/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    device_id: deviceId || null,
                    sensor_data: sensorData
                })
            });

            if (!response.ok) {
                throw new Error('Prediction request failed');
            }

            const result = await response.json();
            displayResult(result);
        } catch (error) {
            console.error('Prediction error:', error);
            alert('Failed to get prediction. Please try again.');
        }
    });

    // Display prediction results
    function displayResult(data) {
        resultSection.style.display = 'block';
        
        // Update status badge and label
        const statusBadge = resultSection.querySelector('.status-badge');
        const statusLabel = resultSection.querySelector('.status-label');
        const isProblem = data.prediction === 'fault';
        
        statusBadge.textContent = isProblem ? 'FAULT DETECTED' : 'NORMAL';
        statusBadge.className = `badge status-badge ${isProblem ? 'fault' : 'normal'}`;
        
        statusLabel.textContent = isProblem ? 'Machine Fault Detected' : 'Machine Operating Normally';
        statusLabel.className = `h4 mb-0 ${isProblem ? 'text-danger' : 'text-success'}`;

        // Update confidence bar
        const confidenceBar = resultSection.querySelector('.confidence-bar');
        const confidenceValue = resultSection.querySelector('.confidence-value');
        const confidence = (data.confidence * 100).toFixed(1);
        
        confidenceBar.style.width = `${confidence}%`;
        confidenceBar.className = `progress-bar confidence-bar ${isProblem ? 'fault' : 'normal'}`;
        confidenceValue.textContent = `${confidence}%`;

        // Update record ID
        resultSection.querySelector('.record-id').textContent = data.record_id;

        // Scroll to result
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Clear result when form is reset
    form.addEventListener('reset', () => {
        resultSection.style.display = 'none';
        form.classList.remove('was-validated');
    });
});