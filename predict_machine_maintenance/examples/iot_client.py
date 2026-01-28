import requests
import json
import time
import random
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000/api/predict/"
API_KEY = "_3XBaT7e5zXKUtnE6M5Tg-7wCQ-kJAqX"
DEVICE_ID = "test-device-1"

# Headers for authentication
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def generate_sensor_data():
    """
    Generate simulated sensor data for a cloth manufacturing machine.
    Returns realistic-looking data that could indicate potential faults.
    """
    # Base measurements
    temperature = random.uniform(35, 85)
    vibration = random.uniform(2.0, 3.5)
    pressure = random.uniform(50, 100)
    flow_rate = random.uniform(50, 100)
    current = random.uniform(35, 75)
    voltage = random.uniform(350, 450)
    
    # Generate FFT components (simplified simulation)
    def generate_fft_components():
        return [
            random.uniform(0.1, 1.0) for _ in range(10)  # FFT components 0-9
        ]
    
    fft_temp = generate_fft_components()
    fft_vib = generate_fft_components()
    fft_pres = generate_fft_components()
    
    return {
        "Temperature": temperature,
        "Vibration": vibration,
        "Pressure": pressure,
        "Flow_Rate": flow_rate,
        "Current": current,
        "Voltage": voltage,
        # FFT components
        **{f"FFT_Temp_{i}": v for i, v in enumerate(fft_temp)},
        **{f"FFT_Vib_{i}": v for i, v in enumerate(fft_vib)},
        **{f"FFT_Pres_{i}": v for i, v in enumerate(fft_pres)}
    }

def send_prediction_request(sensor_data):
    """
    Send sensor data to the prediction API and get fault prediction.
    """
    payload = {
        "device_id": DEVICE_ID,
        "timestamp": datetime.now().isoformat(),
        "sensor_data": sensor_data
    }
    
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction received: {json.dumps(result, indent=2)}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error sending data: {str(e)}")

def main():
    print(f"Starting IoT client simulation for device {DEVICE_ID}")
    print(f"Sending data to {API_URL}")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            # Generate and send sensor data
            sensor_data = generate_sensor_data()
            print("\nSending sensor data:", json.dumps(sensor_data, indent=2))
            send_prediction_request(sensor_data)
            
            # Wait for 5 seconds before next reading
            time.sleep(5)
    
    except KeyboardInterrupt:
        print("\nStopping IoT client simulation...")

if __name__ == "__main__":
    main()
