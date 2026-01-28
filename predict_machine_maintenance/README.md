# Machine Maintenance Prediction

This project uses machine learning to predict machine maintenance needs based on sensor data from IoT devices in a cloth manufacturing setting. The model analyzes various sensor readings (temperature, vibration, pressure, etc.) to detect potential machine faults early and predict future failures.

## Features

✅ **Implemented:**
- Multi-sensor data analysis (temperature, vibration, pressure, etc.)
- Random Forest Classification for robust fault detection
- SMOTE for handling imbalanced fault classes
- Feature standardization for consistent predictions
- Django REST API for real-time predictions
- Interactive dashboard with real-time monitoring
- IoT device integration with API key authentication
- Prediction history and analytics

## Quick Start (Windows PowerShell)

### 1. Setup Environment

```powershell
# Create virtual environment and activate
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the ML Model

```powershell
python train_model.py
```

This will generate:
- `ml_models/model.joblib` - Trained Random Forest model
- `ml_models/scaler.joblib` - Feature standardization parameters
- `ml_models/feature_order.json` - Feature names order

### 3. Setup Django

```powershell
# Run database migrations
python manage.py migrate

# Create a superuser (optional, for admin access)
python manage.py createsuperuser

# Generate an API key for IoT devices
python manage.py generate_apikey --name "main-iot-device"
```

### 4. Run the Development Server

```powershell
python manage.py runserver
```

The application will be available at:
- Dashboard: http://localhost:8000/dashboard/
- API: http://localhost:8000/api/predict/
- Admin: http://localhost:8000/admin/

## Project Structure

```
predict_machine_maintenance/
├── dataset/
│   └── Industrial_fault_detection.csv  # Training data
├── examples/
│   └── iot_client.py                   # IoT device example
├── machine_fault_detection/            # Django project settings
├── ml_models/                          # ML model files (generated)
├── prediction/                         # Main Django app
│   ├── management/commands/
│   │   └── generate_apikey.py         # API key generator
│   ├── ml_models/                     # Model files for app
│   ├── models.py                      # Database models
│   ├── views.py                       # API views
│   ├── urls.py                        # URL routing
│   ├── utils.py                       # ML model utilities
│   └── templates/                     # HTML templates
│       └── prediction/
│           ├── dashboard.html
│           └── predict.html
├── train_model.py                     # ML training script
├── manage.py                          # Django management script
└── requirements.txt                   # Dependencies
```

## API Usage

### Test Prediction API

```python
import requests

url = "http://localhost:8000/api/predict/"
headers = {"Authorization": "Bearer YOUR_API_KEY"}
data = {
    "device_id": "test-device-1",
    "sensor_data": {
        "Temperature": 65.5,
        "Vibration": 2.8,
        "Pressure": 75.0,
        "Flow_Rate": 85.2,
        "Current": 45.3,
        "Voltage": 410.5,
        # Add FFT components...
    }
}
response = requests.post(url, json=data, headers=headers)
print(response.json())
```

### Using the IoT Client Example

```powershell
python examples/iot_client.py
```

## API Endpoints

- `POST /api/predict/` - Submit sensor data for fault prediction
- `GET /api/machines/` - Get latest state of all machines
- `GET /api/records/` - Get prediction history and records

## Dashboard Features

- Real-time machine health monitoring
- Latest sensor readings for each device
- Prediction confidence tracking
- Fault distribution charts
- Historical data visualization

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
DJANGO_SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
```

### Model Paths

The application automatically searches for model files in:
1. `ml_models/` (project root)
2. `prediction/ml_models/` (app directory)

## Troubleshooting

### Model not loading
- Ensure `train_model.py` has been run successfully
- Check that model files exist in `ml_models/` directory
- Verify feature_order.json exists

### Static files not loading
- Run `python manage.py collectstatic`
- Ensure `STATIC_URL` is configured in settings.py

### API authentication failed
- Generate a new API key: `python manage.py generate_apikey`
- Update your IoT client with the new key
- Check that the API key is active in the database

## Development

### Running Tests
```powershell
python manage.py test
```

### Database Migrations
```powershell
python manage.py makemigrations
python manage.py migrate
```

## Future Enhancements

- [ ] Model performance monitoring and drift detection
- [ ] Automated model retraining pipeline
- [ ] Email/alert notifications for faults
- [ ] Advanced analytics and reporting
- [ ] Multi-user authentication
- [ ] Docker containerization
- [ ] Comprehensive unit tests

## License

This project is licensed under the MIT License.
