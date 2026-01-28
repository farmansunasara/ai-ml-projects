from django.shortcuts import render
from django.utils import timezone
from django.db.models import Count
from django.http import JsonResponse, HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
import numpy as np
import pandas as pd
import json
from io import BytesIO

from .models import PredictionRecord, APIKey
from .utils import load_model

class PredictAPIView(APIView):
    def post(self, request):
        # Load model, scaler, and feature order
        model, scaler, feature_order = load_model()
        
        if model is None:
            return Response({
                'error': 'ML model not loaded. Please ensure model file exists.'
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        
        if feature_order is None:
            return Response({
                'error': 'Feature order not found. Please ensure feature_order.json exists.'
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        # Determine if request is from web form or IoT device
        # Web requests can be either JSON or form-encoded
        # IoT requests are JSON but include Authorization header
        has_auth = request.headers.get('Authorization', '').startswith('Bearer ')
        is_web_request = not has_auth
        
        if has_auth:
            # Validate API key for IoT requests
            token = request.headers.get('Authorization', '')
            if not token.startswith('Bearer '):
                return Response({
                    'error': 'Missing API key. Use Bearer authentication.'
                }, status=status.HTTP_401_UNAUTHORIZED)
            
            api_key = token.split(' ')[1]
            try:
                key_obj = APIKey.objects.get(key=api_key, active=True)
            except APIKey.DoesNotExist:
                return Response({
                    'error': 'Invalid or inactive API key.'
                }, status=status.HTTP_401_UNAUTHORIZED)

        try:
            data = request.data
            sensor_data = data.get('sensor_data', {})
            device_id = data.get('device_id')
            
            # Extract features in the correct order from feature_order.json
            features = [sensor_data.get(feature, 0) for feature in feature_order]

            # Scale features if scaler is available
            features_array = np.array(features, dtype=float).reshape(1, -1)
            if scaler is not None:
                features_array = scaler.transform(features_array)

            # Make prediction
            prediction = model.predict(features_array)[0]
            prediction_proba = model.predict_proba(features_array)[0]
            
            # Get confidence (probability of the predicted class)
            confidence = float(prediction_proba[int(prediction)])
            
            # Create prediction record
            record = PredictionRecord.objects.create(
                device_id=device_id,
                source='iot' if not is_web_request else 'manual',
                input_json=sensor_data,
                predicted_label='fault' if prediction == 1 else 'normal',
                confidence=confidence,
                meta={
                    'probabilities': {
                        'normal': float(prediction_proba[0]),
                        'fault': float(prediction_proba[1])
                    }
                }
            )
            
            return Response({
                'prediction': 'fault' if prediction == 1 else 'normal',
                'confidence': confidence,
                'record_id': record.id,
                'probabilities': {
                    'normal': float(prediction_proba[0]),
                    'fault': float(prediction_proba[1])
                }
            })
            
        except Exception as e:
            return Response({
                'error': f'Prediction error: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def dashboard_view(request):
    """Dashboard view showing prediction history and stats."""
    # Get recent predictions
    recent_predictions = PredictionRecord.objects.order_by('-timestamp')[:20]
    
    # Get timeline data (last 24 hours)
    last_24h = timezone.now() - timezone.timedelta(hours=24)
    timeline_records = PredictionRecord.objects.filter(
        timestamp__gte=last_24h
    ).order_by('timestamp')
    
    # Prepare timeline data
    timeline_labels = [r.timestamp.strftime('%H:%M') for r in timeline_records]
    timeline_data = [float(r.confidence * 100) for r in timeline_records]
    
    # Calculate fault distribution
    fault_count = PredictionRecord.objects.filter(
        timestamp__gte=last_24h,
        predicted_label='fault'
    ).count()
    normal_count = PredictionRecord.objects.filter(
        timestamp__gte=last_24h,
        predicted_label='normal'
    ).count()
    distribution_data = [normal_count, fault_count]
    
    context = {
        'recent_predictions': recent_predictions,
        'timeline_labels': timeline_labels,
        'timeline_data': timeline_data,
        'distribution_data': distribution_data,
    }
    return render(request, 'prediction/dashboard.html', context)

def predict_view(request):
    """View for manual prediction input."""
    return render(request, 'prediction/predict.html')


class MachinesAPIView(APIView):
    """Return latest machine state for each device (used by dashboard)."""
    def get(self, request):
        # collect unique device ids (exclude null/empty)
        device_ids = PredictionRecord.objects.exclude(device_id__isnull=True).exclude(device_id='')
        device_ids = device_ids.values_list('device_id', flat=True).distinct()

        machines = []
        for did in device_ids:
            latest = PredictionRecord.objects.filter(device_id=did).order_by('-timestamp').first()
            if not latest:
                continue
            # pull out common metrics if they exist in input_json
            inp = latest.input_json or {}
            machines.append({
                'device_id': did,
                'timestamp': latest.timestamp,
                'predicted_label': latest.predicted_label,
                'confidence': latest.confidence,
                'temperature': inp.get('Temperature') or inp.get('Temperature_C') or inp.get('temp'),
                'vibration': inp.get('Vibration'),
                'pressure': inp.get('Pressure'),
                'current': inp.get('Current'),
                'voltage': inp.get('Voltage'),
            })

        return Response({'machines': machines})

class RecordsAPIView(APIView):
    """API endpoint for fetching prediction records."""
    def get(self, request):
        agg_type = request.GET.get('agg', 'recent')
        
        if agg_type == 'count_by_label':
            # Last 24 hours distribution
            last_24h = timezone.now() - timezone.timedelta(hours=24)
            counts = PredictionRecord.objects.filter(
                timestamp__gte=last_24h
            ).values('predicted_label').annotate(
                count=Count('id')
            )
            # return a list for frontend compatibility
            return Response({'counts': list(counts)})
            
        # Default: return recent records
        records = PredictionRecord.objects.order_by('-timestamp')[:100]
        data = [{
            'id': record.id,
            'timestamp': record.timestamp,
            'device_id': record.device_id,
            'source': record.source,
            'predicted_label': record.predicted_label,
            'confidence': record.confidence,
        } for record in records]
        return Response(data)


class BulkUploadAPIView(APIView):
    """API endpoint for bulk CSV/Excel file upload and prediction."""
    
    def post(self, request):
        try:
            # Check if file is uploaded
            if 'file' not in request.FILES:
                return Response({
                    'error': 'No file uploaded. Please select a CSV or Excel file.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            file = request.FILES['file']
            
            # Load model and utilities
            model, scaler, feature_order = load_model()
            
            if model is None:
                return Response({
                    'error': 'ML model not loaded. Please ensure model file exists.'
                }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
            
            if feature_order is None:
                return Response({
                    'error': 'Feature order not found. Please ensure feature_order.json exists.'
                }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
            
            # Read the file based on extension
            file_extension = file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file)
            else:
                return Response({
                    'error': 'Unsupported file format. Please upload CSV or Excel file.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Validate that required columns exist
            missing_columns = set(feature_order) - set(df.columns)
            if missing_columns:
                return Response({
                    'error': f'Missing required columns: {", ".join(missing_columns)}',
                    'required_columns': feature_order,
                    'provided_columns': list(df.columns)
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Extract features in correct order
            X = df[feature_order].values
            
            # Scale features if scaler is available
            if scaler is not None:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X
            
            # Make predictions
            predictions = model.predict(X_scaled)
            prediction_probas = model.predict_proba(X_scaled)
            
            # Get confidence scores
            confidences = [float(proba[int(pred)]) for pred, proba in zip(predictions, prediction_probas)]
            
            # Create results dataframe
            results_df = df.copy()
            results_df['Prediction'] = ['fault' if p == 1 else 'normal' for p in predictions]
            results_df['Confidence'] = [round(c, 4) for c in confidences]
            results_df['Fault_Probability'] = [round(float(proba[1]), 4) for proba in prediction_probas]
            results_df['Normal_Probability'] = [round(float(proba[0]), 4) for proba in prediction_probas]
            
            # Convert to dictionary for JSON response
            results_dict = results_df.to_dict('records')
            
            # Optionally save to CSV in memory for download
            csv_buffer = BytesIO()
            results_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            return Response({
                'success': True,
                'message': f'Successfully processed {len(df)} rows',
                'results': results_dict,
                'summary': {
                    'total_rows': len(df),
                    'fault_count': int(np.sum(predictions == 1)),
                    'normal_count': int(np.sum(predictions == 0)),
                    'avg_confidence': round(np.mean(confidences), 4)
                }
            })
            
        except pd.errors.EmptyDataError:
            return Response({
                'error': 'The uploaded file is empty.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        except Exception as e:
            return Response({
                'error': f'Error processing file: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def bulk_upload_view(request):
    """View for bulk CSV/Excel upload page."""
    return render(request, 'prediction/bulk_upload.html')
