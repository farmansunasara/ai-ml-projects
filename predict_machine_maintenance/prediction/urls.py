from django.urls import path
from django.shortcuts import redirect
from . import views

app_name = 'prediction'

def redirect_to_dashboard(request):
    return redirect('prediction:dashboard')

urlpatterns = [
    path('', redirect_to_dashboard, name='home'),
    path('api/predict/', views.PredictAPIView.as_view(), name='api_predict'),
    path('api/machines/', views.MachinesAPIView.as_view(), name='api_machines'),
    path('api/records/', views.RecordsAPIView.as_view(), name='api_records'),
    path('api/bulk-upload/', views.BulkUploadAPIView.as_view(), name='api_bulk_upload'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('predict/', views.predict_view, name='predict'),
    path('bulk-upload/', views.bulk_upload_view, name='bulk_upload'),
]
