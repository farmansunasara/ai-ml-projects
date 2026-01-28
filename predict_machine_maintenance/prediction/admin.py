from django.contrib import admin
from .models import PredictionRecord

from .models import APIKey


@admin.register(PredictionRecord)
class PredictionRecordAdmin(admin.ModelAdmin):
    list_display = ('id', 'predicted_label', 'confidence', 'source', 'device_id', 'timestamp')
    list_filter = ('source', 'predicted_label')
    readonly_fields = ('timestamp',)


@admin.register(APIKey)
class APIKeyAdmin(admin.ModelAdmin):
    list_display = ('name', 'key', 'active', 'created_at')
    readonly_fields = ('created_at',)
    search_fields = ('name', 'key')
    list_filter = ('active',)
