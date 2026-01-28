from django.db import models


class PredictionRecord(models.Model):
    SOURCE_CHOICES = [
        ('iot', 'IoT'),
        ('manual', 'Manual'),
    ]

    device_id = models.CharField(max_length=128, blank=True, null=True)
    source = models.CharField(max_length=10, choices=SOURCE_CHOICES, default='manual')
    input_json = models.JSONField()
    predicted_label = models.CharField(max_length=128)
    confidence = models.FloatField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    meta = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f"{self.predicted_label} @ {self.timestamp} ({self.source})"


class APIKey(models.Model):
    """Simple API key model for IoT devices / clients."""
    name = models.CharField(max_length=100, help_text='Human-friendly name for this key')
    key = models.CharField(max_length=128, unique=True)
    active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({'active' if self.active else 'inactive'})"

