from rest_framework import serializers


class PredictInputSerializer(serializers.Serializer):
    # Accept either a dict of feature_name: value or a list of ordered values
    data = serializers.DictField(child=serializers.FloatField(), required=False)
    values = serializers.ListField(child=serializers.FloatField(), required=False)
    device_id = serializers.CharField(required=False, allow_null=True, allow_blank=True)


class PredictOutputSerializer(serializers.Serializer):
    predicted_label = serializers.CharField()
    confidence = serializers.FloatField()
    timestamp = serializers.DateTimeField()
    record_id = serializers.IntegerField()
