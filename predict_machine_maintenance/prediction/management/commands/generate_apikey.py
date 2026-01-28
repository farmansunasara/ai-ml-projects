from django.core.management.base import BaseCommand
from prediction.models import APIKey
import secrets


class Command(BaseCommand):
    help = 'Generate a new API key for IoT devices or clients and print it.'

    def add_arguments(self, parser):
        parser.add_argument('--name', type=str, default='iot-device', help='Human friendly name for the API key')
        parser.add_argument('--bytes', type=int, default=24, help='Number of random bytes to use for token_urlsafe (default: 24)')
        parser.add_argument('--activate', action='store_true', help='Create the key as active (default true)')

    def handle(self, *args, **options):
        name = options.get('name')
        num_bytes = options.get('bytes') or 24
        active_flag = True  # activated by default

        # generate a url-safe token
        key = secrets.token_urlsafe(num_bytes)

        # ensure uniqueness
        max_tries = 5
        tries = 0
        while APIKey.objects.filter(key=key).exists() and tries < max_tries:
            key = secrets.token_urlsafe(num_bytes)
            tries += 1

        if APIKey.objects.filter(key=key).exists():
            raise RuntimeError('Could not generate a unique API key after several attempts')

        api = APIKey.objects.create(name=name, key=key, active=active_flag)

        self.stdout.write(self.style.SUCCESS('API key generated successfully'))
        self.stdout.write(f'Name: {api.name}')
        self.stdout.write(f'Key: {api.key}')
        self.stdout.write(f'Active: {api.active}')
