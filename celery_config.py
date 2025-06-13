from celery import Celery
import os

RABBITMQ_BROKER_URL = os.environ.get('RABBITMQ_BROKER_URL', 'amqp://guest:guest@localhost:5672//')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'rpc://')

celery_app = Celery(
    'tasks',
    broker=RABBITMQ_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['tasks']
)

celery_app.conf.update(
    task_track_started=True,
    broker_connection_retry_on_startup=True
)