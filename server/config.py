
import os

def config(app):
    # Celery configuration
    app.config['CELERY_BROKER_URL'] = os.environ.get('CELERY_BROKER_URL')  # Replace with your Redis URL
    app.config['CELERY_RESULT_BACKEND'] = os.environ.get('CELERY_RESULT_BACKEND')  # Replace with your Redis URL
    return app