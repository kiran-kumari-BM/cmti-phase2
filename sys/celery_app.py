from celery import Celery

celery = Celery(
    "cmti",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)

celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=True,
)

# 🔥 Force import of task module
import celery_worker