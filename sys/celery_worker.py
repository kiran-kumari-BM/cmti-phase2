from celery_app import celery
from app import app
from models import db, Document
from ocr_pipeline import run_ocr
import logging


@celery.task(name="celery_worker.process_ocr_task")
def process_ocr_task(doc_id, file_path):

    with app.app_context():

        doc = Document.query.get(doc_id)

        if not doc:
            return

        try:
            text = run_ocr(file_path)
            doc.extracted_text = text
            doc.status = "completed"
            logging.info(f"OCR completed for doc {doc_id}")

        except Exception as e:
            doc.status = "failed"
            doc.extracted_text = f"OCR Error: {str(e)}"
            logging.error(f"OCR failed for doc {doc_id}: {e}")

        db.session.commit()