import os

class Config:
    SECRET_KEY = "super-secret-key"

    SQLALCHEMY_DATABASE_URI = "postgresql:///cmti_docs"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
    OUTPUT_FOLDER = os.path.join(os.getcwd(), "outputs")

import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"