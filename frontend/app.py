import os
import sys
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# Add parent directory (CMTII) to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ocr_pipeline import run_ocr   # Import your OCR function

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), UPLOAD_FOLDER)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_url = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Run OCR
            result = run_ocr(filepath)

            image_url = f"/{UPLOAD_FOLDER}/{filename}"

    return render_template("index.html", result=result, image_url=image_url)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return app.send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True,port=5001)