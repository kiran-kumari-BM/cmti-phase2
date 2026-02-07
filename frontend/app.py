from flask import Flask, render_template, request
import os
import subprocess

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["image"]
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        # Run your existing backend
        process = subprocess.run(["python", "../yolo-model/main.py", image_path],text=True,capture_output=True)

        print("STDOUT:", process.stdout)
        print("STDERR:", process.stderr)

        result = process.stdout or process.stderr

        

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=False, threaded=True, host='127.0.0.1')
