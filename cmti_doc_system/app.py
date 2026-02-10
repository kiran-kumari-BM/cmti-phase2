import os
import threading
from flask import Flask, render_template, request, redirect, url_for, send_file
from flask_login import LoginManager, login_required, current_user
from werkzeug.utils import secure_filename
from docx import Document as WordDocument
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import zipfile
from io import BytesIO
from config import Config
from models import db, User, Document
from auth import auth
from ocr_engine import run_ocr
from rag_engine import ask_question

# -------------------------------------------------
# Flask App Setup
# -------------------------------------------------

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

app.register_blueprint(auth)

# -------------------------------------------------
# Background OCR Worker
# -------------------------------------------------

def process_ocr_background(app, doc_id, path):
    with app.app_context():
        doc = Document.query.get(doc_id)

        if not doc:
            return

        try:
            text = run_ocr(path)
            doc.extracted_text = text
            doc.status = "completed"
        except Exception as e:
            doc.status = "failed"
            doc.extracted_text = f"OCR Error: {str(e)}"

        db.session.commit()

# -------------------------------------------------
# Dashboard
# -------------------------------------------------

@app.route("/")
@login_required
def dashboard():
    documents = Document.query.filter_by(
        user_id=current_user.id
    ).order_by(Document.id.desc()).all()

    return render_template(
        "dashboard.html",
        user=current_user,
        documents=documents
    )

# -------------------------------------------------
# Upload (Non Blocking)
# -------------------------------------------------

@app.route("/upload", methods=["POST"])
@login_required
def upload():
    files = request.files.getlist("documents")

    for file in files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            save_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(save_path)

            doc = Document(
                filename=filename,
                stored_path=save_path,
                user_id=current_user.id,
                status="processing"
            )

            db.session.add(doc)
            db.session.commit()

            # 🚀 Background OCR
            thread = threading.Thread(
                target=process_ocr_background,
                args=(app, doc.id, save_path)
            )
            thread.daemon = True
            thread.start()

    return redirect(url_for("dashboard"))

# -------------------------------------------------
# View & Edit Document
# -------------------------------------------------

@app.route("/document/<int:doc_id>", methods=["GET", "POST"])
@login_required
def view_document(doc_id):
    doc = Document.query.get_or_404(doc_id)

    if doc.user_id != current_user.id:
        return "Unauthorized", 403

    if request.method == "POST":
        edited_text = request.form.get("edited_text")
        doc.extracted_text = edited_text
        db.session.commit()
        return redirect(url_for("dashboard"))

    return render_template("view_document.html", doc=doc)

# -------------------------------------------------
# Download Word
# -------------------------------------------------

@app.route("/download/word/<int:doc_id>")
@login_required
def download_word(doc_id):
    doc = Document.query.get_or_404(doc_id)

    if doc.user_id != current_user.id:
        return "Unauthorized", 403

    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", f"{doc.id}.docx")

    word_doc = WordDocument()
    word_doc.add_paragraph(doc.extracted_text or "")
    word_doc.save(output_path)

    return send_file(
        output_path,
        as_attachment=True,
        download_name=f"{doc.filename}.docx"
    )

# -------------------------------------------------
# Download PDF
# -------------------------------------------------

@app.route("/download/pdf/<int:doc_id>")
@login_required
def download_pdf(doc_id):
    doc = Document.query.get_or_404(doc_id)

    if doc.user_id != current_user.id:
        return "Unauthorized", 403

    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", f"{doc.id}.pdf")

    pdf = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    for line in (doc.extracted_text or "").split("\n"):
        elements.append(Paragraph(line, styles["Normal"]))

    pdf.build(elements)

    return send_file(
        output_path,
        as_attachment=True,
        download_name=f"{doc.filename}.pdf"
    )


@app.route("/download/bulk", methods=["POST"])
@login_required
def bulk_download():
    selected_ids = request.form.getlist("selected_docs")

    if not selected_ids:
        return redirect(url_for("dashboard"))

    memory_file = BytesIO()

    with zipfile.ZipFile(memory_file, "w") as zf:
        for doc_id in selected_ids:
            doc = Document.query.get(int(doc_id))

            if doc and doc.user_id == current_user.id:
                filename = f"{doc.filename}.txt"
                zf.writestr(filename, doc.extracted_text or "")

    memory_file.seek(0)

    return send_file(
        memory_file,
        download_name="documents.zip",
        as_attachment=True
    )

@app.route("/ask/<int:doc_id>", methods=["GET", "POST"])
@login_required
def ask_question_route(doc_id):

    doc = Document.query.get_or_404(doc_id)

    if doc.user_id != current_user.id:
        return "Unauthorized", 403

    answer = None
    context = None

    if request.method == "POST":
        question = request.form.get("question")
        answer, context = ask_question(doc.extracted_text, question)

    return render_template(
        "ask_document.html",
        doc=doc,
        answer=answer,
        context=context
    )

# -------------------------------------------------
# Run App
# -------------------------------------------------

if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True)