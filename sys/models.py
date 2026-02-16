from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(120), unique=True)
    password = db.Column(db.String(200))
    role = db.Column(db.String(20), default="user")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255))
    stored_path = db.Column(db.String(500))
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50), default="uploaded")
    embedding_data = db.Column(db.PickleType, nullable=True)
    extracted_text = db.Column(db.Text)

    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))

from datetime import datetime

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    document_id = db.Column(db.Integer, db.ForeignKey('document.id'))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

    role = db.Column(db.String(20))  # "user" or "assistant"
    content = db.Column(db.Text)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)



from datetime import datetime

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    document_id = db.Column(
        db.Integer,
        db.ForeignKey("document.id"),
        nullable=False
    )

    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    citation = db.Column(db.Text)

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    document = db.relationship(
        "Document",
        backref=db.backref("chats", lazy=True)
    )
