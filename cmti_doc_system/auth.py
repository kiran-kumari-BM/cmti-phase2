from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User

auth = Blueprint("auth", __name__)


# ================= REGISTER =================
@auth.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")

        # Check if email already exists
        existing_user = User.query.filter_by(email=email).first()

        if existing_user:
            flash("⚠ Email already registered. Please login.", "danger")
            return redirect(url_for("auth.register"))

        hashed_password = generate_password_hash(password)

        new_user = User(
            name=name,
            email=email,
            password=hashed_password
        )

        db.session.add(new_user)
        db.session.commit()

        flash("✅ Registration successful. Please login.", "success")
        return redirect(url_for("auth.login"))

    return render_template("register.html")


# ================= LOGIN =================
@auth.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()

        # User not found
        if not user:
            flash("❌ No account found with this email.", "danger")
            return redirect(url_for("auth.login"))

        # Wrong password
        if not check_password_hash(user.password, password):
            flash("❌ Incorrect password.", "danger")
            return redirect(url_for("auth.login"))

        login_user(user)
        flash("✅ Login successful!", "success")

        return redirect(url_for("dashboard"))

    return render_template("login.html")


# ================= LOGOUT =================
@auth.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "info")
    return redirect(url_for("auth.login"))