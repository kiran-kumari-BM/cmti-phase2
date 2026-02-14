from functools import wraps
from flask import abort
from flask_login import current_user

def role_required(required_role):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                abort(401)

            if current_user.role != required_role:
                abort(403)

            return func(*args, **kwargs)

        return wrapper
    return decorator