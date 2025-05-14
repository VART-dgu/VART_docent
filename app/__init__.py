from flask import Flask
from app.routes.generate_db import generate_db_bp
from app.routes.query import query_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(generate_db_bp)
    app.register_blueprint(query_bp)
    return app
