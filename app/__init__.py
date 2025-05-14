from flask import Flask
from app.routes.generate_db import generate_db_bp
from app.routes.query import query_bp
import subprocess
import requests

def create_app():
    try:
        requests.get("http://localhost:11434")
    except requests.exceptions.ConnectionError:
        subprocess.Popen("ollama serve &", shell=True)

        # 3. Wait for server to actually start (최대 10초까지 재시도)
        for _ in range(10):
            try:
                time.sleep(1)
                requests.get("http://localhost:11434")
                break
            except requests.exceptions.ConnectionError:
                continue
        else:
            raise RuntimeError("Ollama 서버를 시작할 수 없습니다.")
    app = Flask(__name__)
    app.register_blueprint(generate_db_bp)
    app.register_blueprint(query_bp)
    return app
