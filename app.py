#!/usr/bin/env python3
import os
import subprocess
import threading
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

UPLOAD_FOLDER = "/tmp/transcripciones"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

estado = {"progreso": "esperando", "texto": "", "archivo": ""}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transcribir", methods=["POST"])
def transcribir():
    if "video" not in request.files:
        return jsonify({"error": "No se recibió ningún archivo"}), 400

    archivo = request.files["video"]
    nombre = archivo.filename
    ruta_video = os.path.join(UPLOAD_FOLDER, nombre)
    archivo.save(ruta_video)

    estado["progreso"] = "procesando"
    estado["texto"] = ""
    estado["archivo"] = nombre

    hilo = threading.Thread(target=_procesar, args=(ruta_video,), daemon=True)
    hilo.start()

    return jsonify({"ok": True})


@app.route("/estado")
def ver_estado():
    return jsonify(estado)


def _procesar(ruta_video):
    ruta_audio = ruta_video + "_comprimido.mp3"
    try:
        estado["progreso"] = "comprimiendo audio..."
        subprocess.run(
            ["ffmpeg", "-i", ruta_video, "-vn", "-ar", "16000", "-ac", "1", "-b:a", "32k", ruta_audio, "-y"],
            capture_output=True
        )

        estado["progreso"] = "transcribiendo..."
        with open(ruta_audio, "rb") as f:
            resultado = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
            )

        texto = resultado.text.strip()

        estado["progreso"] = "listo"
        estado["texto"] = texto

    except Exception as e:
        estado["progreso"] = "error"
        estado["texto"] = str(e)
    finally:
        for ruta in [ruta_video, ruta_audio]:
            if os.path.exists(ruta):
                os.remove(ruta)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
