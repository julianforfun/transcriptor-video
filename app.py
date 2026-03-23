#!/usr/bin/env python3
import os
import subprocess
import threading
import webbrowser
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
UPLOAD_FOLDER = os.path.expanduser("~/Desktop/transcripciones")
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

    hilo = threading.Thread(target=_procesar, args=(ruta_video, nombre), daemon=True)
    hilo.start()

    return jsonify({"ok": True})


@app.route("/estado")
def ver_estado():
    return jsonify(estado)


def _procesar(ruta_video, nombre):
    nombre_base = os.path.splitext(ruta_video)[0]
    ruta_audio = nombre_base + "_temp.mp3"
    ruta_txt = nombre_base + "_transcripcion.txt"

    try:
        estado["progreso"] = "extrayendo audio..."
        resultado = subprocess.run(
            ["ffmpeg", "-i", ruta_video, "-q:a", "0", "-map", "a", ruta_audio, "-y"],
            capture_output=True, text=True
        )
        if resultado.returncode != 0:
            estado["progreso"] = "error"
            estado["texto"] = "No se pudo extraer el audio del video."
            return

        estado["progreso"] = "transcribiendo..."
        import whisper
        modelo = whisper.load_model("base")
        resultado_whisper = modelo.transcribe(ruta_audio)
        texto = resultado_whisper["text"].strip()
        idioma = resultado_whisper.get("language", "desconocido")

        with open(ruta_txt, "w", encoding="utf-8") as f:
            f.write(texto)

        if os.path.exists(ruta_audio):
            os.remove(ruta_audio)

        estado["progreso"] = f"listo | idioma: {idioma}"
        estado["texto"] = texto
        estado["ruta"] = ruta_txt

    except Exception as e:
        estado["progreso"] = "error"
        estado["texto"] = str(e)


if __name__ == "__main__":
    webbrowser.open("http://localhost:5000")
    app.run(debug=False)
