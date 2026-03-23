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


LIMITE_BYTES = 24 * 1024 * 1024  # 24MB para tener margen


def _procesar(ruta_video):
    archivos_temp = [ruta_video]
    try:
        # Comprimir audio
        ruta_audio = ruta_video + ".mp3"
        archivos_temp.append(ruta_audio)
        estado["progreso"] = "comprimiendo audio..."
        subprocess.run(
            ["ffmpeg", "-i", ruta_video, "-vn", "-ar", "16000", "-ac", "1", "-b:a", "16k", ruta_audio, "-y"],
            capture_output=True
        )

        # Dividir en partes si supera el límite
        partes = _dividir_si_necesario(ruta_audio, archivos_temp)

        # Transcribir cada parte
        textos = []
        for i, parte in enumerate(partes):
            estado["progreso"] = f"transcribiendo parte {i+1}/{len(partes)}..."
            with open(parte, "rb") as f:
                resultado = client.audio.transcriptions.create(model="whisper-1", file=f)
            textos.append(resultado.text.strip())

        estado["progreso"] = "listo"
        estado["texto"] = " ".join(textos)

    except Exception as e:
        estado["progreso"] = "error"
        estado["texto"] = str(e)
    finally:
        for ruta in archivos_temp:
            if os.path.exists(ruta):
                os.remove(ruta)


def _dividir_si_necesario(ruta_audio, archivos_temp):
    if os.path.getsize(ruta_audio) <= LIMITE_BYTES:
        return [ruta_audio]

    # Obtener duración total
    resultado = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", ruta_audio],
        capture_output=True, text=True
    )
    duracion = float(resultado.stdout.strip())
    partes_necesarias = int(os.path.getsize(ruta_audio) / LIMITE_BYTES) + 1
    duracion_parte = duracion / partes_necesarias

    partes = []
    for i in range(partes_necesarias):
        inicio = i * duracion_parte
        ruta_parte = ruta_audio + f"_parte{i}.mp3"
        archivos_temp.append(ruta_parte)
        subprocess.run(
            ["ffmpeg", "-i", ruta_audio, "-ss", str(inicio), "-t", str(duracion_parte),
             "-c", "copy", ruta_parte, "-y"],
            capture_output=True
        )
        partes.append(ruta_parte)

    return partes


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
