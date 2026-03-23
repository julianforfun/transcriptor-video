#!/usr/bin/env python3
import os
import subprocess
import threading
import uuid
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=600)

UPLOAD_FOLDER = "/tmp/transcripciones"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

LIMITE_BYTES = 24 * 1024 * 1024  # 24MB
sesiones = {}  # { session_id: { "estado": ..., "texto": ... } }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chunk", methods=["POST"])
def recibir_chunk():
    session_id = request.form.get("session_id")
    chunk_index = int(request.form.get("chunk_index"))
    total_chunks = int(request.form.get("total_chunks"))
    nombre = request.form.get("nombre", "archivo")
    chunk = request.files["chunk"]

    carpeta = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(carpeta, exist_ok=True)
    chunk.save(os.path.join(carpeta, f"chunk_{chunk_index}"))

    if session_id not in sesiones:
        sesiones[session_id] = {"estado": "subiendo...", "texto": "", "total": total_chunks, "recibidos": 0}

    sesiones[session_id]["recibidos"] += 1

    # Si llegaron todos los chunks, procesar
    if sesiones[session_id]["recibidos"] == total_chunks:
        sesiones[session_id]["estado"] = "procesando..."
        nombre_base = os.path.splitext(nombre)[0]
        ruta_completa = os.path.join(carpeta, nombre_base + ".orig")

        # Ensamblar archivo
        with open(ruta_completa, "wb") as f:
            for i in range(total_chunks):
                with open(os.path.join(carpeta, f"chunk_{i}"), "rb") as c:
                    f.write(c.read())

        hilo = threading.Thread(target=_procesar, args=(session_id, ruta_completa, carpeta), daemon=True)
        hilo.start()

    return jsonify({"ok": True})


@app.route("/estado/<session_id>")
def ver_estado(session_id):
    return jsonify(sesiones.get(session_id, {"estado": "no encontrado", "texto": ""}))


def _procesar(session_id, ruta_video, carpeta):
    ruta_audio = ruta_video + ".mp3"
    try:
        sesiones[session_id]["estado"] = "comprimiendo audio..."
        subprocess.run(
            ["ffmpeg", "-i", ruta_video, "-vn", "-ar", "16000", "-ac", "1", "-b:a", "16k", ruta_audio, "-y"],
            capture_output=True
        )

        partes = _dividir_si_necesario(session_id, ruta_audio, carpeta)

        textos = []
        for i, parte in enumerate(partes):
            sesiones[session_id]["estado"] = f"transcribiendo parte {i+1}/{len(partes)}..."
            with open(parte, "rb") as f:
                resultado = client.audio.transcriptions.create(model="whisper-1", file=f)
            textos.append(resultado.text.strip())

        sesiones[session_id]["estado"] = "listo"
        sesiones[session_id]["texto"] = " ".join(textos)

    except Exception as e:
        sesiones[session_id]["estado"] = "error"
        sesiones[session_id]["texto"] = str(e)
    finally:
        import shutil
        shutil.rmtree(carpeta, ignore_errors=True)


def _dividir_si_necesario(session_id, ruta_audio, carpeta):
    if os.path.getsize(ruta_audio) <= LIMITE_BYTES:
        return [ruta_audio]

    resultado = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", ruta_audio],
        capture_output=True, text=True
    )
    duracion = float(resultado.stdout.strip())
    n_partes = int(os.path.getsize(ruta_audio) / LIMITE_BYTES) + 1
    dur_parte = duracion / n_partes

    partes = []
    for i in range(n_partes):
        ruta_parte = os.path.join(carpeta, f"parte_{i}.mp3")
        subprocess.run(
            ["ffmpeg", "-i", ruta_audio, "-ss", str(i * dur_parte), "-t", str(dur_parte),
             "-c", "copy", ruta_parte, "-y"],
            capture_output=True
        )
        partes.append(ruta_parte)

    return partes


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
