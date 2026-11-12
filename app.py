import struct
import zlib
import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

app = Flask(__name__)
paquete = joblib.load("modelo_alquiler.pkl")

if isinstance(paquete, dict):
    modelo, columnas = paquete["modelo"], paquete["columnas"]
else:
    modelo = paquete
    columnas = list(modelo.feature_names_in_)

def crear_png_azul(size):
    raw = bytearray()
    cx = size / 2.0
    for y in range(size):
        raw.append(0)
        # Degradado azul moderno (de celeste #0099ff arriba a azul intenso #0052cc abajo)
        t = y / size
        r_bg = 0
        g_bg = int(153 * (1 - t) + 82 * t)
        b_bg = int(255 * (1 - t) + 204 * t)
        
        for x in range(size):
            # Coordenadas normalizadas entre -1 y 1
            nx = (x - cx) / (size * 0.35)
            ny = (y - cx) / (size * 0.35)
            
            # Techo triangular de la casa
            en_techo = (-0.85 <= ny <= -0.1) and (abs(nx) <= (ny + 0.85) * 1.35)
            # Cuerpo cuadrado de la casa
            en_cuerpo = (-0.1 < ny <= 0.75) and (abs(nx) <= 0.72)
            # Puerta central
            en_puerta = (0.25 <= ny <= 0.75) and (abs(nx) <= 0.2)
            
            if (en_techo or en_cuerpo) and not en_puerta:
                raw.extend((255, 255, 255, 255)) # Casa blanca
            else:
                raw.extend((r_bg, g_bg, b_bg, 255)) # Fondo degradado azul

    def chunk(tag, data):
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xffffffff)
    ihdr = struct.pack("!IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b"")

    def chunk(tag, data):
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xffffffff)
    ihdr = struct.pack("!IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b"")

def numero(valor):
    valor = (valor or "").strip().replace(",", ".")
    return float(valor) if valor != "" else np.nan

@app.route("/")
def inicio():
    return render_template("index.html")

@app.route("/manifest.json")
def manifest():
    return send_from_directory("static", "manifest.json", mimetype="application/manifest+json")

@app.route("/sw.js")
def sw():
    return send_from_directory("static", "sw.js", mimetype="application/javascript")

@app.route("/icon-192.png")
def icon_192():
    return Response(crear_png_azul(192), mimetype="image/png")

@app.route("/icon-512.png")
def icon_512():
    return Response(crear_png_azul(512), mimetype="image/png")

@app.route("/predecir", methods=["POST"])
def predecir():
    d = request.get_json(force=True)
    try:
        fila = {c: 0.0 for c in columnas}
        fila["dormitorios"] = numero(str(d.get("dormitorios")))
        fila["banos"] = numero(str(d.get("banos")))
        fila["area_m2"] = numero(str(d.get("area_m2")))
        fila["antiguedad"] = numero(str(d.get("antiguedad", "")))
        fila["mantenimiento_soles"] = numero(str(d.get("mantenimiento_soles", "")))
        fila["zona_lima"] = float(d.get("zona_lima", 0))
        
        tipo = "tipo_" + str(d.get("tipo", "Apartamento"))
        if tipo in fila:
            fila[tipo] = 1.0
            
        if np.isnan([fila["dormitorios"], fila["banos"], fila["area_m2"]]).any():
            return jsonify(error="Completa dormitorios, baños y área."), 400
            
        X = pd.DataFrame([fila], columns=columnas)
        precio = float(modelo.predict(X)[0])
        return jsonify(precio=round(precio, 2))
    except (ValueError, TypeError):
        return jsonify(error="Datos no válidos."), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
