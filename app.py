import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__)
paquete = joblib.load("modelo_alquiler.pkl")

if isinstance(paquete, dict):
    modelo, columnas = paquete["modelo"], paquete["columnas"]
else:
    modelo = paquete
    columnas = list(modelo.feature_names_in_)

def numero(valor):
    valor = (valor or "").strip().replace(",", ".")
    return float(valor) if valor != "" else np.nan

@app.route("/")
def inicio():
    return render_template("index.html")

@app.route("/manifest.json")
def manifest():
    return send_from_directory("static", "manifest.json")
@app.route("/sw.js")
def service_worker():
    return send_from_directory("static", "sw.js", mimetype="application/javascript")

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
