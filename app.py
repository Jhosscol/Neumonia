from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image
import os

app = Flask(__name__)

# -------------------------------
# CARGAR MODELO
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "modelo_neumonia")

print("🔹 Cargando modelo...")
modelo = keras.models.load_model(MODEL_PATH)
print("✅ Modelo cargado correctamente")

# -------------------------------
# RUTAS
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    img = Image.open(file).resize((224,224)).convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = modelo.predict(img_array)

    neumonia_prob = float(pred[0][0])
    normal_prob = float(pred[0][1])

    decision = "NEUMONÍA" if neumonia_prob >= 0.372 else "NORMAL"

    return jsonify({
        "P(Neumonía)": neumonia_prob,
        "P(Normal)": normal_prob,
        "Decisión": decision
    })

# -------------------------------
# EJECUTAR (RENDER)
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
