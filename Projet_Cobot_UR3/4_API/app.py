from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import tensorflow as tf
import logging

# Configuration initiale
app = Flask(__name__)
CORS(app)  # Autorise les requêtes cross-origin
logging.basicConfig(level=logging.INFO)

# Chargement des modèles (une seule fois au démarrage)
try:
    model = tf.keras.models.load_model("lstm_model.keras")
    scaler = joblib.load("scaler.pkl")
    logging.info("✅ Modèles chargés avec succès")
except Exception as e:
    logging.error(f"❌ Erreur lors du chargement des modèles : {str(e)}")
    raise

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Récupération des données
        data = request.get_json()
        if not data or "input" not in data:
            return jsonify({"error": "Format invalide. Utilisez {'input': [...]}"}), 400
        
        input_data = np.array(data["input"])
        
        # 2. Validation du shape
        if input_data.ndim != 3:
            return jsonify({
                "error": f"Shape invalide. Attendu (1, 10, n_features), reçu {input_data.shape}"
            }), 400

        # 3. Prétraitement
        n_samples, n_steps, n_features = input_data.shape
        input_reshaped = input_data.reshape(-1, n_features)
        input_scaled = scaler.transform(input_reshaped).reshape(n_samples, n_steps, n_features)
        
        # 4. Prédiction
        prediction = model.predict(input_scaled)
        pred_class = (prediction > 0.5).astype(int).tolist()

        return jsonify({
            "prediction_proba": prediction.tolist(),
            "prediction_class": pred_class,
            "model_used": "LSTM"
        })

    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {str(e)}")
        return jsonify({"error": f"Erreur interne : {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)