from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
model = load_model("binary_model.h5")  # Load model saat aplikasi Flask dijalankan

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Mengambil data input dari request JSON
        features = np.array(data['features']).reshape(1, -1)  # Data diubah ke bentuk array
        prediction = model.predict(features)  # Prediksi menggunakan model
        result = "yes" if prediction[0][0] > 0.5 else "no"  # Konversi hasil prediksi
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
