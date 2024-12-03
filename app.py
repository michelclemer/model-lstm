# app.py

from flask import Flask, request, jsonify
from model_prediction import load_trained_model, predict_price

app = Flask(__name__)

model, scaler = load_trained_model()

LOOK_BACK = 60

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = data.get('historical_prices', [])

    # Verifica se os dados de entrada têm o tamanho correto
    if len(input_data) != LOOK_BACK:
        return jsonify({'error': f'Input data must contain exactly {LOOK_BACK} historical prices.'}), 400

    try:
        prediction = predict_price(model, scaler, input_data)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
