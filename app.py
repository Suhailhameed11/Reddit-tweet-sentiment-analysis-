from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load model, vectorizer, and encoder
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    label = label_encoder.inverse_transform([pred])[0]

    return jsonify({'sentiment': label})