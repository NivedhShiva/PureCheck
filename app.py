# app.py

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import base64
import cv2
from ai_edge_litert.interpreter import Interpreter

app = Flask(__name__)
CORS(app)  # allows browser to call Flask

# Load model once at startup
print("Loading model...")
interpreter = Interpreter(model_path='food_model.tflite')
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
class_names    = ['adulterated', 'pure']
print("Model loaded ✅")

def predict_image(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    class_index = np.argmax(output[0])
    confidence  = float(np.max(output[0])) * 100
    result      = class_names[class_index]

    return result, confidence

# Serve the HTML frontend
@app.route('/')
def index():
    return send_from_directory('.', 'food_adulteration_detector.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data     = request.get_json()
        img_data = data['image']

        # Strip base64 header
        if ',' in img_data:
            img_data = img_data.split(',')[1]

        # Decode base64 to image
        img_bytes = base64.b64decode(img_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img       = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400

        result, confidence = predict_image(img)

        return jsonify({
            'safe'      : result == 'pure',
            'result'    : result,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)