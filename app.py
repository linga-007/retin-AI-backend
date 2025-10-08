from flask import Flask, request, jsonify
from flask_cors import CORS   # import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
CORS(app)

# Load your trained model (update the path to your model)
MODEL_PATH = './retinopathy_model.h5'
model = load_model(MODEL_PATH)

# Class labels (modify based on your training)
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    preds = model.predict(img_array)
    result = CLASS_NAMES[np.argmax(preds)]
    return result

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)

    # Run prediction
    result = model_predict(file_path)

    # Remove temporary file
    os.remove(file_path)

    return jsonify({'result': result})

@app.route('/test', methods=['GET'])
def test():
    return "API is working!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
