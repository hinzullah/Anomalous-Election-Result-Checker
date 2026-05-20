from flask import Flask, request, jsonify, send_from_directory, render_template
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image
import numpy as np
import os
import cv2

Image.MAX_IMAGE_PIXELS = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

model = load_model('/home/hinzullah/Documents/FInal year project/anomaly_detection_model.h5')

def preprocess_image(image, target_size):
    if image.mode != "L":
        image = image.convert("L")
    image = image.resize(target_size, Image.LANCZOS)
    image = np.array(image, dtype=np.float32)
    image = cv2.GaussianBlur(image, (5, 5), 0)  # blur only, no Otsu
    image = image / 255.0                        # normalise
    image = np.expand_dims(image, axis=-1)       # → (128, 128, 1)
    image = np.expand_dims(image, axis=0)        # → (1, 128, 128, 1)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        image = Image.open(file_path)
        image.thumbnail((4000, 4000), Image.LANCZOS)  # cap huge scans
        processed_image = preprocess_image(image, target_size=(128, 128))

        prediction = model.predict(processed_image)
        print(f"Raw prediction score: {prediction}")

        score = float(prediction[0][0])
        result = 'Anomaly' if score > 0.3 else 'Normal'

        return jsonify({
            'result': result,
            'features': 'Irregular scoresheet' if result == 'Anomaly' else 'Clean scoresheet',
            'confidence': score,
            'file_path': f'/uploads/{file.filename}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def send_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)