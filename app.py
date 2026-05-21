from flask import Flask, request, jsonify, send_from_directory, render_template
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image
import numpy as np
import os
import cv2

Image.MAX_IMAGE_PIXELS = None

app = Flask(__name__)

# ── Absolute paths — work both locally and on Render ──
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # create on startup, not per-request

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model(os.path.join(BASE_DIR, 'anomaly_detection_model.keras'))

def preprocess_image(image, target_size):
    if image.mode != "L":
        image = image.convert("L")
    image = image.resize(target_size, Image.LANCZOS)
    image = np.array(image, dtype=np.float32)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
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

    # File size check — reject over 20MB before saving
    file.seek(0, 2)
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)
    if size_mb > 20:
        return jsonify({'error': f'File too large ({size_mb:.1f} MB). Please upload under 20 MB.'}), 400

    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        image = Image.open(file_path)
        image.thumbnail((4000, 4000), Image.LANCZOS)
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
        print(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def send_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)