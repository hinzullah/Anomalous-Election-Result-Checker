
import os
import cv2
import numpy as np
import pytesseract
from skimage import filters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical 

# Preprocessing Function
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image at path: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (256, 256))  # Resize to a fixed size
    return image

# Blurriness Detection
def is_blurry(image, threshold=100):
    variance_of_laplacian = cv2.Laplacian(image, cv2.CV_64F).var()
    return variance_of_laplacian < threshold

# Image Quality Metrics
def compute_image_quality(image):
    edges = filters.sobel(image)
    quality = edges.var()  # Variance of edges as a simple quality metric
    return quality

# Alteration Detection (Cancellation/Changes)
def detect_text_alterations(image):
    text = pytesseract.image_to_string(image)
    alterations = ["cancel", "crossed out", "strike through"]  # Example keywords indicating alterations
    for alteration in alterations:
        if alteration in text.lower():
            return True
    return False

# Check if Image is Normal
def is_normal(image):
    if is_blurry(image) or compute_image_quality(image) < 10 or detect_text_alterations(image):
        return False
    return True

# Prepare Data
def prepare_data(image_folder):
    X = []  # List of preprocessed images
    y = []  # Corresponding labels (0 for normal, 1 for anomalous)

    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_path in image_paths:
        image = preprocess_image(image_path)
        X.append(img_to_array(image))
        label = 0 if is_normal(image) else 1
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    y = to_categorical(y, 2)
    return X, y

# Build the Model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the Model
def train_model(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy}")

# Predict Image Class
def predict_image_class(model, image_path):
    image = preprocess_image(image_path)
    image = np.expand_dims(img_to_array(image), axis=0)
    prediction = model.predict(image)
    return "Normal" if np.argmax(prediction) == 0 else "Anomalous"

# Main Execution
if __name__ == "__main__":
    image_folder = r"C:\Users\DONKAMS\Desktop\Final year project\anomalies"  # Folder with images
    X, y = prepare_data(image_folder)
    model = build_model()
    train_model(model, X, y)

    # Test the model with a new image
    '''test_image_path = r"C:\Users\DONKAMS\Desktop\Final year project\images\test_image.jpg"  # Replace with a test image path
    result = predict_image_class(model, test_image_path)
    print(f"The image is classified as: {result}")'''
