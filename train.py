import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATASET_DIR = '/home/hinzullah/Documents/FInal year project'
NORMAL_DIR  = os.path.join(DATASET_DIR, 'normal')
ANOMALY_DIR = os.path.join(DATASET_DIR, 'anomalies')
MODEL_OUT   = os.path.join(DATASET_DIR, 'anomaly_detection_model.h5')
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 16
EPOCHS      = 50
TEST_SPLIT  = 0.2
RANDOM_SEED = 42
# ─────────────────────────────────────────


def preprocess_image(path):
    """Grayscale → resize → blur → normalise. Matches app.py exactly."""
    try:
        img = Image.open(path)
        img.thumbnail((4000, 4000), Image.LANCZOS)
        if img.mode != 'L':
            img = img.convert('L')
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        arr = cv2.GaussianBlur(arr, (5, 5), 0)
        arr = arr / 255.0
        arr = np.expand_dims(arr, axis=-1)  # → (128, 128, 1)
        return arr
    except Exception as e:
        print(f"  ⚠  Skipping {path}: {e}")
        return None


def load_dataset():
    images, labels = [], []
    print("\n📂 Loading images...")

    # Explicit tuples — never depends on folder sort order
    sources = [
        (NORMAL_DIR,  0, 'NORMAL'),
        (ANOMALY_DIR, 1, 'ANOMALY'),
    ]

    for folder, label, name in sources:
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        print(f"  [{label}] {name}: {len(files)} files → {folder}")
        for fname in files:
            arr = preprocess_image(os.path.join(folder, fname))
            if arr is not None:
                images.append(arr)
                labels.append(label)

    print(f"\n✅ Loaded {len(images)} images total")
    print(f"   label=0 NORMAL:  {labels.count(0)}")
    print(f"   label=1 ANOMALY: {labels.count(1)}")
    return np.array(images), np.array(labels)


def augment_image(img):
    if np.random.rand() > 0.5:
        img = np.fliplr(img)
    if np.random.rand() > 0.5:
        img = np.flipud(img)
    delta = np.random.uniform(-0.15, 0.15)
    img = np.clip(img + delta, 0.0, 1.0)
    return img


def augment_dataset(X, y, multiplier):
    print(f"  Augmenting ×{multiplier} → {len(X) * multiplier} images")
    X_aug, y_aug = [X.copy()], [y.copy()]
    for _ in range(multiplier - 1):
        X_aug.append(np.array([augment_image(img.copy()) for img in X]))
        y_aug.append(y.copy())
    X_out = np.concatenate(X_aug, axis=0)
    y_out = np.concatenate(y_aug, axis=0)
    idx = np.random.permutation(len(X_out))
    return X_out[idx], y_out[idx]


def build_model():
    model = models.Sequential([
        # Block 1 — small and simple
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Block 2
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Block 3
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Classifier — small dense layer
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid'),  # 0 = normal, 1 = anomaly
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # slower lr
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Val')
    axes[0].set_title('Accuracy')
    axes[0].legend()
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Val')
    axes[1].set_title('Loss')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DATASET_DIR, 'training_history.png'), dpi=120)
    print("📊 Saved → training_history.png")
    plt.show()


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(DATASET_DIR, 'confusion_matrix.png'), dpi=120)
    print("📊 Saved → confusion_matrix.png")
    plt.show()


# ─────────────────────────────────────────
# MAIN — only one block, no duplicates
# ─────────────────────────────────────────
if __name__ == '__main__':

    # 1. Load
    X, y = load_dataset()

    if len(X) == 0:
        print("❌ No images loaded. Check folder paths.")
        exit(1)

    # 2. Label verification — pause before training
    print("\n🔍 LABEL VERIFICATION:")
    print(f"   First 10 labels : {y[:10].tolist()}")
    print(f"   label=0 NORMAL  : {(y == 0).sum()} images")
    print(f"   label=1 ANOMALY : {(y == 1).sum()} images")
    go = input("\n   ⏸  Labels look correct? Press Enter to train, Ctrl+C to abort: ")

    # 3. Split BEFORE augmentation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, stratify=y, random_state=RANDOM_SEED
    )
    print(f"\n📊 Split → Train: {len(X_train)}, Test: {len(X_test)}")

    # 4. Balance classes via targeted augmentation
    print("\n🔁 Balancing classes...")
    X_normal  = X_train[y_train == 0]
    y_normal  = y_train[y_train == 0]
    X_anomaly = X_train[y_train == 1]
    y_anomaly = y_train[y_train == 1]

    X_anomaly_aug, y_anomaly_aug = augment_dataset(X_anomaly, y_anomaly, multiplier=6)
    X_normal_aug,  y_normal_aug  = augment_dataset(X_normal,  y_normal,  multiplier=2)

    X_train = np.concatenate([X_normal_aug, X_anomaly_aug], axis=0)
    y_train = np.concatenate([y_normal_aug, y_anomaly_aug], axis=0)
    idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    print(f"\n⚖️  Balanced training set:")
    print(f"   Normal  (0): {(y_train == 0).sum()}")
    print(f"   Anomaly (1): {(y_train == 1).sum()}")

    # 5. Build
    print("\n🏗  Building model...")
    model = build_model()
    model.summary()

    # 6. Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=12,
              restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_OUT, monitor='val_loss',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=4, min_lr=1e-6, verbose=1),
    ]

    # 7. Train
    print("\n🚀 Training...\n")
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight={0: 1.0, 1: 2.0},
    )

    # 8. Evaluate
    print("\n📋 Evaluating on test set...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    print("\n" + classification_report(y_test, y_pred,
                                       target_names=['Normal', 'Anomaly']))

    # 9. Plots
    plot_history(history)
    plot_confusion(y_test, y_pred)

    print(f"\n✅ Model saved → {MODEL_OUT}")