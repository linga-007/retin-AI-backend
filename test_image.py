import os
import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model("./retinopathy_model.h5")

# Class mapping
stage_mapping = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

def predict_stage(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not read {image_path}")
        return None, None

    # Resize and preprocess
    img = cv2.resize(img, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img, verbose=0)
    class_idx = np.argmax(pred, axis=1)[0]

    return class_idx, stage_mapping[class_idx]

# Folder path
folder_path = "../Dataset/test"

# Loop through images and print results
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(folder_path, filename)
        pred_class, pred_stage = predict_stage(img_path)
        if pred_stage:
            print(f"{filename} -> Class {pred_class}, Stage: {pred_stage}")
