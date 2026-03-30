import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from src.gradcam import generate_gradcam, overlay_heatmap

# Load model safely
MODEL_PATH = os.path.join("models", "cancer_model.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = load_model(MODEL_PATH)

# Force model initialization (important)
_ = model(np.zeros((1, 50, 50, 3)))

IMG_SIZE = 50


def predict_image(img_path):
    try:
        # -------------------------
        # 1. Load Image Safely
        # -------------------------
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError("Image not loaded properly. Check file path.")

        # Convert BGR → RGB (IMPORTANT for CNN consistency)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Normalize
        img = img / 255.0

        # Reshape for model
        img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

        # -------------------------
        # 2. Prediction
        # -------------------------
        prediction = model.predict(img, verbose=0)[0][0]

        confidence = prediction if prediction > 0.5 else 1 - prediction

        if prediction > 0.5:
            result = f"⚠️ Cancer Detected ({confidence*100:.2f}%)"
        else:
            result = f"✅ No Cancer ({confidence*100:.2f}%)"

        # -------------------------
        # 3. Grad-CAM (SAFE)
        # -------------------------
        gradcam_path = None

        try:
            heatmap = generate_gradcam(model, img)
            gradcam_path = overlay_heatmap(img_path, heatmap)
        except Exception as e:
            print("Grad-CAM Error:", e)
            gradcam_path = None  # don't crash app

        return result, gradcam_path

    except Exception as e:
        print("Prediction Error:", e)
        return f"Error: {str(e)}", None