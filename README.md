# Breast Cancer Detection AI

A professional, end-to-end medical AI application that classifies breast cancer from histopathology images using Deep Learning. Designed with a premium Glassmorphic UI, this tool not only predicts the likelihood of cancer but also emphasizes **Explainable AI (XAI)** by generating **Grad-CAM (Gradient-weighted Class Activation Mapping)** visualizations to highlight the regions of interest the model used to make its decision.

## 🌟 Key Features

- **Deep Learning Model**: Uses a trained Convolutional Neural Network (CNN) to classify breast tissue images (IDC - Invasive Ductal Carcinoma).
- **Explainable AI (XAI)**: Integrates Grad-CAM to visualize model attention, crucial for medical diagnostic tools to ensure transparency and clinical trustworthiness.
- **Premium User Interface**: Features a modern, sleek, dark-themed glassmorphism design with responsive elements, animated scan lines, and clinical data gauges.
- **Robust Backend**: Built on Flask for secure file handling, image processing, and efficient prediction serving.

## 🛠️ Technologies Used

- **Backend**: Python, Flask, Werkzeug
- **Machine Learning**: TensorFlow / Keras, NumPy, OpenCV
- **Explainable AI**: Grad-CAM
- **Frontend**: HTML5, CSS3, Vanilla JS (Glassmorphism, custom CSS animations)

## 📁 Project Structure

```text
├── app/
│   ├── app.py              # Main Flask application
│   └── templates/
│       └── index.html      # Glassmorphic UI frontend
├── src/
│   ├── gradcam.py          # Grad-CAM visualization logic
│   ├── model.py            # CNN Model architecture
│   ├── predict.py          # Prediction pipeline
│   ├── preprocess.py       # Image preprocessing
│   └── train.py            # Model training script
├── models/
│   └── cancer_model.h5     # Pretrained weights
├── static/
│   └── (Generated Assets)  # Stores Grad-CAM output dynamically
├── setup_dataset.py        # Dataset downloading / initialization
└── requirements.txt        # Python dependencies
```

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/imtherealsd/breast-cancer-classification-ai.git
cd breast-cancer-classification-ai
```

### 2. Set up a virtual environment (Recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app/app.py
```
Open `http://localhost:5000` in your web browser.

## 📊 Dataset

This project uses the historical IDC (Invasive Ductal Carcinoma) Breast Cancer dataset from Kaggle.
*(Dataset and large generated assets are ignored in `.gitignore` to maintain a lightweight repository).*

## 📄 License

MIT License
