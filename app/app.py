import os
import sys

# Make project root importable (so `src.*` imports work when running from app/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, render_template, request
from src.predict import predict_image

# Point static_folder to the root-level /static so logo.png is served correctly
ROOT = os.path.join(os.path.dirname(__file__), "..")
app = Flask(
    __name__,
    static_folder=os.path.join(ROOT, "static"),
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
)

UPLOAD_FOLDER = os.path.join(ROOT, "uploads")
STATIC_FOLDER = os.path.join(ROOT, "static")

os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    filepath = None
    gradcam = None
    error = None

    if request.method == "POST":
        if "file" not in request.files:
            error = "No file uploaded"
            return render_template("index.html", error=error)

        file = request.files["file"]

        if file.filename == "":
            error = "No selected file"
            return render_template("index.html", error=error)

        try:
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            result, gradcam_full_path = predict_image(filepath)

            # Strip leading "static/" or "static\" so url_for can serve it
            if gradcam_full_path:
                gradcam = os.path.basename(gradcam_full_path)
            else:
                gradcam = None

        except Exception as e:
            error = str(e)
            print("ERROR:", e)

    return render_template(
        "index.html",
        result=result,
        filename=filepath,
        gradcam=gradcam,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)