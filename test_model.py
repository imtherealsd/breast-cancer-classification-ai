from src.predict import predict_image

# Replace with your actual image path
image_path = "dataset/test/1/10254_idx5_x1851_y1301_class1.png"

result = predict_image(image_path)

print("Prediction:", result)
print("Using image:", image_path)