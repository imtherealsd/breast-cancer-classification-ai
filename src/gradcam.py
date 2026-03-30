import numpy as np
import cv2
import time
import os
import tensorflow as tf
import matplotlib.pyplot as plt

def generate_gradcam(model, img_array):

    # 🔥 Auto-detect last conv layer safely
    last_conv_layer = None
    for layer in reversed(model.layers):
        if "conv" in layer.name:
            last_conv_layer = layer.name
            break

    if last_conv_layer is None:
        raise ValueError("No Conv layer found in model")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.outputs[0]]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 🔥 Prevent division crash
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return heatmap.numpy()


def overlay_heatmap(img_path, heatmap):
    
    filename = f"gradcam_{int(time.time())}.jpg"
    output_path = os.path.join("static", filename)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (50, 50))

    heatmap = cv2.resize(heatmap, (50, 50))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = heatmap * 0.4 + img

    cv2.imwrite(output_path, superimposed)

    return output_path