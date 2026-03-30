import cv2
import numpy as np
import os

IMG_SIZE = 50

def load_data(data_dir):
    data = []
    labels = []

    for label in ["0", "1"]:
        path = os.path.join(data_dir, label)
        class_num = int(label)

        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                data.append(image)
                labels.append(class_num)
            except:
                pass

    data = np.array(data) / 255.0
    labels = np.array(labels)

    return data, labels