import os
import shutil

SOURCE_DIR = "IDC_regular_ps50_idx5"
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

# Create folders
for folder in [TRAIN_DIR, TEST_DIR]:
    for label in ["0", "1"]:
        os.makedirs(os.path.join(folder, label), exist_ok=True)

def split_and_copy():
    for patient in os.listdir(SOURCE_DIR):
        patient_path = os.path.join(SOURCE_DIR, patient)

        if not os.path.isdir(patient_path):
            continue

        for label in ["0", "1"]:
            label_path = os.path.join(patient_path, label)

            if not os.path.exists(label_path):
                continue

            images = os.listdir(label_path)

            split = int(0.8 * len(images))

            for img in images[:split]:
                shutil.copy(
                    os.path.join(label_path, img),
                    os.path.join(TRAIN_DIR, label, img)
                )

            for img in images[split:]:
                shutil.copy(
                    os.path.join(label_path, img),
                    os.path.join(TEST_DIR, label, img)
                )

    print("Dataset ready!")

if __name__ == "__main__":
    split_and_copy()