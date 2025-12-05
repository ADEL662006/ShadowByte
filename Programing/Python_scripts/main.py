import os
import cv2
import numpy as np
from tensorflow import keras
from keras.utils import to_categorical 


train_folder = r"C:\Users\Adel\Downloads\fruits-360-100x100-main\Training"
test_folder = r"C:\Users\Adel\Downloads\fruits-360-100x100-main\Test"

def load_data(folder):
    images = []
    labels = []
    classes = os.listdir(folder)
    for label, fruit in enumerate(classes):
        fruit_folder = os.path.join(folder, fruit)
        for file in os.listdir(fruit_folder):
            img_path = os.path.join(fruit_folder, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (100,100))
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels), classes

# تحميل الداتا
X_train, y_train, classes = load_data(train_folder)
X_test, y_test, _ = load_data(test_folder)

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode
y_train = to_categorical(y_train, num_classes=len(classes))
y_test = to_categorical(y_test, num_classes=len(classes))

print("Training set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)
print("Classes:", classes)
