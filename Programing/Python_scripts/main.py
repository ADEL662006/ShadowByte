import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import cv2
import numpy as np
import os

train_folder = r"C:\Users\Adel\Downloads\fruits-360-100x100-main\Training"
test_folder = r"C:\Users\Adel\Downloads\fruits-360-100x100-main\Test"

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_folder,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

classes = list(train_generator.class_indices.keys())
num_classes = len(classes)

with open("classes.txt", "w", encoding="utf-8") as f:
    for c in classes:
        f.write(c + "\n")

print("Number of classes:", num_classes)
print(classes[:15])
