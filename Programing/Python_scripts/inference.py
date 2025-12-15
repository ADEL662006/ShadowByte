from keras.models import load_model
import cv2
import numpy as np
from main import classes  # لو محتاج أسماء الكلاسات

loss, acc = model.evaluate(test_generator)
print("Test Accuracy:", acc)
 
def predict_fruit(image_path):
    model = load_model("fruit_classifier_tl.h5")

    with open("classes.txt", "r", encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines()]

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("الصورة غير موجودة")

    original = img.copy()
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    prediction = model.predict(img, verbose=0)
    
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"Prediction: {classes[class_index]}")
    print(f"Confidence: {confidence:.2%}")

    return classes[class_index]

