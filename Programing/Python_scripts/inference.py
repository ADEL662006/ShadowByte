from keras.models import load_model
import cv2
import numpy as np
from main import classes  # لو محتاج أسماء الكلاسات

model = load_model("fruit_classifier.h5")

img = cv2.imread("path_to_new_image.jpg")
img = cv2.resize(img, (100,100))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
class_index = np.argmax(prediction)
print("Predicted class:", classes[class_index])

cv2.imshow("Fruit", cv2.imread("path_to_new_image.jpg"))
cv2.waitKey(0)
cv2.destroyAllWindows()
