from main import X_train, y_train, X_test, y_test, classes
from tensorflow import keras
from keras.utils import to_categorical 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# تعريف الموديل
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

# compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# تدريب الموديل
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

model.save("fruit_classifier.h5")
