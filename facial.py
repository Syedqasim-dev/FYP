import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


data_dir = r"C:\Users\Qasim's Pc\Desktop\FYP Qasim\datasets\Video dataset\CK+48"

data = []
labels = []

IMG_SIZE = 224  # MobileNetV2 input size

for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    if os.path.isdir(label_dir):
        for file in os.listdir(label_dir):
            if file.endswith('.jpg') or file.endswith('.png'):
                file_path = os.path.join(label_dir, file)
                img = cv2.imread(file_path)
                if img is None:
                    continue  # skip unreadable files
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype(np.float32) / 255.0  # normalize
                data.append(img)
                labels.append(label)

data = np.array(data)
labels = np.array(labels)

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

#mobileNet model
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape = (224, 224, 3))

model.trainable = False

#add custom layers
x = model.output
x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)   
x = tf.keras.layers.Dropout(0.4)(x)
predictions = Dense(len(le.classes_), activation='softmax')(x)

model = Model(inputs=model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# train model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=10, validation_data=(X_test, y_test))

# === Evaluate ===
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=le.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))
# === Plot Training History ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# === Save Model ===
model.save("mobilenet_emotion_model.h5")

print("Model saved at:", os.path.abspath("mobilenet_emotion_model.h5"))
