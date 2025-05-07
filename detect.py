# ASL Detection Project: Main Template

# 1. Import Libraries
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# 2. Dataset Preprocessing
data_dir = 'asl_alphabet_train'  # Path to your dataset with folders A, B, ..., Z
img_size = 200
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 3. Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Model Training
epochs = 5
model.fit(train_data, validation_data=val_data, epochs=epochs)

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)
model.save('model/asl_model.h5')

# Save labels
labels = list(train_data.class_indices.keys())
with open("model/labels.txt", "w") as f:
    for label in labels:
        f.write(f"{label}\n")

print("✅ Model and labels saved successfully.")
