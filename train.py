# train.py

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

print("Step 1: Loading pretrained model...")
base = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base.trainable = False
print("Done ✅")

print("Step 2: Building classifier...")
model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("Done ✅")

print("Step 3: Loading dataset...")
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

train_data = datagen.flow_from_directory(
    'dataset/',
    target_size=(224, 224),
    batch_size=8,
    subset='training'
)

val_data = datagen.flow_from_directory(
    'dataset/',
    target_size=(224, 224),
    batch_size=8,
    subset='validation'
)
print("Done ✅")
print(f"Classes: {train_data.class_indices}")

print("Step 4: Training...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

model.save('food_model_1.h5')
print("\n✅ Model saved as food_model.h5")
print(f"Best accuracy: {max(history.history['val_accuracy'])*100:.1f}%")

plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')
print("Accuracy graph saved ✅")