# convert.py

import tensorflow as tf

print("Loading model...")
model = tf.keras.models.load_model('food_model.h5')
print("Done ✅")

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('food_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Done ✅")
print("Saved as food_model.tflite")