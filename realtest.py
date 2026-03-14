# realtest.py
import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('food_model.h5')
class_names = ['adulterated', 'pure']

def test(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    pred = model.predict(img)
    result     = class_names[np.argmax(pred[0])]
    confidence = np.max(pred[0]) * 100
    print(f"Result: {result} ({confidence:.1f}%)")

# test with a fresh photo not in your dataset
test('fresh_pure.jpg')