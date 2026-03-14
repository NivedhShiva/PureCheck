# detect.py

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Load model
print("Loading model...")
model = Interpreter(model_path='food_model.tflite')
model.allocate_tensors()
input_details  = model.get_input_details()
output_details = model.get_output_details()
print("Model loaded ✅")

class_names = ['adulterated', 'pure']

# Try camera index 0 then 1
cam = None
for index in [0, 1, 2]:
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cam = cap
        print(f"Camera found on index {index} ✅")
        break
    cap.release()

if cam is None:
    print("❌ No camera detected — check USB webcam connection")
    exit()

# Skip first 5 frames
for _ in range(5):
    cam.read()

print("\nPlace sample in front of camera.")
print("Press ENTER to scan.")
input()

ret, frame = cam.read()
if not ret or frame is None:
    print("❌ Could not capture frame — check camera")
    cam.release()
    exit()

print("Frame captured ✅")

# Save captured frame so you can verify what camera saw
cv2.imwrite('last_capture.jpg', frame)
print("Saved last_capture.jpg — check what camera saw")

# Preprocess
img = cv2.resize(frame, (224, 224))
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)

# Run inference
model.set_tensor(input_details[0]['index'], img)
model.invoke()
output = model.get_tensor(output_details[0]['index'])

# Result
class_index = np.argmax(output[0])
confidence  = np.max(output[0]) * 100
result      = class_names[class_index]

print("\n" + "="*30)
if result == 'pure':
    print(f"✅ PURE — {confidence:.1f}% confidence")
else:
    print(f"❌ ADULTERATED — {confidence:.1f}% confidence")
print("="*30)

cam.release()