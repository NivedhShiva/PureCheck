# detect.py — Windows version with live preview

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Load model
print("Loading model...")
interpreter = Interpreter(model_path='food_model.tflite')
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded ✅")

class_names = ['adulterated', 'pure']
result     = "Waiting..."
confidence = 0.0

# Start camera
cam = cv2.VideoCapture(1)
if not cam.isOpened():
    cam = cv2.VideoCapture(1)
if not cam.isOpened():
    print("❌ No camera found")
    exit()

print("Camera ready ✅")
print("Press SPACE to scan, Q to quit")

frame_count = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("❌ Camera read failed")
        break

    frame_count += 1

    # Run inference every 10th frame
    if frame_count % 10 == 0:
        img = cv2.resize(frame, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        class_index = np.argmax(output[0])
        confidence  = np.max(output[0]) * 100
        result      = class_names[class_index]

    # Draw result on live frame
    if result == 'pure':
        color = (0, 255, 0)      # green
        label = f"PURE - {confidence:.1f}%"
    elif result == 'adulterated':
        color = (0, 0, 255)      # red
        label = f"ADULTERATED - {confidence:.1f}%"
    else:
        color = (255, 255, 255)  # white
        label = result

    # Draw background box behind text
    cv2.rectangle(frame, (0, 0), (400, 60), (0, 0, 0), -1)

    # Draw result text
    cv2.putText(
        frame, label,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2, color, 2
    )

    # Show live feed
    cv2.imshow('PureCheck — Food Adulteration Detector', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()