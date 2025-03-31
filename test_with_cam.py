import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load the trained ASL model.
model = load_model('asl_model.h5')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for natural interaction.
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Convert the BGR image to RGB.
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands.
    results = hands.process(image_rgb)
    
    predicted_letter = ""
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Compute bounding box coordinates based on landmarks.
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
            x_min = int(min(x_coords) * width) - 100
            x_max = int(max(x_coords) * width) + 100
            y_min = int(min(y_coords) * height) - 100
            y_max = int(max(y_coords) * height) + 100

            # Ensure bounding box is within frame bounds.
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, width)
            y_max = min(y_max, height)
            
            # Draw the bounding box.
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Crop the detected hand region.
            hand_img = frame[y_min:y_max, x_min:x_max]
            # Preprocess the cropped image to match the model's input size.
            hand_img = cv2.resize(hand_img, (200, 200))
            hand_img = hand_img / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)
            
            # Make a prediction using the model.
            prediction = model.predict(hand_img)
            class_index = np.argmax(prediction)
            predicted_letter = chr(65 + class_index)  # Mapping index to letter (A=0, B=1, etc.)
            
            # Optionally, draw hand landmarks.
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Process only the first detected hand.
            break

    # Display the prediction on the frame.
    cv2.putText(frame, f'Predicted: {predicted_letter}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('ASL Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
