import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# 1. Load trained gesture model
model = tf.keras.models.load_model('model/gesture_model.keras')
class_names = ['Open Palm (Light ON)', 'Fist (Light OFF)', 'Victory (Fan ON)', 'Thumbs Up (Fan OFF)']

# 2. Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 3. Open Webcam
cap = cv2.VideoCapture(0)

print("--- REAL-TIME GESTURE PREDICTION ---")
print("Press 'q' to exit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 42 (x, y) coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            # Reshape input to (1, 42) for inference
            input_data = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(input_data, verbose=0)
            class_id = np.argmax(prediction)
            confidence = prediction[0][class_id] * 100

            # Display predicted gesture and confidence on camera frame
            label_text = f"{class_names[class_id]} ({confidence:.1f}%)"
            cv2.putText(frame, label_text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("AI Smart Home - Real-Time Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()