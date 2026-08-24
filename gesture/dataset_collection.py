import cv2
import mediapipe as mp
import csv
import os

# Define the dataset file path
CSV_FILE = 'dataset/gesture_dataset.csv'

# Ensure the dataset folder exists
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Initialize the CSV file with headers if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Create headers: label, x0, y0, x1, y1... up to x20, y20
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}'])
        writer.writerow(header)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("--- DATASET COLLECTION STARTED ---")
print("Hold '0' for Open Palm (Light ON)")
print("Hold '1' for Fist (Light OFF)")
print("Hold '2' for Victory (Fan ON)")
print("Hold '3' for Thumbs Up (Fan OFF)")
print("Press 'q' to exit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame so it acts like a mirror
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    key = cv2.waitKey(1) & 0xFF

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # If a valid number key is pressed, record the data
            if ord('0') <= key <= ord('3'):
                label = chr(key)
                landmark_list = [label]
                
                # Extract x and y coordinates for all 21 points
                for lm in hand_landmarks.landmark:
                    landmark_list.extend([lm.x, lm.y])
                
                # Save the row to the CSV
                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(landmark_list)
                
                print(f"Recorded frame for gesture: {label}")

    cv2.imshow("Dataset Collection", frame)
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera closed. Data collection stopped.")