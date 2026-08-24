import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import json
import queue
import threading
import sys
import os

# ==========================================
# 1. INITIALIZATION & SETUP
# ==========================================
print("Initializing Multi-Modal Smart Home System...")

# --- Gesture Recognition Setup ---
tf_model = tf.keras.models.load_model('model/gesture_model.keras')
# Map the AI classes directly to our hardware commands
gesture_commands = ['LIGHT_ON', 'LIGHT_OFF', 'FAN_ON', 'FAN_OFF'] 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Voice Recognition Setup ---
VOSK_MODEL_PATH = "model/vosk-model"
if not os.path.exists(VOSK_MODEL_PATH):
    print(f"Error: Vosk model not found at '{VOSK_MODEL_PATH}'.")
    sys.exit(1)

ALLOWED_COMMANDS = [
    "turn on light", "turn off light", "light on", "light off",
    "turn on fan", "turn off fan", "fan on", "fan off", "[unk]"
]
vosk_model = Model(VOSK_MODEL_PATH)
recognizer = KaldiRecognizer(vosk_model, 16000, json.dumps(ALLOWED_COMMANDS))
audio_queue = queue.Queue()

VOICE_COMMAND_MAP = {
    "light on": "LIGHT_ON", "turn on light": "LIGHT_ON",
    "light off": "LIGHT_OFF", "turn off light": "LIGHT_OFF",
    "fan on": "FAN_ON", "turn on fan": "FAN_ON",
    "fan off": "FAN_OFF", "turn off fan": "FAN_OFF"
}

# ==========================================
# 2. VOICE THREAD FUNCTION
# ==========================================
def audio_callback(indata, frames, time, status):
    if status:
        pass # Ignored to keep console clean
    audio_queue.put(bytes(indata))

def voice_listener():
    print("[VOICE] Listening for commands in background...")
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=audio_callback):
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").lower().strip()
                
                if text and text != "[unk]":
                    for phrase, action in VOICE_COMMAND_MAP.items():
                        if phrase in text:
                            print(f"\n---> [VOICE COMMAND DETECTED]: {action} <---")
                            break

# ==========================================
# 3. START THREADS
# ==========================================
# Start voice recognition in a background thread
voice_thread = threading.Thread(target=voice_listener, daemon=True)
voice_thread.start()

# ==========================================
# 4. GESTURE LOOP (MAIN THREAD)
# ==========================================
cap = cv2.VideoCapture(0)
print("[GESTURE] Camera started. Perform gestures to control appliances.")
print("Press 'q' to exit the entire system.\n")

# State tracker to prevent terminal spam
last_gesture = None 

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_gesture = "NONE"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            input_data = np.array(landmarks).reshape(1, -1)
            prediction = tf_model.predict(input_data, verbose=0)
            class_id = np.argmax(prediction)
            confidence = prediction[0][class_id] * 100

            if confidence > 85: # Strict confidence threshold
                current_gesture = gesture_commands[class_id]
                
                # Display on camera window
                label_text = f"CMD: {current_gesture} ({confidence:.0f}%)"
                cv2.putText(frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Print gesture command only if it changes
    if current_gesture != "NONE" and current_gesture != last_gesture:
        print(f"\n---> [GESTURE COMMAND DETECTED]: {current_gesture} <---")
        last_gesture = current_gesture

    cv2.imshow("Multi-Modal Smart Home Hub", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("System shut down safely.")