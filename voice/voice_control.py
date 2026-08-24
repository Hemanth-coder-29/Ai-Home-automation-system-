import os
import sys
import json
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# 1. Verify Model Path
MODEL_PATH = "model/vosk-model"
if not os.path.exists(MODEL_PATH):
    print(f"Error: Vosk model not found at '{MODEL_PATH}'. Download and extract it first.")
    sys.exit(1)

# 2. Define Strict Command Grammar
# Restricting Vosk to ONLY these phrases forces 100% accurate phonetic matching
ALLOWED_COMMANDS = [
    "turn on light",
    "turn off light",
    "light on",
    "light off",
    "turn on fan",
    "turn off fan",
    "fan on",
    "fan off",
    "[unk]"  # Handles unknown background noise
]

# Convert list to JSON string format required by Vosk
GRAMMAR_JSON = json.dumps(ALLOWED_COMMANDS)

# 3. Initialize Vosk Model with Grammar Constraint
print("Loading offline voice model with constrained grammar...")
model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, 16000, GRAMMAR_JSON)

# Thread-safe audio buffer queue
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """Callback function to grab audio blocks from microphone."""
    if status:
        print(f"Audio Status Warning: {status}", file=sys.stderr)
    audio_queue.put(bytes(indata))

# 4. Command Mapping Setup
COMMAND_MAP = {
    "light on": "LIGHT_ON",
    "turn on light": "LIGHT_ON",
    "light off": "LIGHT_OFF",
    "turn off light": "LIGHT_OFF",
    "fan on": "FAN_ON",
    "turn on fan": "FAN_ON",
    "fan off": "FAN_OFF",
    "turn off fan": "FAN_OFF"
}

print("\n=== OFFLINE VOICE CONTROL STARTED ===")
print("Recognized phrases: 'turn on light', 'turn off light', 'fan on', 'fan off'")
print("Press Ctrl+C in terminal to stop.\n")

# 5. Open Live Microphone Stream
try:
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=audio_callback):
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").lower().strip()
                
                if text and text != "[unk]":
                    print(f"Heard: '{text}'")
                    
                    # Check for recognized smart home commands
                    command_found = False
                    for phrase, action in COMMAND_MAP.items():
                        if phrase in text:
                            print(f"--> MATCHED COMMAND: [{action}]\n")
                            command_found = True
                            break
                    
                    if not command_found:
                        print("--> (Unrecognized phrase)\n")

except KeyboardInterrupt:
    print("\nVoice control stopped safely.")
except Exception as e:
    print(f"\nError initializing audio stream: {e}")