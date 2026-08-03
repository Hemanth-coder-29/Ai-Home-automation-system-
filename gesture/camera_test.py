import cv2

# Open the default webcam (0 = laptop webcam)
camera = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started successfully.")
print("Press 'Q' to exit.")

while True:

    # Read one frame from the webcam
    success, frame = camera.read()

    # Check if frame was captured
    if not success:
        print("Failed to capture frame.")
        break

    # Display the frame
    cv2.imshow("AI Smart Home Camera", frame)

    # Exit when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam
camera.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print("Camera closed successfully.")