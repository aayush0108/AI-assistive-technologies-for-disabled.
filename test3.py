import cv2
import numpy as np
import time
import threading
import os
from ultralytics import YOLO

# Import text-to-speech libraries
try:
    import pyttsx3
    tts_engine = pyttsx3.init()
except ImportError:
    from gtts import gTTS
    from playsound import playsound
    tts_engine = None
    print("pyttsx3 not found. Using gTTS as a fallback.")

# Load YOLOv8 model using ultralytics package
model = YOLO("yolov8n.pt")  # Use a smaller version of YOLOv8 like yolov8n for testing

# Function for YOLOv8 object detection
def detect_objects(frame):
    results = model(frame)  # Run inference on the frame
    detected_objects = []

    # Loop through results and process the detected objects
    for result in results[0].boxes:  # `results[0]` corresponds to the first frame (if batch is used)
        x1, y1, x2, y2 = result.xyxy[0].numpy()  # Get the coordinates (top-left and bottom-right)
        confidence = result.conf[0].item()  # Get the confidence score
        class_id = int(result.cls[0].item())  # Get the class id
        label = model.names[class_id]  # Get the class name

        if confidence > 0.3:  # Confidence threshold
            detected_objects.append(label)  # Collect detected object

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, detected_objects

# Function to handle text-to-speech in a separate thread
def speak_objects_thread(detected_objects):
    def speak():
        for obj in detected_objects:
            print(f"Speaking: {obj}")  # Debugging output
            if tts_engine:  # Using pyttsx3
                tts_engine.say(obj)
                tts_engine.runAndWait()
            else:  # Fallback to gTTS
                tts = gTTS(text=obj, lang="en")
                tts.save("temp.mp3")
                playsound("temp.mp3")
                os.remove("temp.mp3")

    # Start the TTS in a separate thread to avoid blocking
    threading.Thread(target=speak).start()

# Open the webcam
cap = cv2.VideoCapture(0)  # Default webcam, change index if needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduced width for better performance
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Reduced height for better performance

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Variables for frame skipping
frame_skip = 3  # Skip more frames (e.g., process every 3rd frame)
frame_counter = 0

# Start the webcam feed
try:
    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            break

        frame_counter += 1
        if frame_counter % frame_skip != 0:
            continue

        # Start timing
        start_time = time.time()

        # Process the frame with YOLOv8 object detection
        processed_frame, detected_objects = detect_objects(frame)

        # Debugging: Print detected objects
        print(f"Detected Objects: {detected_objects}")

        # Calculate FPS
        fps = 1 / (time.time() - start_time)

        # Add FPS text overlay on the frame
        cv2.putText(
            processed_frame, 
            f"FPS: {fps:.2f}",  
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
            (0, 255, 0), 
            2,  
        )

        # Speak out detected objects in a separate thread
        if detected_objects:
            print("Triggering TTS for detected objects...")
            speak_objects_thread(detected_objects)

        # Display the processed frame
        cv2.imshow("YOLOv8 Object Detection", processed_frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()





#