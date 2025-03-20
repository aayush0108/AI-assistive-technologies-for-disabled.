import cv2
import pytesseract
import pyttsx3
import threading
import numpy as np
import time
import os
from ultralytics import YOLO
from gtts import gTTS
from playsound import playsound

# Set tesseract cmd path if it's not in your PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize pyttsx3 TTS engine
tts_engine = pyttsx3.init()

# Load YOLOv8 model using ultralytics package
model = YOLO("yolov8n.pt")  # Use a smaller version of YOLOv8 like yolov8n for testing

# Function to speak text using TTS
def speak_text(text):
    print(f"Speaking: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

# Function to detect and read text from the frame (OCR)
def detect_and_read_text(frame):
    # Convert frame to grayscale for better OCR performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use pytesseract to extract text from the image
    text = pytesseract.image_to_string(gray)
    
    if text.strip():  # If text is detected
        print(f"Detected text: {text}")
        
        # Use threading to avoid blocking the video frame capture while speaking
        threading.Thread(target=speak_text, args=(text,)).start()
    
    return text

# Function for YOLOv8 object detection
def detect_objects(frame):
    results = model(frame)  # Run inference on the frame
    detected_objects = []

    for result in results[0].boxes:  
        x1, y1, x2, y2 = result.xyxy[0].numpy()  
        confidence = result.conf[0].item()  
        class_id = int(result.cls[0].item()) 
        label = model.names[class_id] 

        if confidence > 0.3:  
            detected_objects.append(label) 

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, detected_objects

# Function to handle text-to-speech for detected objects
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

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Variables for frame skipping
frame_skip = 3  # Skip more frames (e.g., process every 3rd frame)
frame_counter = 0

# Process video feed
try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        frame_counter += 1
        if frame_counter % frame_skip != 0:
            continue

        # Start timing
        start_time = time.time()

        # Detect text from the frame (OCR)
        detected_text = detect_and_read_text(frame)

        # Detect objects using YOLOv8
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

        # Trigger TTS for detected objects in a separate thread
        if detected_objects:
            print("Triggering TTS for detected objects...")
            speak_objects_thread(detected_objects)

        # Display the processed frame with detected text and objects
        cv2.imshow("OCR and YOLOv8 Detection", processed_frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
