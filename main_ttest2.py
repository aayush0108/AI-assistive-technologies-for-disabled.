import cv2
import pytesseract
import pyttsx3
import threading
import time
import os
from ultralytics import YOLO
from gtts import gTTS
from playsound import playsound

# Set tesseract cmd path if it's not in your PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize pyttsx3 TTS engine
tts_engine = pyttsx3.init()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

def speak_text(text):
    print(f"Speaking: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

def detect_and_read_text(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    
    if text.strip():
        threading.Thread(target=speak_text, args=(text,)).start()
    
    return text

def detect_objects(frame):
    results = model(frame)
    detected_objects = []

    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0].numpy()
        confidence = result.conf[0].item()
        class_id = int(result.cls[0].item())
        label = model.names[class_id]

        if confidence > 0.3:
            detected_objects.append(label)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, detected_objects

def speak_objects_thread(detected_objects):
    def speak():
        for obj in detected_objects:
            print(f"Speaking: {obj}")
            if tts_engine:
                tts_engine.say(obj)
                tts_engine.runAndWait()
            else:
                tts = gTTS(text=obj, lang="en")
                tts.save("temp.mp3")
                playsound("temp.mp3")
                os.remove("temp.mp3")

    threading.Thread(target=speak).start()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

frame_skip = 3
frame_counter = 0

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        frame_counter += 1
        if frame_counter % frame_skip != 0:
            continue

        start_time = time.time()

        detected_text = detect_and_read_text(frame)
        processed_frame, detected_objects = detect_objects(frame)

        print(f"Detected Objects: {detected_objects}")

        fps = 1 / (time.time() - start_time)
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if detected_objects:
            print("Triggering TTS for detected objects...")
            speak_objects_thread(detected_objects)

        cv2.imshow("OCR and YOLOv8 Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
