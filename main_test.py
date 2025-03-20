import cv2
import numpy as np
import os
import urllib.request
import time

# Import text-to-speech libraries
try:
    import pyttsx3
    tts_engine = pyttsx3.init()
except ImportError:
    from gtts import gTTS
    from playsound import playsound
    tts_engine = None
    print("pyttsx3 not found. Using gTTS as a fallback.")

# File paths
yolov3_weights_file = "./yolov3-tiny.weights"  # Using tiny YOLO for better performance
yolov3_cfg_file = "./yolov3-tiny.cfg"
coco_names_file = "./coco.names"

# URLs for YOLOv3 files
yolov3_tiny_weights_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
yolov3_tiny_cfg_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
coco_names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# Download YOLOv3 files if they are not already downloaded
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"{filename} downloaded successfully.")
    else:
        print(f"{filename} already exists.")

# Download necessary files
download_file(yolov3_tiny_weights_url, yolov3_weights_file)
download_file(yolov3_tiny_cfg_url, yolov3_cfg_file)
download_file(coco_names_url, coco_names_file)

# Load YOLO model
net = cv2.dnn.readNet(yolov3_weights_file, yolov3_cfg_file)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open(coco_names_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function for YOLO object detection
def detect_objects(frame):
    height, width, channels = frame.shape

    # Preprocess the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    detected_objects = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    # Draw the bounding boxes on the frame and collect detected objects
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            detected_objects.append(label)  # Add detected object to the list
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame, detected_objects

# Function to speak detected objects
def speak_objects(detected_objects):
    for obj in detected_objects:
        print(f"Speaking: {obj}")
        if tts_engine:  # Using pyttsx3
            tts_engine.say(obj)
            tts_engine.runAndWait()
        else:  # Fallback to gTTS
            tts = gTTS(text=obj, lang="en")
            tts.save("temp.mp3")
            playsound("temp.mp3")
            os.remove("temp.mp3")

# Open the webcam
cap = cv2.VideoCapture(0)  # Default webcam, change index if needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Use GPU acceleration if available
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("Using GPU acceleration")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Variables for frame skipping
frame_skip = 2  # Process 1 frame for every 2 captured frames
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

        # Process the frame with YOLO object detection
        processed_frame, detected_objects = detect_objects(frame)

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

        # Speak out detected objects
        speak_objects(detected_objects)

        # Display the processed frame
        cv2.imshow("YOLO Object Detection", processed_frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
# bicycle
# car
# motorcycle
# airplane
# bus
# train
# truck
# boat
# traffic light
# fire hydrant
# stop sign
# parking meter
# bench
# bird
# cat
# dog
# horse
# sheep
# cow
# elephant
# bear
# zebra
# giraffe
# backpack
# umbrella
# handbag
# tie
# suitcase
# frisbee
# skis
# snowboard
# sports ball
# kite
# baseball bat
# baseball glove
# skateboard
# surfboard
# tennis racket
# bottle
# wine glass
# cup
# fork
# knife
# spoon
# bowl
# banana
# apple
# sandwich
# orange
# broccoli
# carrot
# hot dog
# pizza
# donut
# cake
# chair
# couch
# potted plant
# bed
# dining table
# toilet
# TV
# laptop
# mouse
# remote
# keyboard
# cell phone
# microwave
# oven
# toaster
# sink
# refrigerator
# book
# clock
# vase
# scissors
# teddy bear
# hair drier
# toothbrushpip uninstall opencv-pythonpip uninstall opencv-python opencv-python-headless

  # This installs the headless version
