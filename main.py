import cv2
import numpy as np
import os
import urllib.request
import time

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

    # Draw the bounding boxes on the frame
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

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

        # Process the frame with YOLO object detection
        start_time = time.time()
        processed_frame = detect_objects(frame)
        fps = 1 / (time.time() - start_time)
        print(f"FPS: {fps:.2f}")

        # Display the processed frame
        cv2.imshow("YOLO Object Detection", processed_frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
