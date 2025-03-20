import cv2
import pytesseract
import pyttsx3
import threading

# Set tesseract cmd path if it's not in your PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Initialize pyttsx3 TTS engine
tts_engine = pyttsx3.init()

# Function to speak text using TTS
def speak_text(text):
    print(f"Speaking: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

# Function to detect and read text from the frame
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

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Process video feed
try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Detect and read text from the frame
        detected_text = detect_and_read_text(frame)
        
        # Display the detected text on the video frame
        cv2.putText(frame, detected_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the video frame
        cv2.imshow('Live Text to Speech', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
