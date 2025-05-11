AI-Powered Wearable Assistive Device for the Visually Impaired
This project introduces an AI-based wearable assistive device specifically designed to support visually impaired individuals 
in their daily navigation and interaction with their environment. By integrating real-time object detection, live text recognition, 
and voice feedback into a compact wearable format, the system helps users better understand their surroundings. It aims to improve 
independence, accessibility, and safety using affordable hardware and open-source software.

The core features of this device include object detection using YOLOv8, optical character recognition (OCR) with PyTesseract, and 
text-to-speech conversion through Google Text-to-Speech (gTTS). These components work together to analyze the environment captured 
by a camera mounted on the wearable glasses. The system identifies objects, reads out texts such as signs or labels, and delivers 
instant audio feedback to the user, allowing them to make informed decisions in real time.

To install the project, users should first clone the repository and install the required dependencies using the provided requirements.txt file. 
The system can then be launched using the main Python script. Tesseract-OCR must be installed separately and added to the system path 
for text recognition to function properly.

The hardware requirements are minimal and cost-effective, comprising a Raspberry Pi (or any processing unit), a USB or Pi camera, 
and a speaker or earphones. The components can be mounted onto a lightweight glasses frame for ease of use. The system is designed 
to be hands-free, portable, and suitable for everyday tasks like shopping, commuting, or navigating unfamiliar environments.

Future versions of this device could include features like GPS-based navigation, facial recognition, haptic feedback, and 
multi-language voice support. This project is released under the MIT License, making it freely available for further research 
and development.

