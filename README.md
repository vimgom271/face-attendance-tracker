Face Attendance Tracker
Overview
This project is a real-time attendance tracking system built on an NVIDIA Jetson AGX Orin. It uses YOLOv8 to detect and track people through a live camera feed, automatically incrementing or decrementing an attendance count based on movement across a virtual boundary line drawn down the center of the frame.
How It Works
A vertical line is drawn at the center of the video frame. When a tracked person crosses from left to right, the attendance count goes up by one. When they cross from right to left, it goes down by one. Each person is assigned a unique tracking ID so the system never double-counts the same individual.
Features

Real-time person detection and tracking using YOLOv8
Virtual boundary line for entry and exit detection
Live attendance count displayed on screen
Persistent tracking IDs across frames
Runs fully on-device with no internet dependency

Tech Stack

Hardware: NVIDIA Jetson AGX Orin
Model: YOLOv8 (person detection and tracking)
Library: Ultralytics, OpenCV
Resolution: 1280x720
Language: Python 3.10

How To Run

Clone the repository
Install dependencies with pip install -r requirements.txt
Place your YOLOv8 model file in the project folder
Run python3 attendance.py
Press q to quit

Author
Vimal Gomathisankar
