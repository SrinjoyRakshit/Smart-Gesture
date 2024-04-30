import cv2 as camera
import numpy as np 
import pyautogui
import imutils
import time
import json

# Initialize the camera
capture = camera.VideoCapture(0)

# Define the keyboard layout
keyboard_layout = []
number_keys = ["1","2","3","4","5","6","7","8","9","0"]
top_row_keys = ["Q","W","E","R","T","Y","U","I","O","P"]
second_row_keys = ["A","S","D","F","G","H","J","K","L"]
third_row_keys = ["Z","X","C","V","B","N","M"]
control_keys = ["space","enter","backspace","shift"]
navigation_keys = ["left","up","down","right"]
volume_keys = ["volumeup","volumedown","volumemute"]

# Function to add keys to the layout
def add_keys_to_layout(keys, start_x, start_y, width, height, offset_x):
    for index, key in enumerate(keys):
        key_data = {
            "x": start_x + index * offset_x,
            "y": start_y,
            "w": width,
            "h": height,
            "value": key
        }
        keyboard_layout.append(key_data)

# Add all keys to the layout
add_keys_to_layout(number_keys, 10, 20, 100, 80, 100)
add_keys_to_layout(top_row_keys, 10, 100, 100, 80, 100)
add_keys_to_layout(second_row_keys, 110, 180, 100, 80, 100)
add_keys_to_layout(third_row_keys, 210, 260, 100, 80, 100)

# Add control and navigation keys separately due to different widths and positions
keyboard_layout.extend([
    {"x": 110, "y": 340, "w": 200, "h": 80, "value": control_keys[0]},
    {"x": 310, "y": 340, "w": 200, "h": 80, "value": control_keys[1]},
    {"x": 510, "y": 340, "w": 250, "h": 80, "value": control_keys[2]},
    {"x": 760, "y": 340, "w": 200, "h": 80, "value": control_keys[3]},
    {"x": 110, "y": 420, "w": 200, "h": 80, "value": navigation_keys[0]},
    {"x": 310, "y": 420, "w": 200, "h": 80, "value": navigation_keys[1]},
    {"x": 510, "y": 420, "w": 200, "h": 80, "value": navigation_keys[2]},
    {"x": 710, "y": 420, "w": 200, "h": 80, "value": navigation_keys[3]},
    {"x": 10, "y": 500, "w": 200, "h": 80, "value": volume_keys[0]},
    {"x": 210, "y": 500, "w": 200, "h": 80, "value": volume_keys[1]},
    {"x": 410, "y": 500, "w": 200, "h": 80, "value": volume_keys[2]}
])

# Convert the layout to JSON
layout_json = json.dumps(keyboard_layout)
layout_data = json.loads(layout_json)

# Main loop for the virtual keyboard
while True:
    ret, frame = capture.read()
    frame = camera.GaussianBlur(frame, (6,6), 0)
    frame = imutils.resize(frame, width=1035, height=720)

    # Draw the keyboard on the frame
    for key in layout_data:
        camera.rectangle(frame, (key["x"], key["y"]), (key["x"] + key["w"], key["y"] + key["h"]), (0,255,255), 3)
        camera.putText(frame, key["value"], (key["x"] + key["w"]//2, key["y"] + key["h"]//2), camera.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3, camera.LINE_AA)

    # Process the gesture recognition
    hsv_frame = camera.cvtColor(frame, camera.COLOR_BGR2HSV)
    mask = camera.inRange(hsv_frame, np.array([65,60,60]), np.array([80,255,255]))
    mask = camera.morphologyEx(mask, camera.MORPH_CLOSE, np.ones((20,20)))

    contours, _ = camera.findContours(mask.copy(), camera.RETR_EXTERNAL, camera.CHAIN_APPROX_SIMPLE)
    if len(contours) == 1:
        x, y, w, h = camera.boundingRect(contours[0])
        center_x = round(x + w/2)
        center_y = round(y + h/2)
        camera.circle(frame, (center_x, center_y), 20, (0,0,255), 2)

        # Check if the gesture is within any key's area
        for key in layout_data:
            if center_x >= key["x"] and center_x <= key["x"] + key["w"] and center_y >= key["y"] and center_y <= key["y"] + key["h"]:
                pyautogui.press(key["value"])
                break

    # Display the frame
    camera.imshow("Virtual Keyboard", frame)
    camera.waitKey(12)
