import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from math import sqrt, hypot

# Constants for gesture recognition
PINCH_THRESHOLD = 0.05
V_GESTURE_THRESHOLD = 1.7
DEPTH_DIFF_THRESHOLD = 0.1
DOUBLE_CLICK_INTERVAL = 0.3  # Time interval for double-click detection
SCROLL_THRESHOLD = 0.1  # Threshold for scrolling gesture
DRAG_THRESHOLD = 0.1  # Threshold for drag gesture
RIGHT_CLICK_THRESHOLD = 0.1  # Threshold for right-click gesture

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return hypot(point1.x - point2.x, point1.y - point2.y)

# Function to calculate the depth difference between two points
def calculate_depth_difference(point1, point2):
    return abs(point1.z - point2.z)

# Function to convert hand coordinates to screen coordinates
def convert_to_screen_coordinates(x, y, image_width, image_height):
    screen_width, screen_height = pyautogui.size()
    return (x * screen_width / image_width, y * screen_height / image_height)

# Main function to process video frames and interpret gestures
def process_video():
    cap = cv2.VideoCapture(0)
    previous_click_time = 0
    is_dragging = False

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and detect hands
        results = hands.process(image)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get fingertip points
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                # Calculate distances and depth differences for gestures
                pinch_distance = calculate_distance(index_tip, thumb_tip)
                depth_difference = calculate_depth_difference(index_tip, middle_tip)

                # Convert hand coordinates to screen coordinates
                screen_x, screen_y = convert_to_screen_coordinates(index_tip.x, index_tip.y, cap.get(3), cap.get(4))

                # Perform actions based on gestures
                if pinch_distance < PINCH_THRESHOLD:
                    current_click_time = cv2.getTickCount()
                    if (current_click_time - previous_click_time) / cv2.getTickFrequency() < DOUBLE_CLICK_INTERVAL:
                        pyautogui.doubleClick()
                    else:
                        pyautogui.click()
                    previous_click_time = current_click_time
                elif pinch_distance > V_GESTURE_THRESHOLD:
                    pyautogui.scroll(100)  # Scroll up
                elif depth_difference < DEPTH_DIFF_THRESHOLD:
                    if not is_dragging:
                        pyautogui.mouseDown()
                        is_dragging = True
                    else:
                        pyautogui.moveTo(screen_x, screen_y)
                else:
                    if is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False
                    pyautogui.moveTo(screen_x, screen_y)

                # Additional feature: Right-click
                if pinch_distance > RIGHT_CLICK_THRESHOLD:
                    pyautogui.rightClick()

                # Additional feature: Volume control
                # Assuming a horizontal swipe gesture increases/decreases volume
                if pinch_distance > SCROLL_THRESHOLD:
                    # Determine the direction of the swipe
                    if index_tip.x > thumb_tip.x:  # Swipe to the right
                        pyautogui.press('volumeup')
                    else:  # Swipe to the left
                        pyautogui.press('volumedown')

        # Show the image
        cv2.imshow('Virtual Mouse', image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_video()
