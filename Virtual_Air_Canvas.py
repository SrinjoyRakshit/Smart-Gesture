import cv2
import numpy as np
from collections import deque

# Function called by trackbar, it does nothing here but we need it for trackbar
def adjust_color(dummy=None):
   pass

# Initialize the window and trackbars for color selection
cv2.namedWindow("Color Adjustments")
cv2.createTrackbar("High Hue", "Color Adjustments", 154, 181, adjust_color)
cv2.createTrackbar("High Saturation", "Color Adjustments", 255, 255, adjust_color)
cv2.createTrackbar("High Value", "Color Adjustments", 255, 255, adjust_color)
cv2.createTrackbar("Low Hue", "Color Adjustments", 64, 180, adjust_color)
cv2.createTrackbar("Low Saturation", "Color Adjustments", 72, 255, adjust_color)
cv2.createTrackbar("Low Value", "Color Adjustments", 49, 255, adjust_color)

# Indexes for current position in color deques
blue_idx, green_idx, red_idx, yellow_idx = 0, 0, 0, 0

# Initialize deques to store different colored points
blue_trails = [deque(maxlen=1024)]
green_trails = [deque(maxlen=1024)]
red_trails = [deque(maxlen=1024)]
yellow_trails = [deque(maxlen=1024)]

# Kernel for morphological operations
dilation_kernel = np.ones((5,5),np.uint8)

# Color definitions in BGR
palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
palette_idx = 0

# Setup the drawing canvas
sketch_pad = np.full((470,637,3), 255)
sketch_pad = cv2.rectangle(sketch_pad, (40,1), (140,65), (0,0,0), 2)
for i, col in enumerate(palette):
    sketch_pad = cv2.rectangle(sketch_pad, (160 + i*115,1), (255 + i*115,65), col, -1)

# Label the buttons
btn_labels = ["CLEAR", "BLUE", "GREEN", "RED", "YELLOW"]
for i, text in enumerate(btn_labels):
    cv2.putText(sketch_pad, text, (49 + i*115, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Main loop
while True:
    # Capture the image from the webcam
    ret, captured_image = video_capture.read()
    captured_image = cv2.flip(captured_image, 1)
    converted_hsv = cv2.cvtColor(captured_image, cv2.COLOR_BGR2HSV)

    # Get the current positions of the trackbars
    high_hue = cv2.getTrackbarPos("High Hue", "Color Adjustments")
    high_saturation = cv2.getTrackbarPos("High Saturation", "Color Adjustments")
    high_value = cv2.getTrackbarPos("High Value", "Color Adjustments")
    low_hue = cv2.getTrackbarPos("Low Hue", "Color Adjustments")
    low_saturation = cv2.getTrackbarPos("Low Saturation", "Color Adjustments")
    low_value = cv2.getTrackbarPos("Low Value", "Color Adjustments")

    # Create the HSV color bounds
    hsv_high = np.array([high_hue, high_saturation, high_value])
    hsv_low = np.array([low_hue, low_saturation, low_value])

    # Create the mask and perform morphological operations
    detected_colors = cv2.inRange(converted_hsv, hsv_low, hsv_high)
    detected_colors = cv2.erode(detected_colors, dilation_kernel, iterations=1)
    detected_colors = cv2.morphologyEx(detected_colors, cv2.MORPH_OPEN, dilation_kernel)
    detected_colors = cv2.dilate(detected_colors, dilation_kernel, iterations=1)

    # Find contours in the mask
    contours, _ = cv2.findContours(detected_colors.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pointer = None

    # If contours are found, find the largest one
    if len(contours) > 0:
        largest = sorted(contours, key = cv2.contourArea, reverse = True)[0]
        ((ctr_x, ctr_y), radius) = cv2.minEnclosingCircle(largest)
        cv2.circle(captured_image, (int(ctr_x), int(ctr_y)), int(radius), (0, 255, 255), 3)
        moments = cv2.moments(largest)
        pointer_center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))

        # Check if the user clicked any button
        if pointer[1] <= 65:
            if 40 <= pointer[0] <= 140: # Clear Button
                blue_trails, green_trails, red_trails, yellow_trails = [deque(maxlen=512)], [deque(maxlen=512)], [deque(maxlen=512)], [deque(maxlen=512)]
                blue_idx, green_idx, red_idx, yellow_idx = 0, 0, 0, 0
                sketch_pad[67:,:,:] = 255
            else:
                for i in range(len(palette)):
                    if 160 + i*115 <= pointer[0] <= 255 + i*115:
                        palette_idx = i
                        break
        else:
            if palette_idx == 0:
                blue_trails[blue_idx].appendleft(pointer)
            elif palette_idx == 1:
                green_trails[green_idx].appendleft(pointer)
            elif palette_idx == 2:
                red_trails[red_idx].appendleft(pointer)
            elif palette_idx == 3:
                yellow_trails[yellow_idx].appendleft(pointer)
    else:
        blue_trails.append(deque(maxlen=512))
        blue_idx += 1
        green_trails.append(deque(maxlen=512))
        green_idx += 1
        red_trails.append(deque(maxlen=512))
        red_idx += 1
        yellow_trails.append(deque(maxlen=512))
        yellow_idx += 1

    # Draw the lines on the canvas and image
    trails = [blue_trails, green_trails, red_trails, yellow_trails]
    for idx, trail in enumerate(trails):
        for d_idx in range(len(trail)):
            for p_idx in range(1, len(trail[d_idx])):
                if trail[d_idx][p_idx - 1] is None or trail[d_idx][p_idx] is None:
                    continue
                cv2.line(captured_image, trail[d_idx][p_idx - 1], trail[d_idx][p_idx], palette[idx], 2)
                cv2.line(sketch_pad, trail[d_idx][p_idx - 1], trail[d_idx][p_idx], palette[idx], 2)

    # Display the windows
    cv2.imshow("Color Tracking", captured_image)
    cv2.imshow("Drawing Canvas", sketch_pad)
    cv2.imshow("Color Detection", detected_colors)

    # Check for key presses
key_pressed = cv2.waitKey(1)

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
