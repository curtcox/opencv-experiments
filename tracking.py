#!/usr/bin/env python

'''
Welcome to the Object Tracking Program!

Using real-time streaming video from your built-in webcam, this program:
  - Creates a bounding box around a moving object
  - Calculates the coordinates of the centroid of the object
  - Tracks the centroid of the object

Author:
  - Addison Sears-Collins
  - https://automaticaddison.com
'''

from __future__ import print_function # Python 2/3 compatibility
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library

# Project: Object Tracking
# Author: Addison Sears-Collins
# Website: https://automaticaddison.com
# Date created: 06/13/2020
# Python version: 3.7

def main():

    cap = cv2.VideoCapture(0) # Create a VideoCapture object

    # Create the background subtractor object. Use the last 700 video frames to build the background
    back_sub = cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=25, detectShadows=True)

    # Create kernel for morphological operation.
    # You can tweak the dimensions of the kernel e.g. instead of 20,20 you can try 30,30.
    kernel = np.ones((20,20),np.uint8)

    while(True):

        ret, frame = cap.read() # Capture frame-by-frame. This method returns True/False as well as the video frame.
        print(ret,frame)

        # Find the index of the largest contour and draw bounding box
        fg_mask_bb = create_fg_mask(back_sub, frame, kernel)
        contours, hierarchy = cv2.findContours(fg_mask_bb,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        areas = [cv2.contourArea(c) for c in contours]

        # If there are countours
        if len(areas) > 0:
            max_index = np.argmax(areas) # Find the largest moving object in the image
            cnt = contours[max_index]
            highlight_contour(cnt, frame)
        display_frame(frame)

    # Close down the video stream
    cap.release()
    cv2.destroyAllWindows()


def highlight_contour(cnt, frame):
    x, y, w, h = cv2.boundingRect(cnt)
    cx = x + int(w / 2)
    cy = y + int(h / 2)
    draw_bounding_box(frame, h, w, x, y)
    draw_circle_in_box(cx, cy, frame)
    print_coordinates(cx, cy, frame)

def draw_bounding_box(frame, h, w, x, y):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

def display_frame(frame):
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

def create_fg_mask(back_sub, frame, kernel):
    fg_mask = back_sub.apply(frame)                                  # Use every frame to calculate the foreground mask and update the background
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)     # Close dark gaps in foreground object using closing
    fg_mask = cv2.medianBlur(fg_mask, 5)                             # Remove salt and pepper noise with a median filter
    _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY) # Threshold the image to make it either black or white
    return fg_mask


# Print the centroid coordinates (we'll use the center of the bounding box) on the image
def print_coordinates(cx, cy, frame):
    text = "x: " + str(cx) + ", y: " + str(cy)
    cv2.putText(frame, text, (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# Draw circle in the center of the bounding box
def draw_circle_in_box(cx, cy, frame):
    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)


if __name__ == '__main__':
    print(__doc__)
    main()
