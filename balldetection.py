import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

position = []
currentframe = 0


class detection:
    def __init__(self, frame, x, y):
        self.frame = frame
        self.x = x
        self.y = y

def nothing(x):
    pass
cap = cv2.VideoCapture("realtest.mp4")

cv2.namedWindow('Tracking')

cv2.createTrackbar('lower hue', 'Tracking', 0, 179 ,nothing)
cv2.createTrackbar('lower sat', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('lower val', 'Tracking', 0, 255, nothing)

cv2.createTrackbar('upper hue', 'Tracking', 179, 255, nothing)
cv2.createTrackbar('upper sat', 'Tracking', 255, 255, nothing)
cv2.createTrackbar('upper val', 'Tracking', 255, 255, nothing)

while True:
    
    ret, frame = cap.read()
    ret, frame_next = cap.read()
    if not ret:
        break
    #MOVEMENT DETECTION#
    diff = cv2.absdiff(frame, frame_next)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, movement_mask = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY) 


    #COLOR DETECTION#
    #frame = cv2.imread('backview.png')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_hue=cv2.getTrackbarPos('lower hue', 'Tracking')
    lower_sat=cv2.getTrackbarPos('lower sat', 'Tracking')
    lower_val=cv2.getTrackbarPos('lower val', 'Tracking')

    upper_hue=cv2.getTrackbarPos('upper hue', 'Tracking')
    upper_sat=cv2.getTrackbarPos('upper sat', 'Tracking')
    upper_val=cv2.getTrackbarPos('upper val', 'Tracking')
    
    lowerhsv = np.array([21, 26, 113])
    upperhsv = np.array([83, 177, 196])
    #lowerhsv = np.array([lower_hue, lower_sat, lower_val])
    #upperhsv = np.array([upper_hue, upper_sat, upper_val])
    color_mask = cv2.inRange(hsv, lowerhsv, upperhsv)\
    
    combined_mask = cv2.bitwise_and(movement_mask, color_mask)
    result = cv2.bitwise_and(frame, frame, mask = combined_mask)
    dilated = cv2.dilate(combined_mask, None, iterations = 3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 100:
             continue
        (x,y), radius=cv2.minEnclosingCircle(contour)
        center=(int(x), int(y))
        radius=int(radius)
        cv2.circle(result, center, radius, (0, 255, 0), 2)
        position.append(detection(currentframe, x, y))

    
    time.sleep(0.1)
    cv2.imshow("frame", frame)
    cv2.imshow("color mask", color_mask)
    cv2.imshow("movement mask", movement_mask)
    cv2.imshow("result", result)
    #input("press key to go to next frame")
    key = cv2.waitKey(1)
    if key == 27:
        break
    currentframe+=1


x_pos = [data.x for data in position]
y_pos = [data.y for data in position]
framenumber = [data.frame for data in position]

fig, (axs1, axs2) = plt.subplots(2)
axs1.plot(framenumber, x_pos, '+')
axs1.set_title("x pos")

axs2.plot(framenumber, y_pos, 'o')
axs2.set_title("y pos")

plt.show()
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
