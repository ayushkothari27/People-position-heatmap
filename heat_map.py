import numpy as np
import cv2
import time

cap = cv2.VideoCapture("vtest.avi")

ret,frame = cap.read()
width = np.size(frame, 1)
height = np.size(frame, 0)
frame_size=(height, width)
res =  np.zeros((height, width), np.uint8)

fgbg = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=100,detectShadows=True)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

while True:
    ret, frame = cap.read()
    if not ret: break
    fgmask = fgbg.apply(frame, None, 0.01)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 2, 2)
    thresh = 2
    maxValue = 2
    ret, th1 = cv2.threshold(fgmask, thresh, maxValue, cv2.THRESH_BINARY)
    res = cv2.add(res, th1)
    color_image = cv2.applyColorMap(res, cv2.COLORMAP_JET)
    result_overlay = cv2.addWeighted(frame, 0.5, color_image, 40, 0)
    cv2.imshow('Final', result_overlay)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.imshow('Final', result_overlay)
time.sleep(100000)

cap.release()
cv2.destroyAllWindows()
