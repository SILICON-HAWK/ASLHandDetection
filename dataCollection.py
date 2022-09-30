import cv2 as cv
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector

cap = cv.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = 'L'
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-offset:y+h + offset, x - offset:x+w+offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = imgResize

        imgCrop_flip = cv.flip(imgWhite, 1)                                  # code to flip image
        cv.imshow("Image Cropped", imgCrop_flip)

    cv.imshow("Image", img)
    key = cv.waitKey(1)
    if key == ord("s"):
        counter += 1
        # cv.im write(f'{folder}/Image_{time.time()}.jpg', imgCrop_flip)
        cv.imwrite(f'D:/UNIVERSITY/PROJECT EXIBITION SEM3/ASLsemProject/ASLHandDetection/venv/Data/{folder}/Image_{time.time()}.jpg', imgCrop_flip)
        print(counter)
