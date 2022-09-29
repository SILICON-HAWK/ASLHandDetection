import cv2 as cv
import numpy as np
import math

from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector

cap = cv.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
counter = 0

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "M"]
sequence = []
sentence = []

while True:
    success, img = cap.read()
    imgOut = img.copy()
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
            prediction, index = classifier.getPrediction(imgWhite)
            # print(prediction, index)

        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)

        sequence.insert(0, prediction)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            print("Printing the Letter", labels[index])

        # Visualize my Sentence
        if len(sentence) > 0:
            if labels[index] != sentence[-1]:
                sentence.append(labels[index])
        else:
            sentence.append(labels[index])

        if len(sentence) > 5:
            sentence = sentence[-5:]

        cv.rectangle(imgOut, (0, 0), (640, 40), (245, 117, 16), -1)
        cv.putText(imgOut, ' '.join(sentence), (3, 30), cv.FONT_HERSHEY_COMPLEX, 1, (255, 225, 255), 2, cv.LINE_AA)
        cv.putText(imgOut, labels[index], (x, y-20), cv.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

        # imgCrop_flip = cv.flip(imgWhite, 1)                                  # code to flip image
        cv.imshow("Image Cropped", imgWhite)

    cv.imshow("Image", imgOut)
    cv.waitKey(1)
