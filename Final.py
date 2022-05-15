import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
from PIL import Image
import cv2
import imutils
import time
import mediapipe as mp
import pandas as pd
import keyboard
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# keypress
import pyautogui

bg = None

# Setup Core Audio Windows Library
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# volume.GetMute()
# volume.GetMasterVolumeLevel()
volumeRange = volume.GetVolumeRange()

minVolume = volumeRange[0]
maxVolume = volumeRange[1]

# time to save 
gesture_count = 0
gesture_number = -1

def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image to get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

class HandDetector():
    def __init__(self, mode = False, maxHands = 2, detectionConfidence = 0.5, trackConfidence = 0.5):
        # Default Values
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        # Draws Hand
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.detectionConfidence, self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.fingerTipID = [4, 8, 12, 16, 20]

    def locateHands(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLandMarks in self.results.multi_hand_landmarks:
                if draw == True:
                    self.mpDraw.draw_landmarks(img, handLandMarks, self.mpHands.HAND_CONNECTIONS)

        return img

    def locatePosition(self, img, handNum = 0, draw = True):
        xList = []
        yList = []
        boundingBox = []


        self.landMarksList = []

        if self.results.multi_hand_landmarks:
            hands = self.results.multi_hand_landmarks[handNum]

            for id, landMarks in enumerate(hands.landmark):
                height, width, channels = img.shape
                cx, cy = int(landMarks.x * width), int(landMarks.y * height)

                xList.append(cx)
                yList.append(cy)
                self.landMarksList.append([id, cx, cy])

                if draw == True and id == 8:
                    cv2.circle(img, (cx, cy), 5, (255, 255, 255), cv2.FILLED)

            xMin, xMax = min(xList), max(xList)
            yMin, yMax = min(yList), max(yList)
            boundingBox = xMin, yMin, xMax, yMax

            if draw == True:
                cv2.rectangle(img, (boundingBox[0] - 20, boundingBox[1] - 20 ),  (boundingBox[2] + 20, boundingBox[3] + 20), (0, 255, 0), 2)
        
        return self.landMarksList, boundingBox

    def calculateDistance(self, p1, p2, img, draw = True):
        x1, y1 = self.landMarksList[p1][1], self.landMarksList[p1][2]
        x2, y2 = self.landMarksList[p2][1], self.landMarksList[p2][2]
        
        lmx, lmy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw == True:
            cv2.circle(img, (x1, y1), 15, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, (lmx, lmy), 15, (255, 255, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2,y2), (255, 255, 255), 3)

        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, lmx, lmy]

    def fingersUp(self):
        fingers  = []

        if self.landMarksList[self.fingerTipID[0]][1] > self.landMarksList[self.fingerTipID[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.landMarksList[self.fingerTipID[id]][2] < self.landMarksList[self.fingerTipID[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


    def fps(self, img, prevTime, currTime, WIDTH, HEIGHT):
        # Calculates the FPS
        currTime = time.time()
        fps  = 1 / (currTime - prevTime)
        prevTime = currTime
        # cv2.putText(img, "FPS: " + str(int(fps)), (int(WIDTH - 200) , 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

        return prevTime, currTime

    def webcamResolution(self, capture):
        # Checks possible webcams
        url = "https://en.wikipedia.org/wiki/List_of_common_resolutions"
        table = pd.read_html(url)[0]
        table.columns = table.columns.droplevel()

        resolutions = {}

        for index, row in table[["W", "H"]].iterrows():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, row["W"])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, row["H"])
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            resolutions[str(width)+" x "+str(height)] = "OK"
    
        return resolutions

    def displayResolution(self, img, resolutions, WIDTH, HEIGHT):
        # cv2.putText(img, "Res: " + str(list(resolutions)[-1]), (int(WIDTH - 520) , 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
        pass

def main():
    # initialize weight for running average
    aWeight = 0.5
    prevTime = 0
    currTime = 0

    camera = cv2.VideoCapture(0)
    
    hand_detector = HandDetector(detectionConfidence = 0.8)
    WIDTH = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    HEIGHT = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    resolutions = [WIDTH, HEIGHT]

    # Volume variables
    vol = 0
    volBar = 400
    volPer = 0
    boundingBoxArea = 0

    top, right, bottom, left = 10, 350, 225, 590

    num_frames = 0
    start_recording = False

    while(True):
        WIDTH = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        HEIGHT = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # get the current frame
        (grabbed, frame) = camera.read()

        clone2 = frame.copy()
        img = clone2

        img = hand_detector.locateHands(img)
        landMarkList, boundingBox  = hand_detector.locatePosition(img)

        if len(landMarkList ) > 0:
            
        # prevTime, currTime = hand_detector.fps(img, prevTime, currTime, WIDTH, HEIGHT)
        # hand_detector.displayResolution(img, resolutions, WIDTH, HEIGHT)
            boundingBoxArea = (boundingBox[2] - boundingBox[0]) * (boundingBox[3] - boundingBox[1]) // 100
            if boundingBoxArea > 100 and boundingBoxArea < 500:

                # Distance between index and thumb
                length, img, lineDetails = hand_detector.calculateDistance(4, 8, img)
                # print(length)

                # Convert Volume 
                # vol = np.interp(length, [20, 200], [minVolume, maxVolume])
                volBar = np.interp(length, [50, 300], [400, 150])
                volPer = np.interp(length, [50, 300], [0, 100])

                # Makes Bar smooth
                smoothness = 10
                volPer = smoothness * round(volPer / smoothness)

                # Detect Fingers up
                fingers = hand_detector.fingersUp()
                # print(fingers)

                # Detect if Ring Finger is up
                if not fingers[4]:
                    volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                    cv2.circle(img, (lineDetails[4], lineDetails[5]), 15, (0, 255, 0), cv2.FILLED)
                    colorVol = (0, 255, 0)
                else:
                    colorVol = (255, 0, 0)
            cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3) 
            cVol = int(volume.GetMasterVolumeLevelScalar() * 100)

        '''
            Hand gesture detector
        '''
        frame = imutils.resize(frame, width = 700)

        frame = cv2.flip(frame, 1)

        clone = frame.copy()

        (height, width) = frame.shape[:2]

        roi = frame[top:bottom, right:left]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if num_frames < 30:
            print("Capturing background frame = " + str(num_frames+1))
            run_avg(gray, aWeight)
        else:
            hand = segment(gray)

            if hand is not None:
                (thresholded, segmented) = hand

                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                if start_recording:
                    cv2.imwrite('Temp.png', thresholded)
                    resizeImage('Temp.png')
                    predictedClass, confidence = getPredictedClass()
                    showStatistics(predictedClass, confidence)
                cv2.imshow("Thesholded", thresholded)

        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        num_frames += 1

        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord("q"):
            break

        if keypress == ord("s"):
            start_recording = True

        cv2.imshow("Video Feed - Volume Gesture", clone2)
        cv2.imshow("Video Feed", clone)

def getPredictedClass():
    image = cv2.imread('Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prediction = model.predict([gray_image.reshape(89, 100, 1)])
    return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2] + prediction[0][3] + prediction[0][4] + prediction[0][5]))

def changeGestureCount(number):
    global gesture_count
    global gesture_number

    if gesture_number != number:
        gesture_count = 1
        gesture_number = number
        return False
    gesture_count += 1
    if gesture_count >= 50:
        gesture_count = 0
        gesture_number = -1
        print("exectute")
        return True
    return False

def showStatistics(predictedClass, confidence):

    textImage = np.zeros((300,512,3), np.uint8)
    className = str(predictedClass)

    if predictedClass == 0:
        className = "Swing"
        if changeGestureCount(0):
            pass
    elif predictedClass == 1:
        className = "Palm"
        if changeGestureCount(1):
            pyautogui.press('playpause')
    elif predictedClass == 2:
        className = "Fist"
        if changeGestureCount(2):
            pass
    elif predictedClass == 3:
        className = "Thumb"
        if changeGestureCount(3):
            pass
    elif predictedClass == 4:
        if changeGestureCount(4):
            pyautogui.press('nexttrack')
        className = "Right Indicator"
    elif predictedClass == 5:
        if changeGestureCount(5):
            pyautogui.press('prevtrack')
        className = "Left Indicator"

    confidence_percentage = confidence * 100

    if (confidence_percentage < 98):
        className = "Nothing"
        confidence_percentage = 0

    cv2.putText(textImage,"Pedicted Class : " + className, 
        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )

    cv2.putText(textImage,"Confidence : " + str(confidence_percentage) + '%', 
        (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )
    cv2.imshow("Statistics", textImage)


# Model defined
tf.reset_default_graph()
convnet=input_data(shape=[None,89,100,1],name='input')
convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=fully_connected(convnet,1000,activation='relu')
convnet=dropout(convnet,0.75)

convnet=fully_connected(convnet,6,activation='softmax')

convnet=regression(convnet,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')

model=tflearn.DNN(convnet,tensorboard_verbose=0)

# Load Saved Model
model.load("TrainedModel/GestureRecogModel.tfl")

main()
