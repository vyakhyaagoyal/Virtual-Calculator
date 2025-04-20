import cv2
import mediapipe as mp
import math

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,  #Set to False for video input (Dynamic)
            max_num_hands=self.maxHands,  #Maximum number of hands to detect
            min_detection_confidence=self.detectionCon,  #Confidence for detection
            min_tracking_confidence=self.trackCon  #Confidence for tracking
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                hand = {}
                lmList = []
                xList = []
                yList = []

                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append((cx, cy))
                    xList.append(cx)
                    yList.append(cy)

                # Bounding box
                xMin, xMax = min(xList), max(xList)
                yMin, yMax = min(yList), max(yList)
                bbox = xMin, yMin, xMax, yMax

                hand['lmList'] = lmList
                hand['bbox'] = bbox
                allHands.append(hand)

        return allHands, img

    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = p1
        x2, y2 = p2

        length = math.hypot(x2 - x1, y2 - y1)

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED)

        return length, [x1, y1, x2, y2], img
