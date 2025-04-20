import cv2
from mediapipe.python.solutions.hands import HAND_CONNECTIONS
from HandTracking import HandDetector

#Button Class
class Button:
    def __init__(self, pos, width, height, value):
        self.pos = pos
        self.width = width
        self.height = height
        self.value = value

    def draw(self, img):
        #Draw button box
        cv2.rectangle(img, self.pos,
                      (self.pos[0] + self.width, self.pos[1] + self.height),
                      (225, 225, 225), cv2.FILLED)
        #Border
        cv2.rectangle(img, self.pos,
                      (self.pos[0] + self.width, self.pos[1] + self.height),
                      (50, 50, 50), 3)
        #Text
        cv2.putText(img, self.value,
                    (self.pos[0] + 30, self.pos[1] + 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (50, 50, 50), 3)

    def isClicked(self, x, y):
        if self.pos[0] < x < self.pos[0] + self.width and \
           self.pos[1] < y < self.pos[1] + self.height:
            return True
        return False

#Setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)

#Calculator UI buttons
buttonValues = [['7', '8', '9', '/'],
                ['4', '5', '6', '*'],
                ['1', '2', '3', '-'],
                ['0', '.', '=', '+']]

buttons = []
for y in range(4):
    for x in range(4):
        xpos = x * 100 + 800
        ypos = y * 100 + 150
        buttons.append(Button((xpos, ypos), 100, 100, buttonValues[y][x]))

#Variables
equation = ""
delayCounter = 0

#Main Loop
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    #Find hands
    hands, img = detector.findHands(img)

    #Draw all buttons
    for button in buttons:
        button.draw(img)

    #Display equation
    cv2.rectangle(img, (800, 70), (1200, 150), (255, 255, 255), cv2.FILLED)
    cv2.rectangle(img, (800, 70), (1200, 150), (50, 50, 50), 3)
    cv2.putText(img, equation, (810, 130), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 0, 0), 3)

    #Hand logic
    if hands:
        hand = hands[0]
        lmList = hand['lmList']
        bbox = hand['bbox']

        x, y = lmList[8]  # Index finger tip

        # Draw bounding box
        cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                      (bbox[2] + 20, bbox[3] + 20),
                      (0, 255, 0), 3)

        # Draw lines (skeleton) using connections from MediaPipe
        from mediapipe.python.solutions.hands import HAND_CONNECTIONS

        for connection in HAND_CONNECTIONS:
            start = lmList[connection[0]]
            end = lmList[connection[1]]
            cv2.line(img, start, end, (0, 255, 255), 2)

        # Draw landmarks and coordinates
        for idx, point in enumerate(lmList):
            cv2.circle(img, point, 5, (0, 0, 255), cv2.FILLED)

        # Detect click (index and middle finger close)
        length, _, _ = detector.findDistance(lmList[8], lmList[12], img, draw=False)

        if length < 40 and delayCounter == 0:
            for button in buttons:
                if button.isClicked(x, y):
                    value = button.value
                    if value == "=":
                        try:
                            equation = str(eval(equation))
                        except:
                            equation = "Error"
                    else:
                        equation += value
                    delayCounter = 1

    #Delay to avoid multi-clicks
    if delayCounter != 0:
        delayCounter += 1
        if delayCounter > 10:
            delayCounter = 0

    #Show the frame
    cv2.imshow("Virtual Calculator", img)
    key = cv2.waitKey(1)
    if key == ord('c'):
        equation = ""
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
