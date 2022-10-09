import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time
import autopy

camWidth, camHeight = 648, 488
frameR = 200
cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)
pTime = 0
detector = HandDetector(maxHands=1, detectionCon=0.7)
screenW, screenH = autopy.screen.size()
smoothening = 7
plocX, plocY = 0, 0
clocX, clocY = 0, 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    # HAND POSITION DETECTOR
    if len(hands) > 0:
        lmList1 = hands[0]["lmList"]  # List of 21 Landmark points
        x1 = lmList1[8][0]
        y1 = lmList1[8][1]
        #print("HAND1 : ", x1, y1)

    # FINGERS UP DETECTOR
        fingers = detector.fingersUp(hands[0])

        if fingers[1] == 1 and fingers[2] == 0:
            cv2.rectangle(img, (frameR, frameR), (camWidth - frameR, camHeight - frameR), (255, 0, 255), 2)
            x3 = np.interp(x1, (frameR, camWidth-frameR), (0, screenW))
            y3 = np.interp(y1, (frameR, camHeight-frameR), (0, screenH))
            #SMOOTHNESS ADDED
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            autopy.mouse.move(clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX,clocY

        #DISTANCE BETWEEN INDEX FINGER AND SECOND FINGER
        if fingers[1] == 1 and fingers[2] == 1:
            #dist = lmList1[12][0]-lmList1[8][0]
            #MOUSE CLICK
            cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
            autopy.mouse.click()
            time.sleep(0.2)


    # FPS COUNTER
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, "FPS : " + str(int(fps)), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    cv2.imshow("frame", img)

    # QUIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
