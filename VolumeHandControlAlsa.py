from time import time
from weakref import ref
import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
import alsaaudio

# wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
# cap.set(3, wCam)
# cap.set(4, hCam)
pTime = 0
volume = 0
length = 0

detector = htm.HandDetector(detectionConf=0.9)


# audio stuff
mixer  = alsaaudio.Mixer()
length = np.interp(mixer.getvolume()[1], [0,100], [50, 300])
volume = np.interp(length, [50,300], [0,100])

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if len(lmList) != 0:

        # getting landmarks coordinates
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        x3, y3 = lmList[12][1], lmList[12][2]
        x4, y4 = lmList[3][1], lmList[3][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        # drawing elements
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x4, y4), 10, (255, 255, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        # checking if volume change is enabled
        len_8_12 = math.hypot(x3-x2, y3-y2)
        if len_8_12 < 40:
            length = math.hypot(x2-x1, y2-y1)
            #print(length)

            # calculatin reference length
            ref_len = math.hypot(x4-x1, y4-y1)
            length = int(length*65/ref_len)
            print(ref_len, length)

            # Hand range: 50 - 300
            # Volume range 0 - 100
            volume = np.interp(length, [50, 300], [0,100])
            # print(int(volume))
            mixer.setvolume(int(volume))

            if length < 30:
                cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (40, 150), (85, 400), (255, 255, 255), 2)
    # cv2.rectangle(img, (40, 400 - int(volume*2.5)), (85, 400), (255, 255, 255), cv2.FILLED)
    volBar = np.interp(length, [50,300], [400, 150])
    cv2.rectangle(img, (40, int(volBar)), (85, 400), (255, 255, 255), cv2.FILLED)

    cv2.putText(img, f"{int(volume)} %", (30, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Img", img)
    cv2.waitKey(1)