import cv2
import numpy as np
import cvzone
from cvzone.PoseModule import PoseDetector
import math

# Open live camera
cap = cv2.VideoCapture(0)

#cap = cv2.VideoCapture('vid1.mp4')
pd = PoseDetector(trackCon=0.3, detectionCon=0.3)

counter = 0   # push-up count
direction = 0 # 0 = down, 1 = up


def angles(lmlist, p1, p2, p3, p4, p5, p6, drawpoints, img):
    """
    Calculate arm angles, show progress bars, and count push-ups.
    """
    global counter, direction

    if len(lmlist) != 0:
        # Get landmark coordinates
        x1, y1 = lmlist[p1][1], lmlist[p1][2]
        x2, y2 = lmlist[p2][1], lmlist[p2][2]
        x3, y3 = lmlist[p3][1], lmlist[p3][2]
        x4, y4 = lmlist[p4][1], lmlist[p4][2]
        x5, y5 = lmlist[p5][1], lmlist[p5][2]
        x6, y6 = lmlist[p6][1], lmlist[p6][2]

        # Draw points & lines
        if drawpoints:
            points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6)]
            for px, py in points:
                cv2.circle(img, (px, py), 10, (255, 0, 255), 5)

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 6)
            cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 6)
            cv2.line(img, (x4, y4), (x5, y5), (0, 0, 255), 6)
            cv2.line(img, (x5, y5), (x6, y6), (0, 0, 255), 6)
            cv2.line(img, (x1, y1), (x4, y4), (0, 0, 255), 6)

        # Calculate angles for both arms
        lefthandangle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        righthandangle = math.degrees(math.atan2(y6 - y5, x6 - x5) - math.atan2(y4 - y5, x4 - x5))

        # Normalize
        lefthandangle = abs(lefthandangle)
        righthandangle = abs(righthandangle)

        # Map angles to percentage (0–100)
        left_perc = np.interp(lefthandangle, (60, 160), (0, 100))
        right_perc = np.interp(righthandangle, (60, 160), (0, 100))

        # Count push-ups
        if left_perc > 90 and right_perc > 90 and direction == 0:
            counter += 0.5
            direction = 1
        if left_perc < 10 and right_perc < 10 and direction == 1:
            counter += 0.5
            direction = 0

        # Show counter
        cv2.putText(img, f'Count: {int(counter)}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Draw progress bars
        cv2.rectangle(img, (50, 100), (100, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(400 - left_perc * 3)), (100, 400), (0, 255, 0), -1)

        cv2.rectangle(img, (900, 100), (950, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (900, int(400 - right_perc * 3)), (950, 400), (255, 0, 0), -1)


# Main loop
while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    img = cv2.resize(img, (1000, 500))
    cvzone.putTextRect(img, 'AI Push Up Counter', [345, 30],
                       thickness=2, border=2, scale=2.5)

    # Pose detection
    img = pd.findPose(img, draw=True)
    lmlist, bbox = pd.findPosition(img, draw=True, bboxWithHands=False)

    if len(lmlist) != 0:
        angles(lmlist, 11, 13, 15, 12, 14, 16, drawpoints=True, img=img)

    cv2.imshow('Live Push-up Counter', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()