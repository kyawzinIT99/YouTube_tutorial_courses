import cv2
cap = cv2.VideoCapture(0)
detector = cv2.QRCodeDetector()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    data, bbox, _ = detector.detectAndDecode(frame)

    if bbox is not None:
        # bbox is float, convert to int
        bbox = bbox.astype(int)
        n = len(bbox)
        for i in range(n):
            pt1 = tuple(bbox[i][0])
            pt2 = tuple(bbox[(i + 1) % n][0])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        if data:
            print("QR Code detected:", data)
            cv2.putText(frame, data, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("QR Code Scanner", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()