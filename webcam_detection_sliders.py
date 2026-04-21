import cv2
from ultralytics import YOLO

WINDOW = "YOLO Webcam Detection"

# Choices: yolo<version><model size: n, s, m, l, x>.pt
# E.g., yolo11x.pt
model = YOLO("yolov8m.pt")

# Create window first so trackbars can attach to it
cv2.namedWindow(WINDOW)
cv2.createTrackbar("Conf %",  WINDOW, 25, 100, lambda _: None)
cv2.createTrackbar("IOU %",   WINDOW, 45, 100, lambda _: None)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    conf = cv2.getTrackbarPos("Conf %", WINDOW) / 100
    iou  = cv2.getTrackbarPos("IOU %",  WINDOW) / 100

    results = model(frame, conf=max(conf, 0.01), iou=max(iou, 0.01), verbose=False)
    annotated = results[0].plot()

    cv2.imshow(WINDOW, annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
