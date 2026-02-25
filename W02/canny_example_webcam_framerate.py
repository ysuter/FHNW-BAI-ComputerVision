import cv2
import numpy as np
import time

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

# Create window and trackbars
cv2.namedWindow("Webcam | Canny Edges")
cv2.createTrackbar("Low Threshold",  "Webcam | Canny Edges", 100, 500, nothing)
cv2.createTrackbar("High Threshold", "Webcam | Canny Edges", 200, 500, nothing)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Read trackbar positions
    low  = cv2.getTrackbarPos("Low Threshold",  "Webcam | Canny Edges")
    high = cv2.getTrackbarPos("High Threshold", "Webcam | Canny Edges")

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, low, high)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    imstack = np.hstack((frame, edges_bgr))

    cv2.putText(imstack, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Webcam | Canny Edges", imstack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()