import cv2
from lane_detection import detect_lanes

cap = cv2.VideoCapture("data/videos/test.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    lane_frame = detect_lanes(frame)
    cv2.imshow("Lane Detection", lane_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

