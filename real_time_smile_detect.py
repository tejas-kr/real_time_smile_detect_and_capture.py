import cv2
import time

face_cascade_path = "haarcascade_frontalface_default.xml"
smile_cascade_path = "haarcascade_smile.xml"

face_cascade = cv2.CascadeClassifier(face_cascade_path)
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)
video_capture.set(4, 480)

while True:
    ret, frame = video_capture.read()
    # image_capture = frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.05,
        minNeighbors = 22,
        minSize = (55, 55),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        smile = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor = 1.8,
            minNeighbors=30,
            minSize = (25, 25),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in smile:
            cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 1)
            cv2.imwrite(str(time.time()) + ".png", frame)
            exit(0)

    cv2.imshow('Real Smile Detect', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()



