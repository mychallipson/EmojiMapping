import numpy as np
import cv2
import time
import dlib

cap = cv2.VideoCapture(0)
time_s = time.time()
counter = 0
max_frames = 120
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
while(cap.isOpened()):
    ret, frame = cap.read()
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray,1.03,3)
    # if len(faces) != 0:
    #     x,y,w,h = faces[0]
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #     roi_gray = gray[y:y + h, x:x + w]
    #     roi_color = frame[y:y + h, x:x + w]
    #     rect = dlib.rectangle(x, y, x + w, y + h)
    #     landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])
    #     for idx, point in enumerate(landmarks):
    #         pos = (point[0, 0], point[0, 1])
    #         cv2.circle(frame, pos, 3, color=(0, 255, 255))
    time_e = time.time()
    counter += 1
    sec = time_e - time_s
    fps = counter / sec
    if (counter > max_frames):
        counter = 0
        time_s = time.time()
    cv2.putText(frame, str(int(fps)), (0, frame.shape[0] - 1), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=1, color=(0, 0, 255), thickness=2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()