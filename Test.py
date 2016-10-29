import numpy as np
import cv2
import dlib
import time

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_mouth.xml')
t0 = time.time()
img = cv2.imread('happy-woman.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = gray
faces = face_cascade.detectMultiScale(gray,1.03,5)
# svm = cv2.ml.load('svm_data.dat')

# def distanceFromCenter(point):
#     return np.linalg.norm(np.array(point) - np.array((.5,.5)))
#
# def getDistances(points,width,height):
#     normalized_distances = []
#     for point in points:
#         norm_cur = (float(point[0,0])/float(width),float(point[0,1])/float(height))
#         normalized_distances.append(distanceFromCenter(norm_cur))
#     return np.float32(normalized_distances)


for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    rect = dlib.rectangle(x,y,x+w,y+h)
    landmarks = np.matrix([[p.x,p.y] for p in predictor(img,rect).parts()])
    # vals = getDistances(landmarks,img.shape[1],img.shape[0])
    # test_data = np.float32(vals).reshape(1,len(vals))
    # result = svm.predict(test_data)
    # print result
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.circle(img, pos, 3, color=(0, 255, 255))
    # eyes = eye_cascade.detectMultiScale(roi_gray,1.1,10)
    # mouths = mouth_cascade.detectMultiScale(roi_gray,1.5,20)
    # for (ex, ey, ew, eh) in eyes:
    #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    # for (mx, my, mw, mh) in mouths:
    #     cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
t1 = time.time()
print(t1-t0)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
