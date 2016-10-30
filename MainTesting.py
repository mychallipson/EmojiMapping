import os
import cv2
import dlib
import numpy as np
import time

emotions_tree = os.walk('Emotion')
images_tree = os.walk('cohn-kanade-images')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')



test_max = 10
# svm_params = dict(kernel_type = cv2.ml.SVM_LINEAR,
#                   svm_type = cv2.ml.SVM_C_SVC,
#                   C=2.67, gamma=5.383)
t_train_start = time.time()
existing_labels = []
label_locations = []
for root,dirs,files in emotions_tree:
    for file in files:
        label_locations.append((root, file))
        existing_labels.append(file[:17])

image_locations = []
for root,dirs,files in images_tree:
    for file in files:
        if file[:17] in existing_labels:
            image_locations.append((root, file))

# for path, file in label_locations:
#     print path, file
#
# for path, file in image_locations:
#     print path, file
#
# print len(label_locations)
# print len(image_locations)

def getEmotion(val):
    return{
        0 : 'neutral.png',
        1 : 'angry.png',
        2 : 'contempt.png',
        3 : 'disgust.png',
        4 : 'fear.png',
        5 : 'happy.png',
        6 : 'sadness.png',
        7 : 'surprise.png',
    }[val]

neutral = cv2.imread('Emojis/neutral.png')
angry = cv2.imread('Emojis/angry.png')
contempt = cv2.imread('Emojis/contempt.png')
disgust = cv2.imread('Emojis/disgust.png')
fear = cv2.imread('Emojis/fear.png')
happy = cv2.imread('Emojis/happy.png')
sadness = cv2.imread('Emojis/sadness.png')
surprise = cv2.imread('Emojis/surprise.png')
def getEmoji(val):
    return{
        0 : neutral,
        1 : angry,
        2 : contempt,
        3 : disgust,
        4 : fear,
        5 : happy,
        6 : sadness,
        7 : surprise,
    }[val]


def distanceFromCenter(point):
    return np.linalg.norm(np.array(point) - np.array((.5,.5)))

def getDistances(points,width,height):
    normalized_distances = []
    for point in points:
        norm_cur = (float(point[0,0])/float(width),float(point[0,1])/float(height))
        normalized_distances.append(distanceFromCenter(norm_cur))
    return np.float32(normalized_distances)

def normalize(points,width,height):
    normalized_points = []
    for point in points:
        norm_cur = (float(point[0, 0]) / float(width), float(point[0, 1]) / float(height))
        normalized_points.append(norm_cur)
    return np.float32(list(sum(normalized_points,())))


train_labels = []
count = 0
for path,file in label_locations:
    # if count > test_max:
    #     break
    f = open(os.path.join(path,file))
    val = f.read()
    train_labels.append(float(val))
    count += 1

train_data = []
counter = 0
for path, file in image_locations:
    # if counter > test_max:
    #     break
    img = cv2.imread(os.path.join(path,file))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray,1.3,3)
    if len(face) != 0:
        x,y,w,h = face[0]
        rect = dlib.rectangle(x,y,x+w,y+h)
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rect).parts()])
        #norm = getDistances(landmarks,img.shape[1],img.shape[0])
        norm = normalize(landmarks,img.shape[1],img.shape[0])
        train_data.append(norm)
    else:
        print train_labels.pop(counter)
        counter -=1
    counter += 1



# for val in train_labels:
#     print val

print len(train_labels)
print len(train_data)

train_data_mat = np.float32(train_data).reshape(len(train_data),len(train_data[0]))
train_labels_mat = np.int32(train_labels).reshape(-1,len(train_labels))


# print train_data_mat
# print train_labels_mat

svm = cv2.ml.SVM_create()
svm.setGamma(100)
svm.setC(100)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(train_data_mat, cv2.ml.ROW_SAMPLE, train_labels_mat)
svm.save('svm_data.dat')

t_train_end = time.time()
print 'Time taken to train {}'.format(t_train_end - t_train_start)

# results = svm.predict(train_data_mat)
#
# correct = 0
# count = 0
# for val in results[1]:
#     t1 = train_labels[count]
#     t2 = val[0]
#     if t1 == t2:
#         correct += 1
#     count += 1
#
# print float(correct)/len(results[1])

# t_test_start = time.time()
#
# img = cv2.imread('happy-woman.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# face = face_cascade.detectMultiScale(gray,1.3,5)[0]
# x, y, w, h = face
# cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
# roi_gray = gray[y:y + h, x:x + w]
# roi_color = img[y:y + h, x:x + w]
# rect = dlib.rectangle(x,y,x+w,y+h)
# landmarks = np.matrix([[p.x,p.y] for p in predictor(img,rect).parts()])
# vals = getDistances(landmarks,img.shape[1],img.shape[0])
# test_data = np.float32(vals).reshape(-1,len(vals))
# result = svm.predict(test_data)
# emotion = result[1]
# emotion = emotion[0]
# print 'Guessed: {0} Actual: {1}'.format(getEmotion(int(emotion[0])),'happy')
#
# img = cv2.imread('angry.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# face = face_cascade.detectMultiScale(gray,1.3,5)[0]
# x, y, w, h = face
# cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
# roi_gray = gray[y:y + h, x:x + w]
# roi_color = img[y:y + h, x:x + w]
# rect = dlib.rectangle(x,y,x+w,y+h)
# landmarks = np.matrix([[p.x,p.y] for p in predictor(img,rect).parts()])
# vals = getDistances(landmarks,img.shape[1],img.shape[0])
# test_data = np.float32(vals).reshape(-1,len(vals))
# result = svm.predict(test_data)
# emotion = result[1]
# emotion = emotion[0]
# print 'Guessed: {0} Actual: {1}'.format(getEmotion(int(emotion[0])),'angry')
#
# img = cv2.imread('contempt.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# face = face_cascade.detectMultiScale(gray,1.3,5)[0]
# x, y, w, h = face
# cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
# roi_gray = gray[y:y + h, x:x + w]
# roi_color = img[y:y + h, x:x + w]
# rect = dlib.rectangle(x,y,x+w,y+h)
# landmarks = np.matrix([[p.x,p.y] for p in predictor(img,rect).parts()])
# vals = getDistances(landmarks,img.shape[1],img.shape[0])
# test_data = np.float32(vals).reshape(-1,len(vals))
# result = svm.predict(test_data)
# emotion = result[1]
# emotion = emotion[0]
# print 'Guessed: {0} Actual: {1}'.format(getEmotion(int(emotion[0])),'contempt')
#
# img = cv2.imread('disgust.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# face = face_cascade.detectMultiScale(gray,1.3,5)[0]
# x, y, w, h = face
# cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
# roi_gray = gray[y:y + h, x:x + w]
# roi_color = img[y:y + h, x:x + w]
# rect = dlib.rectangle(x,y,x+w,y+h)
# landmarks = np.matrix([[p.x,p.y] for p in predictor(img,rect).parts()])
# vals = getDistances(landmarks,img.shape[1],img.shape[0])
# test_data = np.float32(vals).reshape(-1,len(vals))
# result = svm.predict(test_data)
# emotion = result[1]
# emotion = emotion[0]
# print 'Guessed: {0} Actual: {1}'.format(getEmotion(int(emotion[0])),'disgust')
#
# img = cv2.imread('fear.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# face = face_cascade.detectMultiScale(gray,1.3,5)[0]
# x, y, w, h = face
# cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
# roi_gray = gray[y:y + h, x:x + w]
# roi_color = img[y:y + h, x:x + w]
# rect = dlib.rectangle(x,y,x+w,y+h)
# landmarks = np.matrix([[p.x,p.y] for p in predictor(img,rect).parts()])
# vals = getDistances(landmarks,img.shape[1],img.shape[0])
# test_data = np.float32(vals).reshape(-1,len(vals))
# result = svm.predict(test_data)
# emotion = result[1]
# emotion = emotion[0]
# print 'Guessed: {0} Actual: {1}'.format(getEmotion(int(emotion[0])),'fear')
#
# img = cv2.imread('sad.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# face = face_cascade.detectMultiScale(gray,1.3,5)[0]
# x, y, w, h = face
# cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
# roi_gray = gray[y:y + h, x:x + w]
# roi_color = img[y:y + h, x:x + w]
# rect = dlib.rectangle(x,y,x+w,y+h)
# landmarks = np.matrix([[p.x,p.y] for p in predictor(img,rect).parts()])
# vals = getDistances(landmarks,img.shape[1],img.shape[0])
# test_data = np.float32(vals).reshape(-1,len(vals))
# result = svm.predict(test_data)
# emotion = result[1]
# emotion = emotion[0]
# print 'Guessed: {0} Actual: {1}'.format(getEmotion(int(emotion[0])),'sad')
#
# img = cv2.imread('surprise.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# face = face_cascade.detectMultiScale(gray,1.3,5)[0]
# x, y, w, h = face
# cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
# roi_gray = gray[y:y + h, x:x + w]
# roi_color = img[y:y + h, x:x + w]
# rect = dlib.rectangle(x,y,x+w,y+h)
# landmarks = np.matrix([[p.x,p.y] for p in predictor(img,rect).parts()])
# vals = getDistances(landmarks,img.shape[1],img.shape[0])
# test_data = np.float32(vals).reshape(-1,len(vals))
# result = svm.predict(test_data)
# emotion = result[1]
# emotion = emotion[0]
# print 'Guessed: {0} Actual: {1}'.format(getEmotion(int(emotion[0])),'surprised')
#
# t_test_end = time.time()
# print 'Time taken to predict 7 faces: {}'.format(t_test_end-t_test_start)
# epath = 'Emojis/'


time_s = time.time()
counter = 0
max_frames = 30
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    eframe = frame.copy()
    faces = face_cascade.detectMultiScale(gray,1.2,2)
    if len(faces) != 0:
        x,y,w,h = faces[0]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        rect = dlib.rectangle(x, y, x + w, y + h)
        landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.circle(frame, pos, 3, color=(0, 255, 255))
        #vals = normalize(landmarks, img.shape[1], img.shape[0])
        vals = normalize(landmarks, img.shape[1], img.shape[0])
        test_data = np.float32(vals).reshape(-1, len(vals))
        result = svm.predict(test_data)
        emotion = result[1]
        emotion = emotion[0]
        emoji = getEmoji(int(emotion[0]))
        ycenter = y + h/2 - emoji.shape[0]/2
        xcenter = x + w/ 2 - emoji.shape[1] / 2
        try:
            eframe[ycenter:ycenter+emoji.shape[0],xcenter:xcenter+emoji.shape[1]] = emoji
        except ValueError:
            pass

    time_e = time.time()
    counter += 1
    sec = time_e - time_s
    fps = counter / sec
    if(counter > max_frames):
        counter = 0
        time_s = time.time()
    cv2.putText(frame, str(int(fps)), (0,frame.shape[0]-1), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=1, color=(0, 0, 255),thickness=2)
    cv2.putText(eframe, str(int(fps)), (0,eframe.shape[0]-1), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=1, color=(0, 0, 255),thickness=2)
    cv2.namedWindow('Regular Feed')
    cv2.imshow('Regular Feed',frame)
    cv2.namedWindow("Emoji Feed")
    cv2.imshow('Emoji Feed', eframe)

    #print fps
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break