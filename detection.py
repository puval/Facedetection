import cv2
from random import randrange
#Load some pre-trained data on the face frontals from opencv(haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#img = cv2.imread('animw.jpg',0)
webcam = cv2.VideoCapture(0)

while True:
    successfull_frame_read, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )
    face_coorditnates = trained_face_data.detectMultiScale(gray)
    #draw rectangle around the face
    for (x, y, w, h) in face_coorditnates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(256),randrange(256), randrange(256)),randrange(40))
    cv2.imshow('Webcam', frame)
    key = cv2.waitKey(1)
    if key==81 or key==113:
        break
    
webcam.release()





