# -*- coding: utf-8 -*-
import cv2

# To capture a video, you need to create a VideoCapture object. Its argument can be either the device index or the
# name of a video file. Device index is just the number to specify which camera. Normally one camera will be
# connected (as in my case). So I simply pass 0 (or -1)

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('https://192.168.43.1:8080/video')

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()  # Since 2 values are provided

    if ret:
        faces = classifier.detectMultiScale(frame)
        for face in faces:
            x, y, w, h = face
            # here its red border --> as its cv2 think?
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 4)
            
        cv2.imshow("My window", frame)

        key = cv2.waitKey(25)

        if key == ord("q"):
            break
    
cap.release()
cv2.destroyAllWindows()
