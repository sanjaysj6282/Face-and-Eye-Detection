import cv2

cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_eye.xml")

while True:
    rel, image = cap.read()

    if rel:
        faces = classifier.detectMultiScale(image)
        for face in faces:
            x, y, w, h = face
            image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255))

        cv2.imshow("Eye detection", image)

        key = cv2.waitKey(10)

        if key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
