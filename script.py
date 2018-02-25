import cv2

# import the classifier
faceCascade = cv2.CascadeClassifier('haar-frontface.xml')


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, image = cap.read()

    # convert to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect all faces
    detected = faceCascade.detectMultiScale(gray, 1.3, 3)

    # loop over the faces and draw a rect. and label for each one
    faces = 0
    for (x, y, w, h) in detected:
        cv2.putText(image, "face", (x, y - 10), 5, 1.0, (0, 255, 0), 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        faces += 1

    # write how many faces are found
    cv2.putText(image, "There are {} faces in the image".format(faces), (50, 0 + 50), 5, 2.0, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

