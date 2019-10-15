import numpy as np
import cv2
import glob

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")


img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "pictures\{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        break

cam.release()

cv2.destroyAllWindows()
dumb_face = cv2.imread('dumb_face.png')
print('hello')

faceDet = cv2.CascadeClassifier("data\haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("data\haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("data\haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("data\haarcascade_frontalface_alt_tree.xml")


files = glob.glob("pictures\\*")  # Get list of all images with emotion

frame = cv2.imread(files[0])  # Open image
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                flags=cv2.CASCADE_SCALE_IMAGE)
face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                        flags=cv2.CASCADE_SCALE_IMAGE)
face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                            flags=cv2.CASCADE_SCALE_IMAGE)
face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          flags=cv2.CASCADE_SCALE_IMAGE)

if len(face) == 1:
    facefeatures = face
elif len(face_two) == 1:
    facefeatures = face_two
elif len(face_three) == 1:
    facefeatures = face_three
elif len(face_four) == 1:
    facefeatures = face_four
else:
    facefeatures = ""
    print('no face detected')



x, y, w, h = facefeatures[0]
cropped_face = frame[y:y + h, x:x + w]
dumb_face = cv2.resize(dumb_face,(cropped_face.shape[0],cropped_face.shape[1]))

from copy import deepcopy
new_frame = frame.copy()
new_frame[y:y + h, x:x + w] = dumb_face
cv2.imwrite('created.jpg',new_frame)