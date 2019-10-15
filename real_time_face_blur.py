import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("Real-time Face Blur")

dumb_face = cv2.imread('dumb_face.png')

faceDet = cv2.CascadeClassifier("data\haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("data\haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("data\haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("data\haarcascade_frontalface_alt_tree.xml")

img_counter = 0

while True:
    ret, frame = cam.read()

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

    if (facefeatures != ''):
        x, y, w, h = facefeatures[0]
        cropped_face = frame[y:y + h, x:x + w]
        frame[y:y + h, x:x + w] = cv2.blur(cropped_face, (60,60))


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
        img_name = "pictures\{}.jpg".format(int(3))
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))

        break

cam.release()

cv2.destroyAllWindows()

print('Bye')


