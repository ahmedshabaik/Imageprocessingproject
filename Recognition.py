import cv2
import numpy as np
import os
import xlwt
from xlwt import Workbook
from datetime import datetime
import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter
id = 0
# names related to ids: example ==> Name1 : id=1,  etc
names = ['Name0', 'Name1', 'Name2', 'Name3', 'Name4', 'Name5']
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height
# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

workbook = Workbook('names.xls')
sheet1 = workbook.add_sheet('Sheet1')
row = 0
col = 0
sheet1.write(row, col, 'Names')
col = 1
sheet1.write(row, col, 'Date')
row = 1
while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)  # Flip vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    # , cell_overwrite_ok=True

    for (x, y, w, h) in faces:

        # row =+1
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # If confidence is less them 100 ==> "0" : perfect match
        if (confidence < 100):
            # row =row+1 #will save the name every time sample in a new row
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))

        else:
            # row =row+1
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        sheet1.write(row, 0, id)
        row = row + 1

        Datet = str(datetime.datetime.now())
        sheet1.write(row, 1, Datet)
        # row =+1

        workbook.save('names.xls')

        cv2.putText(
            img,
            str(id),
            (x + 5, y - 5),
            font,
            1,
            (255, 255, 255),
            2
        )
        cv2.putText(
            img,
            str(confidence),
            (x + 5, y + h - 5),
            font,
            1,
            (255, 255, 0),
            1
        )

    cv2.imshow('camera', img)
    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
