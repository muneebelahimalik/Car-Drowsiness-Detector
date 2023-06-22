import cv2
import os
from keras.models import load_model
import numpy as np
import time

from data_loader import load_cascade_classifiers, load_sound
from data_utils import preprocess_image, draw_rectangle, draw_text, save_image

face_cascade, leye_cascade, reye_cascade = load_cascade_classifiers()
sound = load_sound()

lbl = ['Close', 'Open']

model = load_model('models/cnncat2.h5')
path = os.getcwd()
ip = '10.7.108.100'  # Replace with the IP address of your smartphone
port = '4747'  # Replace with the port number used by the streaming server
url = f'http://{ip}:{port}/video'
cap = cv2.VideoCapture(url)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye_cascade.detectMultiScale(gray)
    right_eye = reye_cascade.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        draw_rectangle(frame, x, y, w, h, (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        draw_rectangle(frame, x, y, w, h, (100, 50, 250), 1)
        r_eye = frame[y:y+h, x:x+w]
        count += 1
        r_eye = preprocess_image(r_eye)
        rrpred = model.predict(r_eye)
        rpred = min(rrpred)
        if min(rpred) > 0.005:
            lbl = 'Open'
        elif min(rpred) <= 0.005:
            lbl = 'Closed'
        else:
            break

    for (x, y, w, h) in left_eye:
        draw_rectangle(frame, x, y, w, h, (100, 50, 250), 1)
        l_eye = frame[y:y+h, x:x+w]
        count += 1
        l_eye = preprocess_image(l_eye)
        llpred = model.predict(l_eye)
        lpred = min(llpred)
        if min(lpred) > 0.005:
            lbl = 'Open'
        elif min(lpred) <= 0.005:
            lbl = 'Closed'
        else:
            break

    if min(rpred) <= 0.005 and min(lpred) <= 0.005:
        score += 1
        draw_text(frame, "Closed", 10, height-20, font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        draw_text(frame, "Eyes Open", 10, height-20, font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0
    draw_text(frame, 'Score:' + str(score), 150, height-20, font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 15:
        save_image(frame, os.path.join(path, 'image.jpg'))
        try:
            sound.play()
        except:
            pass
        if thicc < 16:
            thicc += 2
        else:
            thicc -= 2
            if thicc < 2:
                thicc = 2
        draw_rectangle(frame, 0, 0, width, height, (0, 0, 255), thicc)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
