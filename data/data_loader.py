import cv2
from pygame import mixer

mixer.init()
sound = mixer.Sound('alarm.wav')

def load_cascade_classifiers():
    face_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
    leye_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
    reye_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

    return face_cascade, leye_cascade, reye_cascade

def load_sound():
    return mixer.Sound('alarm.wav')
