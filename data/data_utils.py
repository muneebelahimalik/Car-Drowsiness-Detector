import cv2
import os
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_eye = cv2.resize(gray, (24, 24))
    normalized_eye = resized_eye / 255.0
    reshaped_eye = normalized_eye.reshape(24, 24, -1)
    expanded_eye = np.expand_dims(reshaped_eye, axis=0)
    return expanded_eye

def draw_rectangle(frame, x, y, w, h, color, thickness):
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)

def draw_text(frame, text, x, y, font, font_scale, color, thickness, line_type):
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, line_type)

def save_image(frame, path):
    cv2.imwrite(path, frame)
