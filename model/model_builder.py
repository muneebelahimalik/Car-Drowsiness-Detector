from keras.models import Sequential
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D

def build_model(input_shape):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(1,1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(1,1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(1,1)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    return model
