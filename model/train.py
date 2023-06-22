import os

from keras.models import load_model

from data_generator import generator, get_generator_stats
from model_builder import build_model

BS = 32
TS = (24, 24)
train_batch = generator('data/train', shuffle=True, batch_size=BS, target_size=TS)
valid_batch = generator('data/valid', shuffle=True, batch_size=BS, target_size=TS)
SPE, VS = get_generator_stats(train_batch, valid_batch, batch_size=BS)
print(SPE, VS)

model = build_model(input_shape=(24, 24, 1))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batch, validation_data=valid_batch, epochs=15, steps_per_epoch=SPE, validation_steps=VS)
model.save('models/cnnCat2.h5', overwrite=True)
