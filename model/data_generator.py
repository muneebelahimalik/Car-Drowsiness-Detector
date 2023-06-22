from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True, batch_size=1, target_size=(24,24), class_mode='categorical'):
    return gen.flow_from_directory(dir, batch_size=batch_size, shuffle=shuffle, color_mode='grayscale', class_mode=class_mode, target_size=target_size)

def get_generator_stats(train_batch, valid_batch, batch_size):
    SPE = len(train_batch.classes) // batch_size
    VS = len(valid_batch.classes) // batch_size
    return SPE, VS
